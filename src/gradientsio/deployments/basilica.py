from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import Any

import httpx

from gradientsio.constants import BASILICA_API_KEY_ENV
from gradientsio.constants import DEFAULT_BASILICA_CPU
from gradientsio.constants import DEFAULT_BASILICA_GPU_MODELS
from gradientsio.constants import DEFAULT_BASILICA_MEMORY
from gradientsio.constants import DEFAULT_BASILICA_MIN_GPU_MEMORY_GB
from gradientsio.constants import DEFAULT_BASILICA_PORT
from gradientsio.constants import DEFAULT_BASILICA_TIMEOUT
from gradientsio.constants import DEFAULT_BASILICA_TTL_SECONDS
from gradientsio.constants import DEFAULT_VLLM_MAX_LORA_RANK
from gradientsio.constants import RUNPOD_VLLM_IMAGE_NAME
from gradientsio.deployments.common import _build_vllm_env
from gradientsio.deployments.common import _build_vllm_start_cmd
from gradientsio.deployments.common import _default_gpu_count
from gradientsio.deployments.common import _default_model_name
from gradientsio.deployments.common import _deployment_failure_message
from gradientsio.deployments.common import _deployment_key
from gradientsio.deployments.common import _log_deployment_url
from gradientsio.deployments.common import _raise_provider_error
from gradientsio.deployments.common import _resolve_model_repos
from gradientsio.deployments.common import _Spinner
from gradientsio.deployments.common import logger
from gradientsio.errors import APIError
from gradientsio.errors import ConfigurationError
from gradientsio.errors import NetworkError
from gradientsio.errors import TaskTimeout
from gradientsio.models import BasilicaDeploymentModel
from gradientsio.models import DeploymentDetails
from gradientsio.models import DeploymentProvider
from gradientsio.models import DeploymentStatus
from gradientsio.sampling import RemoteVLLMSampler


if TYPE_CHECKING:
    from gradientsio.deployments.client import DeploymentClient


class BasilicaDeployment:
    def __init__(
        self,
        deployments: DeploymentClient,
        details: DeploymentDetails,
        *,
        model_name: str,
        inference_api_key: str | None = None,
    ) -> None:
        self._deployments = deployments
        self.details = details
        self.model_name = model_name
        self.inference_api_key = inference_api_key

    @property
    def id(self) -> str:
        return self.details.id

    @property
    def deployment_key(self) -> str:
        return self.details.deployment_key

    @property
    def server_url(self) -> str:
        return self.details.server_url

    def refresh(self) -> DeploymentDetails:
        self.details = self._deployments.get_basilica(self.id, deployment_key=self.deployment_key)
        return self.details

    def status(self) -> DeploymentStatus | str:
        return self.refresh().status

    def delete(self) -> None:
        self._deployments.delete_basilica(self.id)

    def logs(self, *, tail: int | None = None) -> str:
        return self._deployments.basilica_logs(self.id, tail=tail)

    def sampler(self, *, timeout: float = 60.0, verify_ssl: bool = True) -> RemoteVLLMSampler:
        return RemoteVLLMSampler(
            base_url=self.server_url,
            model=self.model_name,
            api_key=self.inference_api_key,
            timeout=timeout,
            verify_ssl=verify_ssl,
        )

    def wait_ready(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float = 15.0,
    ) -> DeploymentDetails:
        started_at = time.monotonic()
        logger.warning("Deploying on Basilica now. This may take a few minutes, please grab a coffee.")
        spinner = _Spinner("Waiting for Basilica vLLM server to become ready")

        while True:
            details = self.refresh()
            if details.is_running and self._deployments.is_basilica_endpoint_ready(self.server_url):
                spinner.finish("Basilica deployment is ready.")
                return details

            if details.is_terminal:
                spinner.finish()
                logs = self._deployments.basilica_logs(self.id, tail=80)
                message = f"Basilica deployment {self.id} reached terminal status {details.status}"
                if logs:
                    message = f"{message}\nRecent logs:\n{_log_excerpt(logs)}"
                raise APIError(message)

            failure_message = _deployment_failure_message(details)
            if failure_message:
                spinner.finish()
                raise APIError(failure_message)

            if timeout is not None and time.monotonic() - started_at >= timeout:
                spinner.finish()
                raise TaskTimeout(f"Basilica deployment {self.id} was not ready within {timeout} seconds")

            sleep_for = poll_interval
            if timeout is not None:
                remaining = timeout - (time.monotonic() - started_at)
                sleep_for = max(0.0, min(poll_interval, remaining))
            spinner.sleep(sleep_for)


class BasilicaDeploymentMixin:
    def deploy_basilica(
        self,
        *,
        base_model: str,
        lora: str | None = None,
        deployment_model_name: str | None = None,
        hf_token: str | None = None,
        env: dict[str, Any] | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | str | None = None,
        dtype: str | None = None,
        trust_remote_code: bool | None = None,
        enforce_eager: bool = False,
        max_lora_rank: int | None = DEFAULT_VLLM_MAX_LORA_RANK,
        gpu_count: int | None = None,
        port: int = DEFAULT_BASILICA_PORT,
        gpu_models: list[str] | None = None,
        min_gpu_memory_gb: int = DEFAULT_BASILICA_MIN_GPU_MEMORY_GB,
        cpu: str = DEFAULT_BASILICA_CPU,
        memory: str = DEFAULT_BASILICA_MEMORY,
        ttl_seconds: int | None = DEFAULT_BASILICA_TTL_SECONDS,
        timeout: int = DEFAULT_BASILICA_TIMEOUT,
        deployment_name: str | None = None,
        wait: bool = True,
        reuse_existing: bool = True,
        inference_api_key: str | None = None,
    ) -> BasilicaDeployment:
        self._require_basilica_api_key()
        resolved = _resolve_model_repos(base_model=base_model, lora=lora, hf_token=hf_token)
        resolved_gpu_count = gpu_count or _default_gpu_count(resolved.base_model)
        resolved_gpu_models = gpu_models or list(DEFAULT_BASILICA_GPU_MODELS)
        deployment_model_name = deployment_model_name or _default_model_name(resolved.served_model)
        vllm_env = _build_vllm_env(
            base_model=resolved.base_model,
            lora=resolved.lora,
            deployment_model_name=deployment_model_name,
            hf_token=hf_token,
            env=env or {},
        )
        vllm_args = _build_vllm_start_cmd(
            base_model=resolved.base_model,
            lora=resolved.lora,
            deployment_model_name=deployment_model_name,
            port=port,
            env=env or {},
            hf_token=hf_token,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager,
            max_lora_rank=max_lora_rank,
            gpu_count=resolved_gpu_count,
        )
        basilica_args = _build_basilica_vllm_args(resolved.base_model, vllm_args)
        deployment_key = _deployment_key(
            provider=DeploymentProvider.BASILICA,
            base_model=resolved.base_model,
            lora=resolved.lora,
            template_id=RUNPOD_VLLM_IMAGE_NAME,
            port=port,
            gpu_count=resolved_gpu_count,
            gpu_type_ids=resolved_gpu_models,
            env=vllm_env,
            docker_start_cmd=basilica_args,
        )
        resolved_deployment_name = deployment_name or _default_basilica_deployment_name(deployment_key)

        if reuse_existing:
            existing = self._find_basilica_deployment(deployment_key, resolved_deployment_name)
            if existing:
                return self._basilica_deployment_from_model(
                    existing,
                    deployment_key=deployment_key,
                    base_model=resolved.base_model,
                    lora=resolved.lora,
                    model_name=deployment_model_name,
                    inference_api_key=inference_api_key,
                )

        metadata_env = {
            "GRADIENTS_DEPLOYMENT_KEY": deployment_key,
            "GRADIENTS_BASE_MODEL": resolved.base_model,
            "GRADIENTS_SDK_PROVIDER": DeploymentProvider.BASILICA.value,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
        if resolved.lora:
            metadata_env["GRADIENTS_LORA"] = resolved.lora

        app = self._create_basilica_deployment(
            deployment_name=resolved_deployment_name,
            port=port,
            command=["vllm"],
            args=["serve", *basilica_args],
            env={**vllm_env, **metadata_env},
            gpu_count=resolved_gpu_count,
            gpu_models=resolved_gpu_models,
            min_gpu_memory_gb=min_gpu_memory_gb,
            cpu=cpu,
            memory=memory,
            ttl_seconds=ttl_seconds,
        )
        deployment = self._basilica_deployment_from_model(
            app,
            deployment_key=deployment_key,
            base_model=resolved.base_model,
            lora=resolved.lora,
            model_name=deployment_model_name,
            inference_api_key=inference_api_key,
        )
        if wait:
            deployment.wait_ready(timeout=timeout)
        return deployment

    def get_basilica(self, deployment_name: str, *, deployment_key: str | None = None) -> DeploymentDetails:
        app = _basilica_deployment_from_payload(self._basilica_request("GET", f"/deployments/{deployment_name}"))
        key = deployment_key or _deployment_key_from_basilica(app)
        return DeploymentDetails(
            provider=DeploymentProvider.BASILICA,
            id=app.instance_name,
            deployment_key=key,
            base_model=str(app.env.get("GRADIENTS_BASE_MODEL") or ""),
            lora=app.env.get("GRADIENTS_LORA"),
            status=app.normalized_status,
            server_url=_basilica_server_url_from_model(app),
            pod=app,
        )

    def delete_basilica(self, deployment_name: str) -> None:
        self._basilica_request("DELETE", f"/deployments/{deployment_name}")

    def basilica_logs(self, deployment_name: str, *, tail: int | None = None) -> str:
        params = {"follow": "false"}
        if tail is not None:
            params["tail"] = str(tail)
        payload = self._basilica_request(
            "GET",
            f"/deployments/{deployment_name}/logs",
            params=params,
            accept="text/plain",
        )
        return payload if isinstance(payload, str) else ""

    def is_basilica_endpoint_ready(self, server_url: str) -> bool:
        if not server_url:
            return False
        try:
            response = self._client.get(f"{server_url.rstrip('/')}/v1/models", timeout=10.0, follow_redirects=True)
            if response.status_code != 200:
                return False
            payload = response.json()
            return isinstance(payload, dict) and isinstance(payload.get("data"), list)
        except httpx.HTTPError:
            return False
        except ValueError:
            return False

    def _find_basilica_deployment(
        self,
        deployment_key: str,
        deployment_name: str,
    ) -> BasilicaDeploymentModel | None:
        payload = self._basilica_request("GET", "/deployments")
        items = _basilica_list_items(payload)
        for item in items:
            summary = _basilica_deployment_from_payload(item)
            if summary.normalized_status.is_terminal:
                continue
            if summary.instance_name == deployment_name:
                return self._get_basilica_deployment_or_summary(summary)
            app = self._get_basilica_deployment_or_summary(summary)
            if _deployment_key_from_basilica(app) == deployment_key:
                return app
        return None

    def _get_basilica_deployment_or_summary(
        self,
        summary: BasilicaDeploymentModel,
    ) -> BasilicaDeploymentModel:
        try:
            return _basilica_deployment_from_payload(
                self._basilica_request("GET", f"/deployments/{summary.instance_name}")
            )
        except Exception:
            return summary

    def _create_basilica_deployment(
        self,
        *,
        deployment_name: str,
        port: int,
        command: list[str],
        args: list[str],
        env: dict[str, str],
        gpu_count: int,
        gpu_models: list[str],
        min_gpu_memory_gb: int,
        cpu: str,
        memory: str,
        ttl_seconds: int | None,
    ) -> BasilicaDeploymentModel:
        payload = {
            "instanceName": deployment_name,
            "image": RUNPOD_VLLM_IMAGE_NAME,
            "replicas": 1,
            "port": port,
            "command": command,
            "args": args,
            "env": env,
            "resources": {
                "cpu": cpu,
                "memory": memory,
                "gpus": {
                    "count": gpu_count,
                    "model": gpu_models,
                    "minGpuMemoryGb": min_gpu_memory_gb,
                },
            },
            "ttlSeconds": ttl_seconds,
            "public": True,
            "healthCheck": _basilica_vllm_health_check(port),
        }
        return _basilica_deployment_from_payload(self._basilica_request("POST", "/deployments", json=payload))

    def _basilica_deployment_from_model(
        self,
        app: BasilicaDeploymentModel,
        *,
        deployment_key: str,
        base_model: str,
        lora: str | None,
        model_name: str,
        inference_api_key: str | None,
    ) -> BasilicaDeployment:
        details = DeploymentDetails(
            provider=DeploymentProvider.BASILICA,
            id=app.instance_name,
            deployment_key=deployment_key,
            base_model=base_model,
            lora=lora,
            status=app.normalized_status,
            server_url=_basilica_server_url_from_model(app),
            pod=app,
        )
        _log_deployment_url(details)
        return BasilicaDeployment(self, details, model_name=model_name, inference_api_key=inference_api_key)

    def _basilica_request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        accept: str = "application/json",
    ) -> Any:
        self._require_basilica_api_key()
        url = f"{self.basilica_base_url}/{path.lstrip('/')}"
        headers = {
            "Accept": accept,
            "Authorization": f"Bearer {self.basilica_api_key}",
        }
        if json is not None:
            headers["Content-Type"] = "application/json"

        try:
            response = self._client.request(method, url, headers=headers, json=json, params=params)
        except httpx.HTTPError as exc:
            raise NetworkError(str(exc)) from exc

        if response.status_code >= 400:
            error_payload: Any
            try:
                error_payload = response.json()
            except ValueError:
                error_payload = {"message": response.text}
            _raise_provider_error("Basilica", response, error_payload)

        if response.status_code == 204 or not response.content:
            return None
        if accept == "text/plain":
            return response.text
        try:
            return response.json()
        except ValueError as exc:
            raise NetworkError(f"Basilica returned a non-JSON response from {path}") from exc

    def _require_basilica_api_key(self) -> None:
        if not self.basilica_api_key:
            raise ConfigurationError(f"Set {BASILICA_API_KEY_ENV} in your environment to deploy on Basilica.")


def _build_basilica_vllm_args(base_model: str, vllm_args: list[str]) -> list[str]:
    command = [base_model]
    index = 0
    while index < len(vllm_args):
        arg = vllm_args[index]
        if arg == "--model":
            index += 2
            continue
        command.append(arg)
        index += 1
    return command


def _default_basilica_deployment_name(deployment_key: str) -> str:
    return f"grad-basilica-{deployment_key}"


def _basilica_vllm_health_check(port: int) -> dict[str, Any]:
    # Large models can spend 10-20 minutes downloading and loading weights before /health exists.
    return {
        "liveness": {
            "path": "/health",
            "port": port,
            "initialDelaySeconds": 1800,
            "periodSeconds": 30,
            "timeoutSeconds": 10,
            "failureThreshold": 3,
        },
        "readiness": {
            "path": "/health",
            "port": port,
            "initialDelaySeconds": 30,
            "periodSeconds": 10,
            "timeoutSeconds": 5,
            "failureThreshold": 3,
        },
        "startup": {
            "path": "/health",
            "port": port,
            "initialDelaySeconds": 0,
            "periodSeconds": 10,
            "timeoutSeconds": 5,
            "failureThreshold": 180,
        },
    }


def _basilica_deployment_from_payload(item: dict[str, Any] | None) -> BasilicaDeploymentModel:
    item = item or {}
    return BasilicaDeploymentModel(
        instance_name=str(_get_any(item, "instance_name", "instanceName", "name") or ""),
        state=_get_any(item, "state", "status"),
        url=_get_any(item, "url", "endpointUrl", "endpoint_url"),
        message=_get_any(item, "message", "reason"),
        phase=_get_any(item, "phase"),
        env=_coerce_basilica_env(_get_any(item, "env", "environment", "envs")),
        replicas=_get_any(item, "replicas"),
        progress=_get_any(item, "progress"),
        pods=_get_any(item, "pods"),
    )


def _basilica_list_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        items = payload.get("deployments", payload.get("items", []))
        return items if isinstance(items, list) else []
    return payload if isinstance(payload, list) else []


def _coerce_basilica_env(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        result: dict[str, Any] = {}
        for item in value:
            if isinstance(item, dict) and item.get("name") is not None:
                result[str(item["name"])] = item.get("value")
        return result
    return {}


def _basilica_server_url_from_model(app: BasilicaDeploymentModel) -> str:
    return app.url or ""


def _deployment_key_from_basilica(app: BasilicaDeploymentModel) -> str:
    key = app.env.get("GRADIENTS_DEPLOYMENT_KEY")
    return str(key) if key else ""


def _log_excerpt(logs: str, *, max_lines: int = 20) -> str:
    lines = [line.rstrip() for line in logs.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


def _get_any(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return None
