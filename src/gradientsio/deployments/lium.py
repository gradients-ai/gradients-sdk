from __future__ import annotations

import shlex
import time
from typing import TYPE_CHECKING
from typing import Any

import httpx

from gradientsio.constants import DEFAULT_RUNPOD_PORT
from gradientsio.constants import DEFAULT_VLLM_MAX_LORA_RANK
from gradientsio.constants import LIUM_API_KEY_ENV
from gradientsio.constants import RUNPOD_VLLM_IMAGE_NAME
from gradientsio.constants import TERMINAL_LIUM_STATUSES
from gradientsio.deployments.common import _build_vllm_env
from gradientsio.deployments.common import _build_vllm_start_cmd
from gradientsio.deployments.common import _default_gpu_count
from gradientsio.deployments.common import _default_model_name
from gradientsio.deployments.common import _deployment_key
from gradientsio.deployments.common import _docker_image_name
from gradientsio.deployments.common import _docker_image_tag
from gradientsio.deployments.common import _ensure_local_ssh_keypair
from gradientsio.deployments.common import _lium_internal_port_from_template
from gradientsio.deployments.common import _lium_server_url_from_pod
from gradientsio.deployments.common import _lium_template_env
from gradientsio.deployments.common import _log_deployment_url
from gradientsio.deployments.common import _raise_provider_error
from gradientsio.deployments.common import _resolve_model_repos
from gradientsio.deployments.common import _Spinner
from gradientsio.deployments.common import logger
from gradientsio.errors import APIError
from gradientsio.errors import ConfigurationError
from gradientsio.errors import NetworkError
from gradientsio.errors import NotFoundError
from gradientsio.errors import TaskTimeout
from gradientsio.models import DeploymentDetails
from gradientsio.models import DeploymentProvider
from gradientsio.models import DeploymentStatus
from gradientsio.models import LiumExecutor
from gradientsio.models import LiumPod
from gradientsio.models import LiumSshKey
from gradientsio.models import LiumTemplate
from gradientsio.sampling import RemoteVLLMSampler


if TYPE_CHECKING:
    from gradientsio.deployments.client import DeploymentClient


class LiumDeployment:
    def __init__(
        self,
        deployments: "DeploymentClient",
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
        self.details = self._deployments.get_lium(self.id, deployment_key=self.deployment_key)
        return self.details

    def status(self) -> DeploymentStatus | str:
        return self.refresh().status

    def delete(self) -> None:
        self._deployments.delete_lium(self.id)

    def sampler(self, *, timeout: float = 60.0) -> RemoteVLLMSampler:
        return RemoteVLLMSampler(
            base_url=self.server_url,
            model=self.model_name,
            api_key=self.inference_api_key,
            timeout=timeout,
        )

    def wait_ready(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float = 15.0,
    ) -> DeploymentDetails:
        started_at = time.monotonic()
        logger.warning("Deploying on Lium now. This may take a few minutes, please grab a coffee.")
        spinner = _Spinner("Waiting for Lium vLLM server to become ready")

        while True:
            details = self.refresh()
            if details.is_running and self._deployments.is_lium_endpoint_ready(self.server_url):
                spinner.finish("Lium deployment is ready.")
                return details

            if details.is_terminal:
                spinner.finish()
                raise APIError(f"Lium deployment {self.id} reached terminal status {details.status}")

            if timeout is not None and time.monotonic() - started_at >= timeout:
                spinner.finish()
                raise TaskTimeout(f"Lium deployment {self.id} was not ready within {timeout} seconds")

            sleep_for = poll_interval
            if timeout is not None:
                remaining = timeout - (time.monotonic() - started_at)
                sleep_for = max(0.0, min(poll_interval, remaining))
            spinner.sleep(sleep_for)



class LiumDeploymentMixin:
    def deploy_lium(
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
        port: int = DEFAULT_RUNPOD_PORT,
        gpu_type: str | None = "H100",
        termination_hours: int | None = None,
        wait: bool = True,
        reuse_existing: bool = True,
        inference_api_key: str | None = None,
    ) -> LiumDeployment:
        self._require_lium_api_key()
        resolved = _resolve_model_repos(base_model=base_model, lora=lora, hf_token=hf_token)
        resolved_gpu_count = gpu_count or _default_gpu_count(resolved.base_model)
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
        lium_startup_args = _build_lium_vllm_startup_cmd(resolved.base_model, vllm_args)
        deployment_key = _deployment_key(
            provider=DeploymentProvider.LIUM,
            base_model=resolved.base_model,
            lora=resolved.lora,
            template_id=RUNPOD_VLLM_IMAGE_NAME,
            port=port,
            gpu_count=resolved_gpu_count,
            gpu_type_ids=[gpu_type] if gpu_type else [],
            env=vllm_env,
            docker_start_cmd=lium_startup_args,
        )
        pod_name = f"gradients-lium-{deployment_key}"

        if reuse_existing:
            existing = self._find_lium_deployment(deployment_key, pod_name)
            if existing:
                return self._lium_deployment_from_pod(
                    existing,
                    deployment_key=deployment_key,
                    base_model=resolved.base_model,
                    lora=resolved.lora,
                    model_name=deployment_model_name,
                    port=port,
                    inference_api_key=inference_api_key,
                )

        metadata_env = {
            "GRADIENTS_DEPLOYMENT_KEY": deployment_key,
            "GRADIENTS_BASE_MODEL": resolved.base_model,
            "GRADIENTS_SDK_PROVIDER": DeploymentProvider.LIUM.value,
        }
        if resolved.lora:
            metadata_env["GRADIENTS_LORA"] = resolved.lora
        template = self._find_or_create_lium_vllm_template(
            name=pod_name,
            port=port,
            env=_lium_template_env_without_json_values({**vllm_env, **metadata_env}),
            startup_command=shlex.join(lium_startup_args),
        )
        executor = self._select_lium_executor(gpu_count=resolved_gpu_count, gpu_type=gpu_type)
        public_key = self._default_lium_public_key()
        payload: dict[str, Any] = {
            "pod_name": pod_name,
            "template_id": template.id,
            "gpu_count": resolved_gpu_count,
            "user_public_key": public_key,
            "initial_port_count": 2,
            "enable_jupyter": False,
        }
        if termination_hours is not None:
            payload["termination_hours"] = termination_hours

        rent_result = self._lium_request("POST", f"/executors/{executor.id}/rent", json=payload)
        pod = self._lium_pod_from_rent_result(rent_result)
        if pod is None:
            pod = self._wait_for_lium_pod_record(pod_name)

        deployment = self._lium_deployment_from_pod(
            pod,
            deployment_key=deployment_key,
            base_model=resolved.base_model,
            lora=resolved.lora,
            model_name=deployment_model_name,
            port=port,
            inference_api_key=inference_api_key,
        )
        if wait:
            deployment.wait_ready()
        return deployment

    def get_lium(self, pod_id: str, *, deployment_key: str | None = None) -> DeploymentDetails:
        pod = LiumPod(**self._lium_request("GET", f"/pods/{pod_id}"))
        template_env = _lium_template_env(pod.template)
        key = deployment_key or str(template_env.get("GRADIENTS_DEPLOYMENT_KEY") or "")
        port = _lium_internal_port_from_template(pod.template) or DEFAULT_RUNPOD_PORT
        details = DeploymentDetails(
            provider=DeploymentProvider.LIUM,
            id=pod.id,
            deployment_key=key,
            base_model=str(template_env.get("GRADIENTS_BASE_MODEL") or ""),
            lora=template_env.get("GRADIENTS_LORA"),
            status=pod.normalized_status,
            server_url=_lium_server_url_from_pod(pod, port),
            pod=pod,
        )
        return details

    def delete_lium(self, pod_id: str) -> None:
        self._lium_request("DELETE", f"/pods/{pod_id}")

    def is_lium_endpoint_ready(self, server_url: str) -> bool:
        if not server_url:
            return False
        try:
            response = self._client.get(f"{server_url.rstrip('/')}/v1/models", timeout=10.0)
            if response.status_code != 200:
                return False
            payload = response.json()
            return isinstance(payload, dict) and isinstance(payload.get("data"), list)
        except httpx.HTTPError:
            return False
        except ValueError:
            return False

    def _find_lium_deployment(self, deployment_key: str, pod_name: str) -> LiumPod | None:
        pods_payload = self._lium_request("GET", "/pods")
        if not isinstance(pods_payload, list):
            return None
        for item in pods_payload:
            pod = LiumPod(**item)
            if str(pod.status or "").upper() in TERMINAL_LIUM_STATUSES:
                continue
            template_env = _lium_template_env(pod.template)
            if template_env.get("GRADIENTS_DEPLOYMENT_KEY") == deployment_key:
                return pod
            if pod.pod_name == pod_name:
                return pod
        return None

    def _lium_deployment_from_pod(
        self,
        pod: LiumPod,
        *,
        deployment_key: str,
        base_model: str,
        lora: str | None,
        model_name: str,
        port: int,
        inference_api_key: str | None,
    ) -> LiumDeployment:
        details = DeploymentDetails(
            provider=DeploymentProvider.LIUM,
            id=pod.id,
            deployment_key=deployment_key,
            base_model=base_model,
            lora=lora,
            status=pod.normalized_status,
            server_url=_lium_server_url_from_pod(pod, port),
            pod=pod,
        )
        _log_deployment_url(details)
        return LiumDeployment(self, details, model_name=model_name, inference_api_key=inference_api_key)

    def _find_or_create_lium_vllm_template(
        self,
        *,
        name: str,
        port: int,
        env: dict[str, str],
        startup_command: str,
    ) -> LiumTemplate:
        templates_payload = self._lium_request("GET", "/templates")
        if isinstance(templates_payload, list):
            for item in templates_payload:
                template = LiumTemplate(**item)
                if template.name == name:
                    return template

        payload = {
            "name": name,
            "docker_image": _docker_image_name(RUNPOD_VLLM_IMAGE_NAME),
            "docker_image_tag": _docker_image_tag(RUNPOD_VLLM_IMAGE_NAME),
            "volumes": ["/root"],
            "environment": env,
            "internal_ports": [22, port],
            "startup_commands": startup_command,
        }
        return LiumTemplate(**self._lium_request("POST", "/templates", json=payload))

    def _select_lium_executor(self, *, gpu_count: int, gpu_type: str | None) -> LiumExecutor:
        for preferred_gpu_type in _lium_gpu_type_preferences(gpu_type):
            params: dict[str, Any] = {
                "gpu_count_gte": gpu_count,
            }
            if preferred_gpu_type:
                params["machine_names"] = preferred_gpu_type

            payload = self._lium_request("GET", "/executors", params=params)
            if not isinstance(payload, list):
                raise NotFoundError("Lium returned no executor list.")
            executors = [LiumExecutor(**item) for item in payload]
            compatible = [
                executor
                for executor in executors
                if (executor.available_gpu_count or executor.gpu_count or 0) >= gpu_count
                and (executor.min_gpu_count_for_rental or 1) <= gpu_count
                and _lium_executor_matches_gpu_type(executor, preferred_gpu_type)
            ]
            if compatible:
                return sorted(compatible, key=lambda executor: executor.price_per_gpu or float("inf"))[0]

        if gpu_type and gpu_type.lower() == "h100":
            payload = self._lium_request("GET", "/executors", params={"gpu_count_gte": gpu_count})
            if not isinstance(payload, list):
                raise NotFoundError("Lium returned no executor list.")
            executors = [LiumExecutor(**item) for item in payload]
            compatible = [
                executor
                for executor in executors
                if (executor.available_gpu_count or executor.gpu_count or 0) >= gpu_count
                and (executor.min_gpu_count_for_rental or 1) <= gpu_count
            ]
            if compatible:
                return sorted(compatible, key=lambda executor: executor.price_per_gpu or float("inf"))[0]

        requested = " or ".join(gpu for gpu in [*_lium_gpu_type_preferences(gpu_type), "any available GPU"] if gpu)
        requested = f" {requested}" if requested else ""
        raise NotFoundError(f"No compatible Lium{requested} executor found for {gpu_count} GPU(s).")

    def _default_lium_public_key(self) -> str:
        keys_payload = self._lium_request("GET", "/ssh-keys")
        if isinstance(keys_payload, list):
            keys = [LiumSshKey(**item) for item in keys_payload]
            if keys:
                logger.warning("Using existing Lium SSH public key: %s", keys[0].name or keys[0].id or "default")
                return keys[0].public_key

        public_key_path = _ensure_local_ssh_keypair()
        public_key = public_key_path.read_text().strip()
        logger.warning("Registering local SSH public key with Lium: %s", public_key_path)
        self._lium_request(
            "POST",
            "/ssh-keys",
            json={
                "name": "gradients-sdk",
                "public_key": public_key,
            },
        )
        return public_key

    def _lium_pod_from_rent_result(self, rent_result: Any) -> LiumPod | None:
        if not isinstance(rent_result, dict):
            return None
        pod_payload = rent_result.get("pod") if isinstance(rent_result.get("pod"), dict) else rent_result
        if isinstance(pod_payload, dict) and pod_payload.get("id") and pod_payload.get("pod_name"):
            return LiumPod(**pod_payload)
        pod_id = rent_result.get("pod_id") or rent_result.get("id")
        if pod_id:
            return LiumPod(**self._lium_request("GET", f"/pods/{pod_id}"))
        return None

    def _wait_for_lium_pod_record(self, pod_name: str, *, timeout: float = 60.0) -> LiumPod:
        started_at = time.monotonic()
        while True:
            pod = self._find_lium_deployment("", pod_name)
            if pod:
                return pod
            if time.monotonic() - started_at >= timeout:
                raise TaskTimeout(f"Lium pod {pod_name} was not visible within {timeout} seconds")
            time.sleep(2.0)

    def _lium_request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        self._require_lium_api_key()
        url = f"{self.lium_base_url}/{path.lstrip('/')}"
        headers = {
            "Accept": "application/json",
            "X-API-Key": self.lium_api_key or "",
        }
        if json is not None:
            headers["Content-Type"] = "application/json"

        try:
            response = self._client.request(method, url, headers=headers, json=json, params=params)
        except httpx.HTTPError as exc:
            raise NetworkError(str(exc)) from exc

        if response.status_code >= 400:
            try:
                error_payload = response.json()
            except ValueError:
                error_payload = {}
            _raise_provider_error("Lium", response, error_payload)

        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise NetworkError(f"Lium returned a non-JSON response from {path}") from exc

    def _require_lium_api_key(self) -> None:
        if not self.lium_api_key:
            raise ConfigurationError(f"Set {LIUM_API_KEY_ENV} in your environment to deploy on Lium.")


def _lium_template_env_without_json_values(env: dict[str, str]) -> dict[str, str]:
    # Lium rejects env values containing quotes. vLLM gets LoRA modules from the startup command instead.
    return {key: value for key, value in env.items() if key != "LORA_MODULES"}


def _build_lium_vllm_startup_cmd(base_model: str, vllm_args: list[str]) -> list[str]:
    # The official vLLM OpenAI image already uses `vllm serve` as ENTRYPOINT.
    # Lium's template command is passed as arguments to that entrypoint.
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


def _lium_executor_matches_gpu_type(executor: LiumExecutor, gpu_type: str | None) -> bool:
    if not gpu_type:
        return True
    requested = gpu_type.lower()
    candidates = [
        executor.machine_name,
        str(executor.specs.get("gpu_name")) if executor.specs else None,
        str(executor.specs.get("gpu_type")) if executor.specs else None,
        str(executor.specs.get("gpu_model")) if executor.specs else None,
    ]
    return any(requested in str(candidate).lower() for candidate in candidates if candidate)


def _lium_gpu_type_preferences(gpu_type: str | None) -> list[str | None]:
    if not gpu_type:
        return [None]
    return [gpu_type]



