from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx

from gradientsio.constants import DEFAULT_H100_GPU_TYPES
from gradientsio.constants import DEFAULT_RUNPOD_GPU_COUNT
from gradientsio.constants import DEFAULT_RUNPOD_PORT
from gradientsio.constants import DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
from gradientsio.constants import DEFAULT_VLLM_HOST
from gradientsio.constants import DEFAULT_VLLM_MAX_LORA_RANK
from gradientsio.constants import DEFAULT_VLLM_MAX_MODEL_LEN
from gradientsio.constants import DEFAULT_VLLM_PORT
from gradientsio.constants import HUGGING_FACE_TOKEN_ENV
from gradientsio.constants import LARGE_MODEL_VLLM_MAX_MODEL_LEN
from gradientsio.constants import MEDIUM_MODEL_VLLM_MAX_MODEL_LEN
from gradientsio.constants import RUNPOD_API_KEY_ENV
from gradientsio.constants import RUNPOD_BASE_URL
from gradientsio.constants import RUNPOD_VLLM_IMAGE_NAME
from gradientsio.constants import TERMINATED_RUNPOD_STATUSES
from gradientsio.errors import APIError
from gradientsio.errors import AuthenticationError
from gradientsio.errors import ConfigurationError
from gradientsio.errors import NetworkError
from gradientsio.errors import NotFoundError
from gradientsio.errors import TaskTimeout
from gradientsio.errors import ValidationError
from gradientsio.models import DeploymentDetails
from gradientsio.models import DeploymentProvider
from gradientsio.models import DeploymentStatus
from gradientsio.models import RunPodDeploymentRequest
from gradientsio.models import RunPodPod
from gradientsio.sampling import RemoteVLLMSampler


logger = logging.getLogger(__name__)


class LocalVLLMDeployment:
    def __init__(
        self,
        details: DeploymentDetails,
        *,
        model_name: str,
        process: subprocess.Popen[str] | None = None,
        started_by_sdk: bool = False,
        inference_api_key: str | None = None,
    ) -> None:
        self.details = details
        self.model_name = model_name
        self.process = process
        self.started_by_sdk = started_by_sdk
        self.inference_api_key = inference_api_key

    @property
    def id(self) -> str:
        return self.details.id

    @property
    def server_url(self) -> str:
        return self.details.server_url

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
        timeout: float | None = 600.0,
        poll_interval: float = 2.0,
    ) -> DeploymentDetails:
        started_at = time.monotonic()
        spinner = _Spinner("Waiting for local vLLM server to become ready")

        while True:
            if _is_openai_endpoint_ready(self.server_url):
                spinner.finish("Local vLLM server is ready.")
                self.details.status = DeploymentStatus.RUNNING
                return self.details

            if self.process and self.process.poll() is not None:
                spinner.finish()
                raise APIError(f"Local vLLM server exited with code {self.process.returncode}")

            if timeout is not None and time.monotonic() - started_at >= timeout:
                spinner.finish()
                raise TaskTimeout(f"Local vLLM server was not ready within {timeout} seconds")

            sleep_for = poll_interval
            if timeout is not None:
                remaining = timeout - (time.monotonic() - started_at)
                sleep_for = max(0.0, min(poll_interval, remaining))
            spinner.sleep(sleep_for)

    def stop(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        self.details.status = DeploymentStatus.TERMINATED

    def delete(self) -> None:
        self.stop()


class RunPodDeployment:
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
        self.details = self._deployments.get_runpod(self.id, deployment_key=self.deployment_key)
        return self.details

    def status(self) -> DeploymentStatus | str:
        return self.refresh().status

    def delete(self) -> None:
        self._deployments.delete_runpod(self.id)

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
        logger.warning("Deploying on RunPod now. This may take a few minutes, please grab a coffee.")
        spinner = _Spinner("Waiting for vLLM server to become ready")

        while True:
            details = self.refresh()
            if details.is_running and self._deployments.is_runpod_endpoint_ready(self.server_url):
                spinner.finish("RunPod deployment is ready.")
                return details

            if details.is_terminal:
                spinner.finish()
                raise APIError(f"RunPod deployment {self.id} reached terminal status {details.status}")

            if timeout is not None and time.monotonic() - started_at >= timeout:
                spinner.finish()
                raise TaskTimeout(f"RunPod deployment {self.id} was not ready within {timeout} seconds")

            sleep_for = poll_interval
            if timeout is not None:
                remaining = timeout - (time.monotonic() - started_at)
                sleep_for = max(0.0, min(poll_interval, remaining))
            spinner.sleep(sleep_for)


class DeploymentClient:
    def __init__(
        self,
        *,
        runpod_base_url: str = RUNPOD_BASE_URL,
        timeout: float | httpx.Timeout = 60.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.runpod_api_key = os.getenv(RUNPOD_API_KEY_ENV)
        self.runpod_base_url = runpod_base_url.rstrip("/")
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def deploy_runpod(
        self,
        *,
        base_model: str,
        lora: str | None = None,
        template_id: str | None = None,
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
        gpu_type_ids: list[str] | None = None,
        cloud_type: str = "SECURE",
        container_disk_in_gb: int = 100,
        volume_in_gb: int = 100,
        interruptible: bool = False,
        inference_api_key: str | None = None,
        **pod_options: Any,
    ) -> RunPodDeployment:
        self._require_runpod_api_key()
        image_name = template_id or RUNPOD_VLLM_IMAGE_NAME
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
        docker_start_cmd = _build_vllm_start_cmd(
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
        deployment_key = _deployment_key(
            base_model=resolved.base_model,
            lora=resolved.lora,
            template_id=image_name,
            port=port,
            gpu_count=resolved_gpu_count,
            gpu_type_ids=gpu_type_ids or DEFAULT_H100_GPU_TYPES,
            env=vllm_env,
            docker_start_cmd=docker_start_cmd,
        )
        pod_name = f"gradients-runpod-{deployment_key}"
        metadata_env = {
            "GRADIENTS_DEPLOYMENT_KEY": deployment_key,
            "GRADIENTS_BASE_MODEL": resolved.base_model,
            "GRADIENTS_SDK_PROVIDER": DeploymentProvider.RUNPOD.value,
        }
        if resolved.lora:
            metadata_env["GRADIENTS_LORA"] = resolved.lora
        request = RunPodDeploymentRequest(
            base_model=resolved.base_model,
            lora=resolved.lora,
            template_id=image_name,
            deployment_name=pod_name,
            deployment_key=deployment_key,
            port=port,
            gpu_count=resolved_gpu_count,
            gpu_type_ids=gpu_type_ids or DEFAULT_H100_GPU_TYPES,
            docker_start_cmd=docker_start_cmd,
            env={**vllm_env, **metadata_env},
        )

        existing = self._find_runpod_deployment(deployment_key, pod_name)
        if existing:
            return self._deployment_from_pod(
                existing,
                request=request,
                model_name=deployment_model_name,
                inference_api_key=inference_api_key,
            )

        payload = {
            "cloudType": cloud_type,
            "computeType": "GPU",
            "containerDiskInGb": container_disk_in_gb,
            "dockerStartCmd": request.docker_start_cmd,
            "env": request.env,
            "gpuCount": request.gpu_count,
            "gpuTypeIds": request.gpu_type_ids,
            "gpuTypePriority": "availability",
            "interruptible": interruptible,
            "name": pod_name,
            "ports": [f"{port}/http"],
            "imageName": image_name,
            "volumeInGb": volume_in_gb,
            **pod_options,
        }
        pod = RunPodPod(**self._runpod_request("POST", "/pods", json=payload))
        return self._deployment_from_pod(
            pod,
            request=request,
            model_name=deployment_model_name,
            inference_api_key=inference_api_key,
        )

    def deploy_local_vllm(
        self,
        *,
        base_model: str,
        lora: str | None = None,
        deployment_model_name: str | None = None,
        hf_token: str | None = None,
        env: dict[str, Any] | None = None,
        host: str = DEFAULT_VLLM_HOST,
        port: int = DEFAULT_VLLM_PORT,
        wait: bool = True,
        reuse_existing: bool = True,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | str | None = None,
        dtype: str | None = None,
        trust_remote_code: bool | None = None,
        enforce_eager: bool = False,
        max_lora_rank: int | None = DEFAULT_VLLM_MAX_LORA_RANK,
        gpu_count: int | None = None,
        inference_api_key: str | None = None,
    ) -> LocalVLLMDeployment:
        resolved = _resolve_model_repos(base_model=base_model, lora=lora, hf_token=hf_token)
        deployment_model_name = deployment_model_name or _default_model_name(resolved.served_model)
        resolved_gpu_count = gpu_count or DEFAULT_RUNPOD_GPU_COUNT
        args = _build_vllm_start_cmd(
            base_model=resolved.base_model,
            lora=resolved.lora,
            deployment_model_name=deployment_model_name,
            host=host,
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
        server_url = f"http://{host}:{port}"
        deployment_key = _deployment_key(
            base_model=resolved.base_model,
            lora=resolved.lora,
            template_id="local-vllm",
            port=port,
            gpu_count=resolved_gpu_count,
            gpu_type_ids=[],
            env=_build_vllm_env(
                base_model=resolved.base_model,
                lora=resolved.lora,
                deployment_model_name=deployment_model_name,
                hf_token=hf_token,
                env=env or {},
            ),
            docker_start_cmd=args,
        )

        details = DeploymentDetails(
            provider="local",
            id=f"local-vllm-{host}-{port}",
            deployment_key=deployment_key,
            base_model=resolved.base_model,
            lora=resolved.lora,
            status=DeploymentStatus.PENDING,
            server_url=server_url,
        )
        if reuse_existing and _is_openai_endpoint_ready(server_url):
            details.status = DeploymentStatus.RUNNING
            return LocalVLLMDeployment(
                details,
                model_name=deployment_model_name,
                process=None,
                started_by_sdk=False,
                inference_api_key=inference_api_key,
            )
        if _port_in_use(host, port):
            raise ConfigurationError(
                f"Port {host}:{port} is already in use, but no OpenAI-compatible vLLM server is ready there."
            )

        process_env = os.environ.copy()
        process_env.update(_build_vllm_env(
            base_model=resolved.base_model,
            lora=resolved.lora,
            deployment_model_name=deployment_model_name,
            hf_token=hf_token,
            env=env or {},
        ))
        command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *args]
        process = subprocess.Popen(command, env=process_env, text=True)
        deployment = LocalVLLMDeployment(
            details,
            model_name=deployment_model_name,
            process=process,
            started_by_sdk=True,
            inference_api_key=inference_api_key,
        )
        if wait:
            deployment.wait_ready()
        return deployment

    def get_runpod(self, pod_id: str, *, deployment_key: str | None = None) -> DeploymentDetails:
        pod = RunPodPod(**self._runpod_request("GET", f"/pods/{pod_id}"))
        pod_env = pod.env or {}
        key = deployment_key or str(pod_env.get("GRADIENTS_DEPLOYMENT_KEY") or "")
        details = DeploymentDetails(
            provider=DeploymentProvider.RUNPOD,
            id=pod.id,
            deployment_key=key,
            base_model=str(pod_env.get("GRADIENTS_BASE_MODEL") or ""),
            lora=pod_env.get("GRADIENTS_LORA"),
            status=pod.normalized_status,
            server_url=_runpod_proxy_url(pod.id, _port_from_pod(pod)),
            pod=pod,
        )
        return details

    def delete_runpod(self, pod_id: str) -> None:
        self._runpod_request("DELETE", f"/pods/{pod_id}")

    def is_runpod_endpoint_ready(self, server_url: str) -> bool:
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

    def _find_runpod_deployment(self, deployment_key: str, pod_name: str) -> RunPodPod | None:
        payload = self._runpod_request("GET", "/pods", params={"computeType": "GPU"})
        pods_payload = payload.get("pods", payload) if isinstance(payload, dict) else payload
        if not isinstance(pods_payload, list):
            return None

        for item in pods_payload:
            pod = RunPodPod(**item)
            if (pod.desiredStatus or "").upper() in TERMINATED_RUNPOD_STATUSES:
                continue

            pod_env = pod.env or {}
            if pod_env.get("GRADIENTS_DEPLOYMENT_KEY") == deployment_key:
                return pod
            if pod.name == pod_name:
                return pod
        return None

    def _deployment_from_pod(
        self,
        pod: RunPodPod,
        *,
        request: RunPodDeploymentRequest,
        model_name: str,
        inference_api_key: str | None,
    ) -> RunPodDeployment:
        details = DeploymentDetails(
            provider=DeploymentProvider.RUNPOD,
            id=pod.id,
            deployment_key=request.deployment_key or "",
            base_model=request.base_model,
            lora=request.lora,
            status=pod.normalized_status,
            server_url=_runpod_proxy_url(pod.id, request.port),
            pod=pod,
        )
        _log_deployment_url(details)
        return RunPodDeployment(self, details, model_name=model_name, inference_api_key=inference_api_key)

    def _runpod_request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        self._require_runpod_api_key()
        url = f"{self.runpod_base_url}/{path.lstrip('/')}"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.runpod_api_key}",
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
            _raise_runpod_error(response, error_payload)

        if response.status_code == 204 or not response.content:
            return None
        try:
            payload = response.json()
        except ValueError as exc:
            raise NetworkError(f"RunPod returned a non-JSON response from {path}") from exc

        return payload

    def _require_runpod_api_key(self) -> None:
        if not self.runpod_api_key:
            raise ConfigurationError(f"Set {RUNPOD_API_KEY_ENV} in your environment to deploy on RunPod.")


def deploy_runpod(**kwargs: Any) -> RunPodDeployment:
    client_kwargs = {
        key: kwargs.pop(key)
        for key in ("runpod_base_url", "timeout", "http_client")
        if key in kwargs
    }
    client = DeploymentClient(**client_kwargs)
    return client.deploy_runpod(**kwargs)


def deploy_local_vllm(**kwargs: Any) -> LocalVLLMDeployment:
    client_kwargs = {
        key: kwargs.pop(key)
        for key in ("runpod_base_url", "timeout", "http_client")
        if key in kwargs
    }
    client = DeploymentClient(**client_kwargs)
    return client.deploy_local_vllm(**kwargs)


@dataclass(frozen=True)
class _ResolvedModelRepos:
    base_model: str
    lora: str | None

    @property
    def served_model(self) -> str:
        return self.lora or self.base_model


def _resolve_model_repos(
    *,
    base_model: str,
    lora: str | None,
    hf_token: str | None,
) -> _ResolvedModelRepos:
    if not lora:
        return _ResolvedModelRepos(base_model=base_model, lora=None)
    if _adapter_base_model(lora, hf_token=hf_token):
        return _ResolvedModelRepos(base_model=base_model, lora=lora)
    return _ResolvedModelRepos(base_model=lora, lora=None)


def _adapter_base_model(repo_id: str, *, hf_token: str | None) -> str | None:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[reportMissingImports]

        adapter_config_path = hf_hub_download(repo_id=repo_id, filename="adapter_config.json", token=hf_token)
    except Exception:
        return None

    try:
        with open(adapter_config_path) as config_file:
            adapter_config = json.load(config_file)
    except (OSError, ValueError):
        return None

    base_model = adapter_config.get("base_model_name_or_path")
    return str(base_model) if base_model else None


def _build_vllm_env(
    *,
    base_model: str,
    lora: str | None,
    deployment_model_name: str,
    hf_token: str | None,
    env: dict[str, Any],
) -> dict[str, str]:
    result = {
        "MODEL_NAME": base_model,
    }
    if lora:
        lora_modules = [
            {
                "name": deployment_model_name,
                "path": lora,
                "base_model_name": base_model,
            }
        ]
        result["ENABLE_LORA"] = "true"
        result["LORA_MODULES"] = json.dumps(lora_modules, separators=(",", ":"))
    if hf_token:
        result[HUGGING_FACE_TOKEN_ENV] = hf_token
    for key, value in env.items():
        if value is not None:
            result[key] = str(value)
    return result


def _build_vllm_start_cmd(
    *,
    base_model: str,
    lora: str | None,
    deployment_model_name: str,
    port: int,
    host: str = "0.0.0.0",
    env: dict[str, Any] | None = None,
    hf_token: str | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | str | None = None,
    dtype: str | None = None,
    trust_remote_code: bool | None = None,
    enforce_eager: bool = False,
    max_lora_rank: int | None = DEFAULT_VLLM_MAX_LORA_RANK,
    gpu_count: int = DEFAULT_RUNPOD_GPU_COUNT,
) -> list[str]:
    env = env or {}
    resolved_max_model_len = str(
        max_model_len or _env_value(env, "MAX_MODEL_LEN") or _default_max_model_len(base_model, hf_token=hf_token)
    )
    resolved_gpu_memory_utilization = str(
        gpu_memory_utilization or _env_value(env, "GPU_MEMORY_UTILIZATION") or DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
    )
    resolved_dtype = dtype or _env_value(env, "DTYPE")
    resolved_trust_remote_code = (
        trust_remote_code if trust_remote_code is not None else _truthy_env_value(env, "TRUST_REMOTE_CODE")
    )
    command = [
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        base_model,
        "--served-model-name",
        deployment_model_name,
        "--max-model-len",
        resolved_max_model_len,
        "--gpu-memory-utilization",
        resolved_gpu_memory_utilization,
    ]
    if resolved_dtype:
        command.extend(["--dtype", resolved_dtype])
    if resolved_trust_remote_code:
        command.append("--trust-remote-code")
    if enforce_eager:
        command.append("--enforce-eager")
    if gpu_count > 1:
        command.extend(["--tensor-parallel-size", str(gpu_count)])
    if lora:
        command.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                str(max_lora_rank or DEFAULT_VLLM_MAX_LORA_RANK),
                "--lora-modules",
                f"{deployment_model_name}={lora}",
            ]
        )
    return command


def _default_gpu_count(model_repo: str) -> int:
    model_size_billions = _model_size_billions(model_repo)
    if model_size_billions is None:
        return DEFAULT_RUNPOD_GPU_COUNT
    if model_size_billions > 80:
        return 4
    if model_size_billions > 40:
        return 2
    return DEFAULT_RUNPOD_GPU_COUNT


def _default_max_model_len(model_repo: str, *, hf_token: str | None) -> int:
    model_size_billions = _model_size_billions(model_repo)
    if model_size_billions is not None and model_size_billions >= 30:
        return LARGE_MODEL_VLLM_MAX_MODEL_LEN
    if model_size_billions is not None and model_size_billions >= 14:
        return DEFAULT_VLLM_MAX_MODEL_LEN
    if model_size_billions is not None and model_size_billions >= 7:
        return MEDIUM_MODEL_VLLM_MAX_MODEL_LEN
    if model_size_billions is not None:
        return DEFAULT_VLLM_MAX_MODEL_LEN

    configured_context = _model_context_length(model_repo, hf_token=hf_token)
    if configured_context:
        return min(configured_context, DEFAULT_VLLM_MAX_MODEL_LEN)
    return DEFAULT_VLLM_MAX_MODEL_LEN


def _model_context_length(model_repo: str, *, hf_token: str | None) -> int | None:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[reportMissingImports]

        config_path = hf_hub_download(repo_id=model_repo, filename="config.json", token=hf_token)
    except Exception:
        return None

    try:
        with open(config_path) as config_file:
            config = json.load(config_file)
    except (OSError, ValueError):
        return None

    for key in ("max_position_embeddings", "seq_length", "n_positions", "model_max_length"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _model_size_billions(model_repo: str) -> float | None:
    match = re.search(r"(?i)(\d+(?:\.\d+)?)\s*b(?:\D|$)", model_repo)
    if not match:
        return None
    return float(match.group(1))


def _env_value(env: dict[str, Any], key: str) -> str | None:
    value = env.get(key)
    if value is None:
        return None
    return str(value)


def _truthy_env_value(env: dict[str, Any], key: str) -> bool:
    value = _env_value(env, key)
    return bool(value and value.lower() in {"1", "true", "yes", "on"})


def _deployment_key(
    *,
    base_model: str,
    lora: str | None,
    template_id: str,
    port: int,
    gpu_count: int,
    gpu_type_ids: list[str],
    env: dict[str, str],
    docker_start_cmd: list[str],
) -> str:
    public_env = {key: value for key, value in env.items() if key not in {HUGGING_FACE_TOKEN_ENV}}
    payload = {
        "provider": DeploymentProvider.RUNPOD.value,
        "base_model": base_model,
        "lora": lora,
        "template_id": template_id,
        "port": port,
        "gpu_count": gpu_count,
        "gpu_type_ids": gpu_type_ids,
        "env": public_env,
        "docker_start_cmd": docker_start_cmd,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _default_model_name(repo: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", repo).strip("-")
    return f"gradients-{slug}"[:63] or "gradients-model"


def _runpod_proxy_url(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def _is_openai_endpoint_ready(server_url: str) -> bool:
    try:
        response = httpx.get(f"{server_url.rstrip('/')}/v1/models", timeout=10.0)
        if response.status_code != 200:
            return False
        payload = response.json()
        return isinstance(payload, dict) and isinstance(payload.get("data"), list)
    except (httpx.HTTPError, ValueError):
        return False


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


class _Spinner:
    frames = "|/-\\"

    def __init__(self, message: str) -> None:
        self.message = message
        self.index = 0
        self.enabled = sys.stderr.isatty()

    def sleep(self, seconds: float) -> None:
        if not self.enabled:
            time.sleep(seconds)
            return

        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._render()
            time.sleep(min(0.25, remaining))

    def finish(self, message: str | None = None) -> None:
        if not self.enabled:
            if message:
                logger.warning(message)
            return
        sys.stderr.write("\r" + " " * (len(self.message) + 8) + "\r")
        if message:
            sys.stderr.write(f"{message}\n")
        sys.stderr.flush()

    def _render(self) -> None:
        frame = self.frames[self.index % len(self.frames)]
        self.index += 1
        sys.stderr.write(f"\r{frame} {self.message}...")
        sys.stderr.flush()


def _log_deployment_url(details: DeploymentDetails) -> None:
    logger.warning("RunPod deployment server URL: %s", details.server_url)


def _port_from_pod(pod: RunPodPod) -> int:
    if pod.ports:
        first_port = pod.ports[0].split("/", 1)[0]
        if first_port.isdigit():
            return int(first_port)
    return DEFAULT_RUNPOD_PORT


def _raise_runpod_error(response: httpx.Response, payload: Any) -> None:
    message = response.reason_phrase
    if isinstance(payload, dict):
        detail = payload.get("error") or payload.get("message") or payload.get("detail")
        if detail is not None:
            message = str(detail)

    kwargs = {"status_code": response.status_code, "response": response}
    if response.status_code == 401:
        raise AuthenticationError(message, **kwargs)
    if response.status_code == 404:
        raise NotFoundError(message, **kwargs)
    if response.status_code in {400, 422}:
        raise ValidationError(message, **kwargs)
    raise APIError(message, **kwargs)
