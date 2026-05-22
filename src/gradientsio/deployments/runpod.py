from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import Any

import httpx

from gradientsio.constants import DEFAULT_H100_GPU_TYPES
from gradientsio.constants import DEFAULT_RUNPOD_PORT
from gradientsio.constants import DEFAULT_VLLM_MAX_LORA_RANK
from gradientsio.constants import RUNPOD_API_KEY_ENV
from gradientsio.constants import RUNPOD_VLLM_IMAGE_NAME
from gradientsio.constants import TERMINATED_RUNPOD_STATUSES
from gradientsio.deployments.common import _build_vllm_env
from gradientsio.deployments.common import _build_vllm_start_cmd
from gradientsio.deployments.common import _default_gpu_count
from gradientsio.deployments.common import _default_model_name
from gradientsio.deployments.common import _deployment_failure_message
from gradientsio.deployments.common import _deployment_key
from gradientsio.deployments.common import _log_deployment_url
from gradientsio.deployments.common import _port_from_pod
from gradientsio.deployments.common import _raise_runpod_error
from gradientsio.deployments.common import _resolve_model_repos
from gradientsio.deployments.common import _runpod_proxy_url
from gradientsio.deployments.common import _Spinner
from gradientsio.deployments.common import logger
from gradientsio.errors import APIError
from gradientsio.errors import ConfigurationError
from gradientsio.errors import NetworkError
from gradientsio.errors import TaskTimeout
from gradientsio.models import DeploymentDetails
from gradientsio.models import DeploymentProvider
from gradientsio.models import DeploymentStatus
from gradientsio.models import RunPodDeploymentRequest
from gradientsio.models import RunPodPod
from gradientsio.sampling import RemoteVLLMSampler


if TYPE_CHECKING:
    from gradientsio.deployments.client import DeploymentClient


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
            failure_message = _deployment_failure_message(details)
            if failure_message:
                spinner.finish()
                raise APIError(failure_message)

            if timeout is not None and time.monotonic() - started_at >= timeout:
                spinner.finish()
                raise TaskTimeout(f"RunPod deployment {self.id} was not ready within {timeout} seconds")

            sleep_for = poll_interval
            if timeout is not None:
                remaining = timeout - (time.monotonic() - started_at)
                sleep_for = max(0.0, min(poll_interval, remaining))
            spinner.sleep(sleep_for)



class RunPodDeploymentMixin:
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


