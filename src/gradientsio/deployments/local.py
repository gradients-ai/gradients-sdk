from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any

from gradientsio.constants import DEFAULT_RUNPOD_GPU_COUNT
from gradientsio.constants import DEFAULT_VLLM_HOST
from gradientsio.constants import DEFAULT_VLLM_MAX_LORA_RANK
from gradientsio.constants import DEFAULT_VLLM_PORT
from gradientsio.deployments.common import _build_vllm_env
from gradientsio.deployments.common import _build_vllm_start_cmd
from gradientsio.deployments.common import _default_model_name
from gradientsio.deployments.common import _deployment_key
from gradientsio.deployments.common import _is_openai_endpoint_ready
from gradientsio.deployments.common import _port_in_use
from gradientsio.deployments.common import _resolve_model_repos
from gradientsio.deployments.common import _Spinner
from gradientsio.errors import APIError
from gradientsio.errors import ConfigurationError
from gradientsio.errors import TaskTimeout
from gradientsio.models import DeploymentDetails
from gradientsio.models import DeploymentStatus
from gradientsio.sampling import RemoteVLLMSampler


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



class LocalDeploymentMixin:
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


