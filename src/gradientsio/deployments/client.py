from __future__ import annotations

import os
from typing import Any

import httpx

from gradientsio.constants import LIUM_API_KEY_ENV
from gradientsio.constants import LIUM_BASE_URL
from gradientsio.constants import RUNPOD_API_KEY_ENV
from gradientsio.constants import RUNPOD_BASE_URL
from gradientsio.constants import TARGON_API_KEY_ENV
from gradientsio.constants import TARGON_BASE_URL
from gradientsio.deployments.lium import LiumDeployment
from gradientsio.deployments.lium import LiumDeploymentMixin
from gradientsio.deployments.local import LocalDeploymentMixin
from gradientsio.deployments.local import LocalVLLMDeployment
from gradientsio.deployments.runpod import RunPodDeployment
from gradientsio.deployments.runpod import RunPodDeploymentMixin
from gradientsio.deployments.targon import TargonDeployment
from gradientsio.deployments.targon import TargonDeploymentMixin


class DeploymentClient(RunPodDeploymentMixin, LiumDeploymentMixin, TargonDeploymentMixin, LocalDeploymentMixin):
    def __init__(
        self,
        *,
        runpod_base_url: str = RUNPOD_BASE_URL,
        lium_base_url: str = LIUM_BASE_URL,
        targon_base_url: str = TARGON_BASE_URL,
        timeout: float | httpx.Timeout = 60.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.runpod_api_key = os.getenv(RUNPOD_API_KEY_ENV)
        self.lium_api_key = os.getenv(LIUM_API_KEY_ENV)
        self.targon_api_key = os.getenv(TARGON_API_KEY_ENV)
        self.runpod_base_url = runpod_base_url.rstrip("/")
        self.lium_base_url = lium_base_url.rstrip("/")
        self.targon_base_url = targon_base_url.rstrip("/")
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def deploy_runpod(**kwargs: Any) -> RunPodDeployment:
    client_kwargs = {
        key: kwargs.pop(key)
        for key in ("runpod_base_url", "lium_base_url", "targon_base_url", "timeout", "http_client")
        if key in kwargs
    }
    client = DeploymentClient(**client_kwargs)
    return client.deploy_runpod(**kwargs)


def deploy_lium(**kwargs: Any) -> LiumDeployment:
    client_kwargs = {
        key: kwargs.pop(key)
        for key in ("runpod_base_url", "lium_base_url", "targon_base_url", "timeout", "http_client")
        if key in kwargs
    }
    client = DeploymentClient(**client_kwargs)
    return client.deploy_lium(**kwargs)


def deploy_targon(**kwargs: Any) -> TargonDeployment:
    client_kwargs = {
        key: kwargs.pop(key)
        for key in ("runpod_base_url", "lium_base_url", "targon_base_url", "timeout", "http_client")
        if key in kwargs
    }
    client = DeploymentClient(**client_kwargs)
    return client.deploy_targon(**kwargs)


def deploy_local_vllm(**kwargs: Any) -> LocalVLLMDeployment:
    client_kwargs = {
        key: kwargs.pop(key)
        for key in ("runpod_base_url", "lium_base_url", "targon_base_url", "timeout", "http_client")
        if key in kwargs
    }
    client = DeploymentClient(**client_kwargs)
    return client.deploy_local_vllm(**kwargs)
