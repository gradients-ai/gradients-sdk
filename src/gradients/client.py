from __future__ import annotations

import os
from types import TracebackType

import httpx

from gradients._transport import Transport
from gradients.chutes import ChutesClient
from gradients.grpo import GRPOClient
from gradients.network import NetworkClient
from gradients.scheduler import SchedulerClient
from gradients.tasks import TasksClient


DEFAULT_BASE_URL = "https://api.gradients.io"
API_KEY_ENV = "GRADIENTS_API_KEY"
BASE_URL_ENV = "GRADIENTS_API_URL"


class GradientsClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | httpx.Timeout = 30.0,
        max_retries: int = 2,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv(API_KEY_ENV)
        self.base_url = base_url or os.getenv(BASE_URL_ENV) or DEFAULT_BASE_URL
        self._transport = Transport(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
            max_retries=max_retries,
            client=http_client,
        )

        self.tasks = TasksClient(self._transport)
        self.scheduler = SchedulerClient(self._transport)
        self.grpo = GRPOClient(self._transport)
        self.network = NetworkClient(self._transport)
        self.chutes = ChutesClient(self._transport)

    @classmethod
    def from_env(cls, **kwargs: object) -> "GradientsClient":
        return cls(**kwargs)

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> "GradientsClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
