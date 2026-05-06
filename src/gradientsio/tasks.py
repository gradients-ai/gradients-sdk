from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import Any

from gradientsio.errors import TaskFailed
from gradientsio.errors import TaskTimeout
from gradientsio.models import ChatTaskRequest
from gradientsio.models import CreateTaskResponse
from gradientsio.models import CustomDatasetChatTaskRequest
from gradientsio.models import CustomDatasetTextTaskRequest
from gradientsio.models import DPOTaskRequest
from gradientsio.models import GRPOTaskRequest
from gradientsio.models import ImageTaskRequest
from gradientsio.models import InstructTaskRequest
from gradientsio.models import PriceQuote
from gradientsio.models import TaskDetails
from gradientsio.models import TaskStatus


if TYPE_CHECKING:
    from gradientsio._transport import Transport


def _price_quote(payload: Any) -> PriceQuote:
    if isinstance(payload, dict):
        return PriceQuote(**payload, raw=payload)
    return PriceQuote(raw={"value": payload})


def _require_task_id(response: CreateTaskResponse) -> str:
    if not response.task_id:
        raise TaskFailed("Task creation response did not include a task_id")
    return response.task_id


class TrainingTask:
    def __init__(self, tasks: "TasksClient", task_id: str, details: TaskDetails | None = None) -> None:
        self._tasks = tasks
        self.task_id = task_id
        self.details = details

    def refresh(self) -> TaskDetails:
        self.details = self._tasks.get(self.task_id)
        return self.details

    def status(self) -> TaskStatus | str:
        return self.refresh().status

    def delete(self) -> None:
        self._tasks.delete(self.task_id)

    def wait(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float = 600.0,
        raise_on_failure: bool = True,
    ) -> TaskDetails:
        started_at = time.monotonic()

        while True:
            details = self.refresh()
            if details.is_success:
                return details
            if details.is_failure:
                if raise_on_failure:
                    raise TaskFailed(f"Task {self.task_id} failed with status {details.status}")
                return details

            if timeout is not None and time.monotonic() - started_at >= timeout:
                raise TaskTimeout(f"Task {self.task_id} did not finish within {timeout} seconds")

            sleep_for = poll_interval
            if timeout is not None:
                remaining = timeout - (time.monotonic() - started_at)
                sleep_for = max(0.0, min(poll_interval, remaining))
            time.sleep(sleep_for)


class TasksClient:
    def __init__(self, transport: "Transport") -> None:
        self._transport = transport

    def check_text_price(self, **payload: Any) -> PriceQuote:
        return _price_quote(self._transport.post("/v1/tasks/text/check_price", json=payload))

    def check_image_price(self, **payload: Any) -> PriceQuote:
        return _price_quote(self._transport.post("/v1/tasks/image/check_price", json=payload))

    def prices(self) -> dict[str, Any]:
        return self._transport.get("/v1/prices")

    def create_instruct(self, **kwargs: Any) -> TrainingTask:
        request = InstructTaskRequest(**kwargs)
        response = CreateTaskResponse(**self._transport.post("/v1/tasks/create", json=request))
        return TrainingTask(self, _require_task_id(response))

    def create_chat(self, **kwargs: Any) -> TrainingTask:
        request = ChatTaskRequest(**kwargs)
        response = CreateTaskResponse(**self._transport.post("/v1/tasks/create_chat", json=request))
        return TrainingTask(self, _require_task_id(response))

    def create_dpo(self, **kwargs: Any) -> TrainingTask:
        request = DPOTaskRequest(**kwargs)
        response = CreateTaskResponse(**self._transport.post("/v1/tasks/create_dpo", json=request))
        return TrainingTask(self, _require_task_id(response))

    def create_grpo(self, **kwargs: Any) -> TrainingTask:
        request = GRPOTaskRequest(**kwargs)
        response = CreateTaskResponse(**self._transport.post("/v1/tasks/create_grpo", json=request))
        return TrainingTask(self, _require_task_id(response))

    def create_image(self, **kwargs: Any) -> TrainingTask:
        request = ImageTaskRequest(**kwargs)
        response = CreateTaskResponse(**self._transport.post("/v1/tasks/create_image", json=request))
        return TrainingTask(self, _require_task_id(response))

    def create_custom_dataset_text(self, **kwargs: Any) -> TrainingTask:
        request = CustomDatasetTextTaskRequest(**kwargs)
        response = CreateTaskResponse(**self._transport.post("/v1/tasks/create_custom_dataset_text", json=request))
        return TrainingTask(self, _require_task_id(response))

    def create_custom_dataset_chat(self, **kwargs: Any) -> TrainingTask:
        request = CustomDatasetChatTaskRequest(**kwargs)
        response = CreateTaskResponse(**self._transport.post("/v1/tasks/create_custom_dataset_chat", json=request))
        return TrainingTask(self, _require_task_id(response))

    def get(self, task_id: str) -> TaskDetails:
        return TaskDetails(**self._transport.get(f"/v1/tasks/{task_id}"))

    def handle(self, task_id: str) -> TrainingTask:
        return TrainingTask(self, task_id)

    def list(self, *, account_id: str, limit: int = 100, page: int = 1) -> list[TaskDetails]:
        payload = self._transport.get(f"/v1/tasks/account/{account_id}", params={"limit": limit, "page": page})
        return [TaskDetails(**item) for item in payload]

    def delete(self, task_id: str) -> None:
        self._transport.delete(f"/v1/tasks/delete/{task_id}")

    def completed_organic(
        self,
        *,
        hours: int | None = None,
        task_type: str | None = None,
        search_model_name: str | None = None,
        limit: int = 100,
        page: int = 1,
    ) -> list[TaskDetails]:
        payload = self._transport.get(
            "/v1/tasks/organic/completed",
            params={
                "hours": hours,
                "task_type": task_type,
                "search_model_name": search_model_name,
                "limit": limit,
                "page": page,
            },
        )
        return [TaskDetails(**item) for item in payload]

