from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Any

import httpx

from gradientsio._transport import Transport
from gradientsio.account import AccountClient
from gradientsio.performance import PerformanceClient
from gradientsio.scheduler import SchedulerClient
from gradientsio.tasks import TasksClient
from gradientsio.tasks import TrainingTask


DEFAULT_BASE_URL = "https://api.gradients.io"
API_KEY_ENV = "GRADIENTS_API_KEY"
SESSION_TOKEN_ENV = "GRADIENTS_SESSION_TOKEN"


class GradientsClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        session_token: str | None = None,
        timeout: float | httpx.Timeout = 60.0,
        max_retries: int = 2,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv(API_KEY_ENV)
        self.session_token = session_token if session_token is not None else os.getenv(SESSION_TOKEN_ENV)
        self.base_url = DEFAULT_BASE_URL
        self._transport = Transport(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
            max_retries=max_retries,
            client=http_client,
        )

        self.account = AccountClient(self._transport, session_token=self.session_token)
        self.tasks = TasksClient(self._transport)
        self.scheduler = SchedulerClient(self._transport)
        self.performance = PerformanceClient(self._transport)

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

    def train(
        self,
        *,
        model: str,
        task_type: TaskType | str,
        hours: float,
        dataset: Datasets | str | dict[str, Any],
        **kwargs: Any,
    ) -> TrainingTask:
        trainer = Gradientsio(
            model=model,
            tasktype=task_type,
            hours=hours,
            data=dataset,
            client=self,
            **kwargs,
        )
        return trainer.train()


class TaskType(str, Enum):
    INSTRUCT = "instruct"
    CHAT = "chat"
    DPO = "dpo"
    GRPO = "grpo"
    IMAGE = "image"


@dataclass(frozen=True)
class Datasets:
    source: str
    format: str = "hf"
    training_data: str | None = None
    test_data: str | None = None

    @classmethod
    def HF(cls, dataset: str) -> "Datasets":
        return cls(source=dataset, format="hf")

    @classmethod
    def S3(cls, training_data: str, *, test_data: str | None = None, source: str | None = None) -> "Datasets":
        return cls(source=source or training_data, format="s3", training_data=training_data, test_data=test_data)


class Gradientsio:
    def __init__(
        self,
        *,
        model: str,
        tasktype: TaskType | str,
        hours: float,
        data: Datasets | str | dict[str, Any],
        api_key: str | None = None,
        client: GradientsClient | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.tasktype = self._coerce_tasktype(tasktype)
        self.hours = hours
        self.data = self._coerce_data(data)
        self.extra = kwargs
        self.client = client or GradientsClient(api_key=api_key)

    def train(self) -> TrainingTask:
        payload = self._base_payload()

        if self.tasktype == TaskType.INSTRUCT:
            payload.update(
                {
                    "field_instruction": self.extra.pop("field_instruction", "instruction"),
                    "field_input": self.extra.pop("field_input", None),
                    "field_output": self.extra.pop("field_output", "output"),
                    "field_system": self.extra.pop("field_system", None),
                }
            )
            payload.update(self.extra)
            if self.data.training_data:
                return self.client.tasks.create_custom_dataset_text(**payload)
            return self.client.tasks.create_instruct(**payload)

        if self.tasktype == TaskType.CHAT:
            payload.update(
                {
                    "chat_template": self.extra.pop("chat_template", "chatml"),
                    "chat_column": self.extra.pop("chat_column", "conversations"),
                    "chat_role_field": self.extra.pop("chat_role_field", "from"),
                    "chat_content_field": self.extra.pop("chat_content_field", "value"),
                    "chat_user_reference": self.extra.pop("chat_user_reference", "user"),
                    "chat_assistant_reference": self.extra.pop("chat_assistant_reference", "assistant"),
                }
            )
            payload.update(self.extra)
            if self.data.training_data:
                return self.client.tasks.create_custom_dataset_chat(**payload)
            return self.client.tasks.create_chat(**payload)

        if self.tasktype == TaskType.DPO:
            payload.update(
                {
                    "field_prompt": self.extra.pop("field_prompt", "prompt"),
                    "field_chosen": self.extra.pop("field_chosen", "chosen"),
                    "field_rejected": self.extra.pop("field_rejected", "rejected"),
                    "field_system": self.extra.pop("field_system", None),
                }
            )
            payload.update(self.extra)
            return self.client.tasks.create_dpo(**payload)

        if self.tasktype == TaskType.GRPO:
            payload.update(
                {
                    "field_prompt": self.extra.pop("field_prompt", "prompt"),
                    "reward_functions": self.extra.pop("reward_functions"),
                    "extra_column": self.extra.pop("extra_column", None),
                }
            )
            payload.update(self.extra)
            return self.client.tasks.create_grpo(**payload)

        if self.tasktype == TaskType.IMAGE:
            if "image_text_pairs" in self.extra:
                payload = {
                    "model_repo": self.model,
                    "hours_to_complete": self.hours,
                    "image_text_pairs": self.extra.pop("image_text_pairs"),
                    "model_type": self.extra.pop("model_type", "sdxl"),
                    **self.extra,
                }
                return self.client.tasks.create_image(**payload)

            payload = {
                "model_repo": self.model,
                "hours_to_complete": self.hours,
                "ds": self.extra.pop("ds", self.data.source),
                "model_type": self.extra.pop("model_type", "sdxl"),
                **self.extra,
            }
            return self.client.tasks.create_image_zip(**payload)

        raise ValueError(f"Unsupported task type: {self.tasktype}")

    def Train(self) -> TrainingTask:
        return self.train()

    @staticmethod
    def _coerce_tasktype(tasktype: TaskType | str) -> TaskType:
        if isinstance(tasktype, TaskType):
            return tasktype
        normalized = tasktype.lower()
        for member in TaskType:
            if normalized in {member.value, member.name.lower()}:
                return member
        raise ValueError(f"Unsupported task type: {tasktype}")

    @staticmethod
    def _coerce_data(data: Datasets | str | dict[str, Any]) -> Datasets:
        if isinstance(data, Datasets):
            return data
        if isinstance(data, str):
            return Datasets.HF(data)
        if "training_data" in data:
            return Datasets.S3(
                data["training_data"],
                test_data=data.get("test_data"),
                source=data.get("source") or data.get("ds_repo"),
            )
        if "hf" in data:
            return Datasets.HF(data["hf"])
        if "json" in data:
            return Datasets.S3(data["json"], test_data=data.get("test_data"), source=data.get("source"))
        if "s3" in data:
            return Datasets.S3(data["s3"], test_data=data.get("test_data"), source=data.get("source"))
        raise ValueError("dataset must be a string, Datasets object, or dict with hf/json/s3/training_data")

    def _base_payload(self) -> dict[str, Any]:
        payload = {
            "ds_repo": self.data.source,
            "model_repo": self.model,
            "file_format": self.data.format,
            "hours_to_complete": self.hours,
        }
        if self.data.training_data:
            payload["training_data"] = self.data.training_data
            payload["test_data"] = self.data.test_data
        return payload
