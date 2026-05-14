from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class SDKModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True, protected_namespaces=())


class FileFormat(str, Enum):
    CSV = "csv"
    JSON = "json"
    HF = "hf"
    S3 = "s3"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PREPARING_DATA = "preparing_data"
    PREP_TASK_FAILURE = "prep_task_failure"
    LOOKING_FOR_NODES = "looking_for_nodes"
    FAILURE_FINDING_NODES = "failure_finding_nodes"
    DELAYED = "delayed"
    READY = "ready"
    TRAINING = "training"
    PREEVALUATION = "preevaluation"
    EVALUATING = "evaluating"
    SUCCESS = "success"
    FAILURE = "failure"

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_TASK_STATUSES

    @property
    def is_success(self) -> bool:
        return self == TaskStatus.SUCCESS

    @property
    def is_failure(self) -> bool:
        return self in FAILURE_TASK_STATUSES


class TaskType(str, Enum):
    INSTRUCT_TEXT = "InstructTextTask"
    IMAGE = "ImageTask"
    DPO = "DpoTask"
    GRPO = "GrpoTask"
    CHAT = "ChatTask"


class ImageModelType(str, Enum):
    FLUX = "flux"
    SDXL = "sdxl"
    Z_IMAGE = "z-image"
    QWEN_IMAGE = "qwen-image"


class Backend(str, Enum):
    RUNPOD = "runpod"
    OBLIVUS = "oblivus"


ACTIVE_TASK_STATUSES = {
    TaskStatus.PENDING,
    TaskStatus.PREPARING_DATA,
    TaskStatus.LOOKING_FOR_NODES,
    TaskStatus.DELAYED,
    TaskStatus.READY,
    TaskStatus.TRAINING,
    TaskStatus.PREEVALUATION,
    TaskStatus.EVALUATING,
}
FAILURE_TASK_STATUSES = {
    TaskStatus.FAILURE,
    TaskStatus.PREP_TASK_FAILURE,
    TaskStatus.FAILURE_FINDING_NODES,
}
TERMINAL_TASK_STATUSES = {TaskStatus.SUCCESS, *FAILURE_TASK_STATUSES}


class RewardFunctionReference(SDKModel):
    reward_id: str
    reward_weight: float = Field(..., ge=0)


class ImageTextPair(SDKModel):
    image_url: str
    text_url: str


class TaskRequest(SDKModel):
    hours_to_complete: float = Field(..., gt=0)
    result_model_name: str | None = None
    backend: Backend | str | None = Backend.RUNPOD
    yarn_factor: int | None = None
    account_id: str | None = None


class InstructTaskRequest(TaskRequest):
    ds_repo: str
    model_repo: str
    file_format: FileFormat | str = FileFormat.HF
    field_instruction: str
    field_input: str | None = None
    field_output: str | None = None
    field_system: str | None = None


class ChatTaskRequest(TaskRequest):
    ds_repo: str
    model_repo: str
    file_format: FileFormat | str = FileFormat.HF
    chat_template: str
    chat_column: str | None = None
    chat_role_field: str | None = None
    chat_content_field: str | None = None
    chat_user_reference: str | None = None
    chat_assistant_reference: str | None = None


class DPOTaskRequest(TaskRequest):
    ds_repo: str
    model_repo: str
    file_format: FileFormat | str = FileFormat.HF
    field_prompt: str
    field_chosen: str
    field_rejected: str
    field_system: str | None = None
    prompt_format: str | None = None
    chosen_format: str | None = None
    rejected_format: str | None = None


class GRPOTaskRequest(TaskRequest):
    ds_repo: str
    model_repo: str
    file_format: FileFormat | str = FileFormat.HF
    field_prompt: str
    reward_functions: list[RewardFunctionReference]
    extra_column: str | None = None


class ImageTaskRequest(TaskRequest):
    model_repo: str
    image_text_pairs: list[ImageTextPair]
    ds_id: str | None = None
    model_type: ImageModelType | str = ImageModelType.SDXL


class ImageZipTaskRequest(TaskRequest):
    model_repo: str
    ds: str
    model_type: ImageModelType | str = ImageModelType.SDXL


class CustomDatasetTextTaskRequest(InstructTaskRequest):
    ds_repo: str | None = None
    training_data: str
    test_data: str | None = None
    file_format: FileFormat | str = FileFormat.S3


class CustomDatasetChatTaskRequest(ChatTaskRequest):
    ds_repo: str | None = None
    training_data: str
    test_data: str | None = None
    file_format: FileFormat | str = FileFormat.S3


class CreateTaskResponse(SDKModel):
    success: bool
    task_id: str | None
    created_at: datetime | None = None
    account_id: str | None = None


class TaskDetails(SDKModel):
    id: str | None = None
    task_id: str | None = None
    account_id: str | None = None
    status: TaskStatus | str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime | None = None
    hours_to_complete: float | None = None
    trained_model_repository: str | None = None
    task_type: TaskType | str | None = None
    result_model_name: str | None = None
    base_model_repository: str | None = None
    ds_repo: str | None = None

    @property
    def resolved_task_id(self) -> str | None:
        return self.task_id or self.id

    @property
    def normalized_status(self) -> TaskStatus | None:
        try:
            return TaskStatus(self.status)
        except ValueError:
            return None

    @property
    def is_terminal(self) -> bool:
        status = self.normalized_status
        return bool(status and status.is_terminal)

    @property
    def is_success(self) -> bool:
        status = self.normalized_status
        return bool(status and status.is_success)

    @property
    def is_failure(self) -> bool:
        status = self.normalized_status
        return bool(status and status.is_failure)


class PriceQuote(SDKModel):
    amount: float | None = None
    price: float | None = None
    total_price: float | None = None
    currency: str | None = None
    raw: dict[str, Any] | None = None


class SchedulerDataset(SDKModel):
    name: str
    field_instruction: str | None = None
    field_input: str | None = None
    field_output: str | None = None
    chat_column: str | None = None
    chat_role_field: str | None = None
    chat_content_field: str | None = None
    chat_user_reference: str | None = None
    chat_assistant_reference: str | None = None
    chat_template: str | None = None
    max_rows: int | None = None


class SchedulerJobRequest(SDKModel):
    task_type: str
    model_repo: str
    hours_to_complete: int
    samples_per_training: int
    final_test_size: float
    datasets: list[SchedulerDataset]
    name: str | None = None
    random_seed: int | None = None
    min_days: int | None = None
    max_days: int | None = None
    min_hours: int | None = None
    max_hours: int | None = None
    per_chunk_test_proportion: float | None = None


class SchedulerJobResponse(SDKModel):
    id: str
    status: str | None = None
    message: str | None = None


class SchedulerJobDetails(SDKModel):
    id: str
    status: str

    @property
    def is_terminal(self) -> bool:
        return self.status in {"completed", "suspended", "failed"}


class SchedulerTaskResult(SDKModel):
    task_id: str | None = None
    training_number: int | None = None
    status: str | None = None
    base_model_repo: str | None = None
    trained_model_repo: str | None = None
    merged_model_repo: str | None = None
    test_loss: float | None = None
    quality_score: float | None = None
    winner_hotkey: str | None = None
    error_message: str | None = None


class SchedulerJobResults(SDKModel):
    id: str | None = None
    status: str | None = None
    results: list[SchedulerTaskResult] = Field(default_factory=list)

    @property
    def latest_merged_model_repo(self) -> str | None:
        for result in reversed(self.results):
            if result.merged_model_repo:
                return result.merged_model_repo
        return None
