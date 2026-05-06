from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import Any

from gradientsio.errors import TaskFailed
from gradientsio.errors import TaskTimeout
from gradientsio.models import SchedulerJobDetails
from gradientsio.models import SchedulerJobRequest
from gradientsio.models import SchedulerJobResponse
from gradientsio.models import SchedulerJobResults


if TYPE_CHECKING:
    from gradientsio._transport import Transport


class SchedulerJob:
    def __init__(self, scheduler: "SchedulerClient", id: str, details: SchedulerJobDetails | None = None) -> None:
        self._scheduler = scheduler
        self.id = id
        self.details = details

    def refresh(self) -> SchedulerJobDetails:
        self.details = self._scheduler.get(self.id)
        return self.details

    def results(self) -> SchedulerJobResults:
        return self._scheduler.results(self.id)

    def delete(self) -> None:
        self._scheduler.delete(self.id)

    def wait(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float = 600.0,
        raise_on_failure: bool = True,
    ) -> SchedulerJobDetails:
        started_at = time.monotonic()

        while True:
            details = self.refresh()
            if details.status in {"completed", "suspended"}:
                return details
            if details.status == "failed":
                if raise_on_failure:
                    raise TaskFailed(f"Scheduler job {self.id} failed")
                return details

            if timeout is not None and time.monotonic() - started_at >= timeout:
                raise TaskTimeout(f"Scheduler job {self.id} did not finish within {timeout} seconds")

            sleep_for = poll_interval
            if timeout is not None:
                remaining = timeout - (time.monotonic() - started_at)
                sleep_for = max(0.0, min(poll_interval, remaining))
            time.sleep(sleep_for)


class SchedulerClient:
    def __init__(self, transport: "Transport") -> None:
        self._transport = transport

    def health(self) -> dict[str, Any]:
        return self._transport.get("/v1/scheduler/health")

    def create_job(self, **kwargs: Any) -> SchedulerJob:
        request = SchedulerJobRequest(**kwargs)
        payload = self._transport.post("/v1/scheduler/jobs/create", json=request)
        if "id" not in payload and "job_id" in payload:
            payload["id"] = payload.pop("job_id")
        response = SchedulerJobResponse(**payload)
        return SchedulerJob(self, response.id)

    def list(self) -> list[SchedulerJobDetails]:
        payload = self._transport.get("/v1/scheduler/jobs")
        return [SchedulerJobDetails(**item) for item in payload]

    def get(self, id: str) -> SchedulerJobDetails:
        return SchedulerJobDetails(**self._transport.get(f"/v1/scheduler/jobs/{id}"))

    def handle(self, id: str) -> SchedulerJob:
        return SchedulerJob(self, id)

    def results(self, id: str) -> SchedulerJobResults:
        return SchedulerJobResults(**self._transport.get(f"/v1/scheduler/jobs/{id}/results"))

    def delete(self, id: str) -> None:
        self._transport.delete(f"/v1/scheduler/jobs/{id}")
