from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import Any

from gradients.errors import TaskFailed
from gradients.errors import TaskTimeout
from gradients.models import SchedulerJobDetails
from gradients.models import SchedulerJobRequest
from gradients.models import SchedulerJobResponse
from gradients.models import SchedulerJobResults


if TYPE_CHECKING:
    from gradients._transport import Transport


class SchedulerJob:
    def __init__(self, scheduler: "SchedulerClient", job_id: str, details: SchedulerJobDetails | None = None) -> None:
        self._scheduler = scheduler
        self.job_id = job_id
        self.details = details

    def refresh(self) -> SchedulerJobDetails:
        self.details = self._scheduler.get(self.job_id)
        return self.details

    def results(self) -> SchedulerJobResults:
        return self._scheduler.results(self.job_id)

    def delete(self) -> None:
        self._scheduler.delete(self.job_id)

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
                    raise TaskFailed(f"Scheduler job {self.job_id} failed")
                return details

            if timeout is not None and time.monotonic() - started_at >= timeout:
                raise TaskTimeout(f"Scheduler job {self.job_id} did not finish within {timeout} seconds")

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
        response = SchedulerJobResponse(**self._transport.post("/v1/scheduler/jobs/create", json=request))
        return SchedulerJob(self, response.job_id)

    def list(self) -> list[SchedulerJobDetails]:
        payload = self._transport.get("/v1/scheduler/jobs")
        return [SchedulerJobDetails(**item) for item in payload]

    def get(self, job_id: str) -> SchedulerJobDetails:
        return SchedulerJobDetails(**self._transport.get(f"/v1/scheduler/jobs/{job_id}"))

    def handle(self, job_id: str) -> SchedulerJob:
        return SchedulerJob(self, job_id)

    def results(self, job_id: str) -> SchedulerJobResults:
        return SchedulerJobResults(**self._transport.get(f"/v1/scheduler/jobs/{job_id}/results"))

    def delete(self, job_id: str) -> None:
        self._transport.delete(f"/v1/scheduler/jobs/{job_id}")
