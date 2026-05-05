from __future__ import annotations

import httpx
import pytest

from gradients import AuthenticationError
from gradients import GradientsClient
from gradients import TaskFailed


def make_client(handler):
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    return GradientsClient(api_key="test-key", base_url="https://api.test", http_client=http_client)


def test_create_instruct_sends_bearer_auth_and_returns_task_handle():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/tasks/create"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = httpx.Request("POST", "https://unused", content=request.content).read()
        assert b"yahma/alpaca-cleaned" in payload
        return httpx.Response(200, json={"success": True, "task_id": "task-1", "account_id": "acct-1"})

    with make_client(handler) as client:
        task = client.tasks.create_instruct(
            ds_repo="yahma/alpaca-cleaned",
            model_repo="Qwen/Qwen2.5-7B-Instruct",
            hours_to_complete=1,
            field_instruction="instruction",
        )

    assert task.task_id == "task-1"


def test_wait_returns_successful_task_details():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        assert request.url.path == "/v1/tasks/task-1"
        calls += 1
        if calls == 1:
            return httpx.Response(200, json={"id": "task-1", "status": "training"})
        return httpx.Response(
            200,
            json={"id": "task-1", "status": "success", "trained_model_repository": "org/model"},
        )

    with make_client(handler) as client:
        details = client.tasks.handle("task-1").wait(poll_interval=0, timeout=1)

    assert details.trained_model_repository == "org/model"


def test_wait_raises_on_failure_state():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "task-1", "status": "failure"})

    with make_client(handler) as client:
        with pytest.raises(TaskFailed):
            client.tasks.handle("task-1").wait(poll_interval=0, timeout=1)


def test_http_401_maps_to_authentication_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "invalid token"})

    with make_client(handler) as client:
        with pytest.raises(AuthenticationError):
            client.tasks.get("task-1")
