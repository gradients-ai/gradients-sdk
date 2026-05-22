from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gradientsio.deployments import DeploymentClient  # noqa: E402
from gradientsio.deployments.basilica import _build_basilica_vllm_args  # noqa: E402
from gradientsio.errors import ConfigurationError  # noqa: E402


class FakeBasilicaHTTPClient:
    def __init__(self, *, deployments: list[dict[str, Any]] | None = None) -> None:
        self.deployments = deployments if deployments is not None else []
        self.requests: list[tuple[str, str, dict[str, Any] | None, dict[str, str] | None]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        self.requests.append((method, url, json, params))
        assert headers["Authorization"] == "Bearer basilica-test"
        if method == "GET" and url.endswith("/deployments"):
            return _json_response({"deployments": self.deployments, "total": len(self.deployments)})
        if method == "GET" and "/deployments/" in url and not url.endswith("/logs"):
            name = url.rsplit("/", 1)[1]
            for deployment in self.deployments:
                if deployment["instanceName"] == name:
                    return _json_response(deployment)
            return _json_response({"message": "not found"}, status_code=404)
        if method == "POST" and url.endswith("/deployments"):
            assert json is not None
            deployment = {
                "instanceName": json["instanceName"],
                "state": "Active",
                "url": "https://basilica.test",
                "env": json["env"],
            }
            self.deployments.append(deployment)
            return _json_response(deployment)
        if method == "DELETE" and "/deployments/" in url:
            name = url.rsplit("/", 1)[1]
            self.deployments = [item for item in self.deployments if item["instanceName"] != name]
            return _json_response({"instanceName": name, "state": "Deleted", "message": "deleted"})
        if method == "GET" and url.endswith("/logs"):
            return httpx.Response(200, text="vLLM ready\n")
        return _json_response({}, status_code=404)

    def get(self, url: str, *, timeout: float, follow_redirects: bool = False) -> httpx.Response:
        if url == "https://basilica.test/v1/models":
            return _json_response({"data": []})
        return _json_response({}, status_code=404)

    def close(self) -> None:
        return None


def _json_response(payload: Any, *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


def test_basilica_requires_env_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BASILICA_API_KEY", raising=False)
    client = DeploymentClient(http_client=FakeBasilicaHTTPClient())

    with pytest.raises(ConfigurationError):
        client.deploy_basilica(base_model="Qwen/Qwen2.5-0.5B-Instruct", wait=False)


def test_build_basilica_vllm_args_uses_vllm_serve_shape() -> None:
    args = _build_basilica_vllm_args(
        "Qwen/Qwen2.5-0.5B-Instruct",
        [
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--model",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--served-model-name",
            "gradients-qwen",
        ],
    )

    assert args[:5] == ["Qwen/Qwen2.5-0.5B-Instruct", "--host", "0.0.0.0", "--port", "8000"]
    assert "--model" not in args
    assert "--served-model-name" in args


def test_deploy_basilica_creates_deployment_with_rest_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BASILICA_API_KEY", "basilica-test")
    http_client = FakeBasilicaHTTPClient()
    client = DeploymentClient(http_client=http_client)

    deployment = client.deploy_basilica(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=2048,
        wait=False,
    )

    create_request = [request for request in http_client.requests if request[0] == "POST"][0]
    payload = create_request[2]
    assert deployment.id.startswith("grad-basilica-")
    assert deployment.server_url == "https://basilica.test"
    assert payload is not None
    assert payload["image"] == "vllm/vllm-openai:latest"
    assert payload["command"] == ["vllm"]
    assert payload["args"][:2] == ["serve", "Qwen/Qwen2.5-0.5B-Instruct"]
    assert payload["resources"]["gpus"]["model"] == ["A100"]
    assert payload["resources"]["gpus"]["minGpuMemoryGb"] == 80
    assert payload["env"]["GRADIENTS_SDK_PROVIDER"] == "basilica"
    assert payload["healthCheck"]["liveness"]["initialDelaySeconds"] == 1800
    assert payload["healthCheck"]["startup"]["failureThreshold"] == 180


def test_deploy_basilica_reuses_existing_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BASILICA_API_KEY", "basilica-test")
    existing = {
        "instanceName": "grad-basilica-existing",
        "state": "Active",
        "url": "https://basilica.test",
        "env": {"GRADIENTS_DEPLOYMENT_KEY": "key"},
    }
    client = DeploymentClient(http_client=FakeBasilicaHTTPClient(deployments=[existing]))

    deployment = client.deploy_basilica(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        deployment_name="grad-basilica-existing",
        wait=False,
    )

    assert deployment.id == "grad-basilica-existing"
    assert deployment.server_url == "https://basilica.test"


def test_delete_basilica_uses_deployment_delete_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BASILICA_API_KEY", "basilica-test")
    http_client = FakeBasilicaHTTPClient()
    client = DeploymentClient(http_client=http_client)

    client.delete_basilica("grad-basilica-delete")

    assert http_client.requests[-1][0] == "DELETE"
    assert http_client.requests[-1][1].endswith("/deployments/grad-basilica-delete")
