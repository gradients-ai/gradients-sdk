from __future__ import annotations

import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from typing import Any

import httpx
from pydantic import BaseModel

from gradientsio.errors import APIError
from gradientsio.errors import AuthenticationError
from gradientsio.errors import AuthorizationError
from gradientsio.errors import NetworkError
from gradientsio.errors import NotFoundError
from gradientsio.errors import RateLimitError
from gradientsio.errors import ValidationError


SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def _sdk_version() -> str:
    try:
        return version("gradientsio")
    except PackageNotFoundError:
        return "0.1.0"


def _serialize_json(data: Any | None) -> Any | None:
    if data is None:
        return None
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json", exclude_none=True)
    if isinstance(data, dict):
        return {key: _serialize_json(value) for key, value in data.items() if value is not None}
    if isinstance(data, list):
        return [_serialize_json(value) for value in data]
    return data


def _message_from_response(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or response.reason_phrase

    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str):
        return detail
    if detail is not None:
        return str(detail)
    if isinstance(payload, dict) and "message" in payload:
        return str(payload["message"])
    return response.reason_phrase


def _raise_for_response(response: httpx.Response) -> None:
    if response.status_code < 400:
        return

    message = _message_from_response(response)
    request_id = response.headers.get("x-request-id")
    kwargs = {"status_code": response.status_code, "response": response, "request_id": request_id}

    if response.status_code == 401:
        raise AuthenticationError(message, **kwargs)
    if response.status_code == 403:
        raise AuthorizationError(message, **kwargs)
    if response.status_code == 404:
        raise NotFoundError(message, **kwargs)
    if response.status_code == 422 or response.status_code == 400:
        raise ValidationError(message, **kwargs)
    if response.status_code == 429:
        raise RateLimitError(message, **kwargs)
    raise APIError(message, **kwargs)


class Transport:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout: float | httpx.Timeout = 60.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        auth_token: str | None = None,
    ) -> Any:
        method = method.upper()
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {
            "Accept": "application/json",
            "User-Agent": f"gradientsio-python/{_sdk_version()}",
        }
        if json is not None:
            headers["Content-Type"] = "application/json"
        token = auth_token if auth_token is not None else self.api_key
        if token:
            headers["Authorization"] = f"Bearer {token}"

        attempts = self.max_retries + 1 if method in SAFE_METHODS else 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                response = self._client.request(
                    method,
                    url,
                    headers=headers,
                    params={key: value for key, value in (params or {}).items() if value is not None},
                    json=_serialize_json(json),
                )
                _raise_for_response(response)
                if response.status_code == 204 or not response.content:
                    return None
                return response.json()
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == attempts - 1:
                    raise NetworkError(str(exc)) from exc

            time.sleep(min(0.5 * (2**attempt), 4.0))

        raise NetworkError(str(last_error) if last_error else "Request failed")

    def get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        auth_token: str | None = None,
    ) -> Any:
        return self.request("GET", path, params=params, auth_token=auth_token)

    def post(self, path: str, *, json: Any | None = None, auth_token: str | None = None) -> Any:
        return self.request("POST", path, json=json, auth_token=auth_token)

    def put(self, path: str, *, json: Any | None = None, auth_token: str | None = None) -> Any:
        return self.request("PUT", path, json=json, auth_token=auth_token)

    def delete(self, path: str, *, auth_token: str | None = None) -> Any:
        return self.request("DELETE", path, auth_token=auth_token)
