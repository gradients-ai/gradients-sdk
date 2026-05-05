from __future__ import annotations

from typing import Any


class GradientsError(Exception):
    """Base exception for all SDK errors."""


class ConfigurationError(GradientsError):
    """Raised when SDK configuration is invalid."""


class NetworkError(GradientsError):
    """Raised when the API cannot be reached or returns malformed data."""


class APIError(GradientsError):
    """Raised for non-successful API responses."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response: Any | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = response
        self.request_id = request_id


class AuthenticationError(APIError):
    """Raised when an API key is missing, invalid, or expired."""


class AuthorizationError(APIError):
    """Raised when the caller is authenticated but not allowed to perform the action."""


class NotFoundError(APIError):
    """Raised when a requested resource does not exist."""


class ValidationError(APIError):
    """Raised when the API rejects a request payload."""


class RateLimitError(APIError):
    """Raised when the API rate limit is exceeded."""


class TaskFailed(GradientsError):
    """Raised when waiting on a task that reaches a failure state."""


class TaskTimeout(GradientsError):
    """Raised when a task or scheduler job does not finish before the timeout."""
