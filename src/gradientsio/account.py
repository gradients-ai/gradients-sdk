from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gradientsio.errors import ConfigurationError


if TYPE_CHECKING:
    from gradientsio._transport import Transport


SESSION_TOKEN_REQUIRED_MESSAGE = (
    "Account endpoints require a Gradients session token, not a Gradients API key. "
    "Set GRADIENTS_SESSION_TOKEN or pass session_token=... to GradientsClient to use "
    "client.account.get_info() and client.account.get_public_key()."
)


class AccountClient:
    def __init__(self, transport: "Transport", *, session_token: str | None = None) -> None:
        self._transport = transport
        self.session_token = session_token

    def get_info(self) -> dict[str, Any]:
        return self._transport.post("/account-get-info", auth_token=self._require_session_token())

    def get_public_key(self) -> dict[str, Any]:
        return self._transport.post("/account-get-public-key", auth_token=self._require_session_token())

    def _require_session_token(self) -> str:
        if not self.session_token:
            raise ConfigurationError(SESSION_TOKEN_REQUIRED_MESSAGE)
        return self.session_token
