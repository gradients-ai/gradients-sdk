from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any


if TYPE_CHECKING:
    from gradients._transport import Transport


class ChutesClient:
    def __init__(self, transport: "Transport") -> None:
        self._transport = transport

    def deploy(self, *, model_id: str, lora_id: str) -> dict[str, Any]:
        return self._transport.post("/v1/chutes/deploy", json={"model_id": model_id, "lora_id": lora_id})

    def status(self, chute_id: str) -> dict[str, Any]:
        return self._transport.get(f"/v1/chutes/status/{chute_id}")
