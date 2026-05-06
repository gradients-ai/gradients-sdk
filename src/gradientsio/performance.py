from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any


if TYPE_CHECKING:
    from gradientsio._transport import Transport


class PerformanceClient:
    def __init__(self, transport: "Transport") -> None:
        self._transport = transport

    def latest_tournament_weights(self) -> dict[str, Any]:
        return self._transport.get("/v1/performance/latest-tournament-weights")

    def weight_projection(self, *, percentage_improvement: float) -> dict[str, Any]:
        return self._transport.get(
            "/v1/performance/weight-projection",
            params={"percentage_improvement": percentage_improvement},
        )

    def weight_projection_static(self) -> dict[str, Any]:
        return self._transport.get("/v1/performance/weight-projection-static")

    def last_boss_battle(self) -> dict[str, Any]:
        return self._transport.get("/v1/performance/last-boss-battle")
