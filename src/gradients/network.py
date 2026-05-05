from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gradients.models import NetworkStatus


if TYPE_CHECKING:
    from gradients._transport import Transport


class NetworkClient:
    def __init__(self, transport: "Transport") -> None:
        self._transport = transport

    def status(self) -> NetworkStatus:
        return NetworkStatus(**self._transport.get("/v1/network/status"))

    def detailed_status(self) -> dict[str, Any]:
        return self._transport.get("/v1/network/detailed_status")

    def leaderboard(self) -> list[dict[str, Any]]:
        return self._transport.get("/v1/leaderboard")

    def latest_tournament_weights(self) -> dict[str, Any]:
        return self._transport.get("/v1/performance/latest-tournament-weights")

    def weight_projection(self) -> dict[str, Any]:
        return self._transport.get("/v1/performance/weight-projection")

    def last_boss_battle(self) -> dict[str, Any]:
        return self._transport.get("/v1/performance/last-boss-battle")
