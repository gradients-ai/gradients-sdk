from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gradients.models import AddRewardFunctionResponse
from gradients.models import RewardFunctionsResponse


if TYPE_CHECKING:
    from gradients._transport import Transport


class GRPOClient:
    def __init__(self, transport: "Transport") -> None:
        self._transport = transport

    def list_reward_functions(self) -> RewardFunctionsResponse:
        return RewardFunctionsResponse(**self._transport.get("/v1/grpo/reward_functions"))

    def add_reward_function(
        self,
        *,
        name: str,
        description: str,
        code: str,
        reward_weight: float | None = None,
        **extra: Any,
    ) -> AddRewardFunctionResponse:
        payload = {
            "name": name,
            "description": description,
            "code": code,
            "reward_weight": reward_weight,
            **extra,
        }
        return AddRewardFunctionResponse(**self._transport.post("/v1/grpo/reward_functions", json=payload))
