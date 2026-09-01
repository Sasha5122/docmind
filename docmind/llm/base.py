"""One contract for every language model backend.

`LLM_BACKEND=azure|anthropic|ollama` picks the implementation; the rest of the code only
ever sees `LLM.complete(system, user) -> LLMResponse`. That is the data-residency switch:
the same question can be answered by a cloud model or by a model running on this machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float  # 0 for local models
    latency_s: float


class LLM(Protocol):
    name: str  # "azure" | "anthropic" | "ollama" | "fake"
    model: str

    def complete(
        self, system: str, user: str, max_tokens: int = 800, temperature: float = 0.0
    ) -> LLMResponse:
        """Send one system + user message and return the model's reply."""
        ...


# USD per 1M tokens (input, output). Kept here so cost per query is computed, not guessed.
# Prices change; update these when you re-measure and note the date in DECISIONS.md.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of one call; unknown models (and local ones) cost 0."""
    key = next((k for k in PRICES_PER_MTOK if model.startswith(k)), None)
    if key is None:
        return 0.0
    price_in, price_out = PRICES_PER_MTOK[key]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
