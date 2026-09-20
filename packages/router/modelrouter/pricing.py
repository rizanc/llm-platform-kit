"""Model registry and prices.

Prices come from LiteLLM's community-maintained price table when litellm is
installed (`litellm.model_cost`, refreshed with each litellm release). When it
is not, the pinned snapshot below is used. The snapshot is dated; treat it as
a fallback, not a source of truth.

Tiers are a routing concept, not a vendor concept: "cheap" models handle
short factual questions, "medium" handle code and multi-step reasoning,
"expensive" is reserved for long or hard requests.
"""
from __future__ import annotations

from dataclasses import dataclass

SNAPSHOT_DATE = "2026-09-20"


@dataclass(frozen=True)
class ModelInfo:
    name: str
    tier: str  # "cheap" | "medium" | "expensive"
    input_cost_per_mtok: float   # USD per 1M input tokens
    output_cost_per_mtok: float  # USD per 1M output tokens
    max_input_tokens: int
    supports: tuple[str, ...] = ()


# (name, tier, input $/Mtok, output $/Mtok, max input tokens) as of SNAPSHOT_DATE
_SNAPSHOT = [
    ("claude-haiku-4-5", "cheap",     1.00,  5.00,   200_000),
    ("gpt-5-mini",       "cheap",     0.25,  2.00,   272_000),
    ("claude-sonnet-5",  "medium",    2.00, 10.00, 1_000_000),
    ("gpt-5",            "medium",    1.25, 10.00,   272_000),
    ("claude-opus-5",    "expensive", 5.00, 25.00, 1_000_000),
]

TIER_OF = {name: tier for name, tier, *_ in _SNAPSHOT}


def _from_litellm(name: str) -> tuple[float, float, int] | None:
    try:
        import litellm  # noqa: WPS433 - optional dependency
    except ImportError:
        return None
    entry = litellm.model_cost.get(name)
    if not entry or entry.get("input_cost_per_token") is None:
        return None
    return (
        float(entry["input_cost_per_token"]) * 1_000_000,
        float(entry.get("output_cost_per_token", 0.0)) * 1_000_000,
        int(entry.get("max_input_tokens") or entry.get("max_tokens") or 128_000),
    )


def build_registry(prefer_litellm: bool = True) -> dict[str, ModelInfo]:
    """Default models with live prices when litellm is available, snapshot otherwise."""
    out = {}
    for name, tier, in_cost, out_cost, max_in in _SNAPSHOT:
        live = _from_litellm(name) if prefer_litellm else None
        if live:
            in_cost, out_cost, max_in = live
        out[name] = ModelInfo(name, tier, in_cost, out_cost, max_in)
    return out


def compute_cost(model: ModelInfo, input_tokens: int, output_tokens: int) -> float:
    return model.input_cost_per_mtok * input_tokens / 1e6 + model.output_cost_per_mtok * output_tokens / 1e6
