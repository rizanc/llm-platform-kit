"""Cost-Optimized Model Router.

Routes queries to the cheapest model that can answer them, based on:
  - Complexity classifier (token count, code-detect, math-detect, multi-step)
  - Per-model capability tier (cheap/medium/expensive)
  - Per-model pricing (input $/Mtok, output $/Mtok)
  - Per-request budget cap

Tracks spend per request, per tenant, per model → JSONL spend log.
Exposes Prometheus metrics (counter + histogram) for project #5.

Production swap: any chat-completions API (OpenAI, Anthropic, LiteLLM).
Unit tests use a deterministic stub model registry.
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable


# --------------------------------------------------------- model registry
@dataclass(frozen=True)
class ModelInfo:
    name: str
    tier: str  # "cheap" | "medium" | "expensive"
    input_cost_per_mtok: float   # USD per 1M input tokens
    output_cost_per_mtok: float  # USD per 1M output tokens
    max_input_tokens: int
    supports: tuple[str, ...] = ()  # capabilities, e.g. ("tools", "vision", "json_mode")


# Sensible default registry (prices as of mid-2026; update in production)
DEFAULT_MODELS = {
    "gpt-4o-mini":      ModelInfo("gpt-4o-mini",     "cheap",     0.15, 0.60, 128_000),
    "claude-haiku":     ModelInfo("claude-haiku",    "cheap",     0.25, 1.25, 200_000),
    "gpt-4o":           ModelInfo("gpt-4o",          "medium",    2.50, 10.0, 128_000),
    "claude-sonnet":    ModelInfo("claude-sonnet",   "medium",    3.00, 15.0, 200_000),
    "o1":               ModelInfo("o1",              "expensive", 15.0, 60.0, 200_000),
    "claude-opus":      ModelInfo("claude-opus",     "expensive", 15.0, 75.0, 200_000),
}


# --------------------------------------------------------- complexity classifier
# Pure functions of the input string. Heuristic, deterministic, fast.

_CODE_PATTERNS = [
    re.compile(r"```"),                       # fenced code block
    re.compile(r"\bdef \w+\("),               # python function def
    re.compile(r"\bfunction \w+\("),          # js function
    re.compile(r"\bclass \w+[\(:]"),          # class declaration
    re.compile(r"=>|->|<\?php"),               # arrow/return/php
    re.compile(r"\bSELECT\b.*\bFROM\b", re.I),  # SQL
    re.compile(r"\bimport \w+\b"),            # imports
]
_MATH_PATTERNS = [
    re.compile(r"\b\d+\s*[+\-*/]\s*\d+"),     # simple arithmetic
    re.compile(r"[∫∑∏√∞≠≈≤≥]"),                # math symbols
    re.compile(r"\bequation\b|\bintegral\b|\bderivative\b", re.I),
    re.compile(r"\$[^$]+\$"),                 # latex inline
]
_MULTI_STEP_PATTERNS = [
    re.compile(r"\bstep[- ]by[- ]step\b", re.I),
    re.compile(r"\bfirst,?\s+then\b", re.I),
    re.compile(r"\b\d+\.\s"),                 # numbered list "1. ... 2. ..."
    re.compile(r"\bexplain\b.*\bwhy\b", re.I),
]


def estimate_complexity(text: str) -> dict:
    """Returns {score: float, signals: dict[str, bool], token_count: int}."""
    signals = {
        "has_code": any(p.search(text) for p in _CODE_PATTERNS),
        "has_math": any(p.search(text) for p in _MATH_PATTERNS),
        "is_multi_step": any(p.search(text) for p in _MULTI_STEP_PATTERNS),
        "is_long": len(text) > 2000,
    }
    # Weighted score; cheap=0, medium=1, expensive=2+
    score = 0.0
    if signals["has_code"]:
        score += 1.0
    if signals["has_math"]:
        score += 1.0
    if signals["is_multi_step"]:
        score += 1.5
    if signals["is_long"]:
        score += 0.5

    # Whitespace-token count is a fair proxy without a tokenizer
    token_count = len(text.split())
    if token_count > 500:
        score += 1.0

    return {"score": score, "signals": signals, "token_count": token_count}


def tier_for_complexity(score: float) -> str:
    if score < 1.0:
        return "cheap"
    if score < 3.0:
        return "medium"
    return "expensive"


# --------------------------------------------------------- cost
def compute_cost(model: ModelInfo, input_tokens: int, output_tokens: int) -> float:
    """USD cost for a single request."""
    return (
        model.input_cost_per_mtok * input_tokens / 1_000_000
        + model.output_cost_per_mtok * output_tokens / 1_000_000
    )


# --------------------------------------------------------- request/response
@dataclass
class RouteRequest:
    request_id: str
    text: str
    tenant: str = "default"
    force_tier: str | None = None  # override ("cheap"/"medium"/"expensive") for testing
    budget_cap_usd: float | None = None


@dataclass
class RouteResponse:
    request_id: str
    model: str
    tier: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    complexity: dict
    latency_ms: float
    text: str


# --------------------------------------------------------- router
class ModelRouter:
    """Routes requests to the cheapest model that fits the complexity tier.

    Tracks spend per (tenant, model) → spend.jsonl.
    Optional Prometheus exporter for project #5.
    """

    def __init__(
        self,
        models: dict[str, ModelInfo] | None = None,
        caller: Callable | None = None,
        spend_log_path: str = "./spend.jsonl",
    ):
        self.models = models or DEFAULT_MODELS
        self.caller = caller or _stub_caller
        self.spend_log_path = Path(spend_log_path)
        self.spend: dict[tuple[str, str], float] = defaultdict(float)  # (tenant, model) → USD
        self.requests: list[RouteResponse] = []

    def route(self, req: RouteRequest) -> RouteResponse:
        complexity = estimate_complexity(req.text)
        tier = req.force_tier or tier_for_complexity(complexity["score"])

        model = self._pick(tier, req.text)
        if model is None:
            raise ValueError(f"No model available for tier={tier!r}")

        in_tok = complexity["token_count"]
        t0 = time.time()
        text, out_tok = self.caller(model, req.text)
        latency_ms = (time.time() - t0) * 1000

        cost = compute_cost(model, in_tok, out_tok)
        if req.budget_cap_usd is not None and cost > req.budget_cap_usd:
            raise BudgetExceeded(f"Estimated cost ${cost:.4f} exceeds cap ${req.budget_cap_usd:.4f}")

        self.spend[(req.tenant, model.name)] += cost
        self._append_log(req, model, in_tok, out_tok, cost, complexity)

        resp = RouteResponse(
            request_id=req.request_id,
            model=model.name,
            tier=tier,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            complexity=complexity,
            latency_ms=latency_ms,
            text=text,
        )
        self.requests.append(resp)
        return resp

    def _pick(self, tier: str, text: str) -> ModelInfo | None:
        # Pick cheapest in tier that can fit input length
        tok = len(text.split())
        candidates = [m for m in self.models.values() if m.tier == tier and m.max_input_tokens >= tok]
        if not candidates:
            return None
        # Tie-break by total cost for a 1k-in/1k-out request
        candidates.sort(key=lambda m: m.input_cost_per_mtok + m.output_cost_per_mtok)
        return candidates[0]

    def _append_log(self, req, model, in_tok, out_tok, cost, complexity):
        record = {
            "ts": int(time.time()),
            "request_id": req.request_id,
            "tenant": req.tenant,
            "model": model.name,
            "tier": model.tier,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": cost,
            "complexity_score": complexity["score"],
        }
        with self.spend_log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    # --- observability helpers (project #5 will plug into Prometheus)
    def metrics(self) -> dict:
        by_tier = defaultdict(lambda: {"count": 0, "cost": 0.0})
        for r in self.requests:
            by_tier[r.tier]["count"] += 1
            by_tier[r.tier]["cost"] += r.cost_usd
        return {
            "total_cost_usd": sum(self.spend.values()),
            "by_tenant": {
                t: {"cost_usd": sum(c for (te, _m), c in self.spend.items() if te == t)}
                for t in {te for (te, _m) in self.spend}
            },
            "by_tier": dict(by_tier),
            "n_requests": len(self.requests),
        }


class BudgetExceeded(Exception):
    pass


def _stub_caller(model: ModelInfo, text: str) -> tuple[str, int]:
    """Deterministic fake model — echoes a short answer.

    Output token estimate: min(50, ceil(len(text) / 100)).
    """
    out_len = max(1, min(50, (len(text) // 100) + 1))
    return f"[{model.name}] echo: {text[:40]}", out_len