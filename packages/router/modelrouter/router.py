"""Cost-aware model router.

Routes each request to the cheapest model in the tier its complexity calls
for, calls it, and records what it cost. Pieces:

  estimate_complexity   regex heuristics (code, math, multi-step, length)
  tier_for_complexity   score -> "cheap" | "medium" | "expensive"
  ModelRouter           pick model, call, enforce budget cap, log spend
  spend_report          aggregate a spend.jsonl by tenant / model / tier

Prices and the default registry live in `modelrouter.pricing`; callers
(stub and LiteLLM) in `modelrouter.callers`. The router trusts the caller's
reported token counts and cost when it provides them, and falls back to
registry prices otherwise, so the log matches what the provider bills.
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from modelrouter.callers import Caller, CallResult, stub_caller
from modelrouter.pricing import ModelInfo, build_registry, compute_cost

DEFAULT_MODELS: dict[str, ModelInfo] = build_registry()

# ------------------------------------------------------------ complexity
_CODE = [re.compile(p, f) for p, f in [
    (r"```", 0), (r"\bdef \w+\(", 0), (r"\bfunction \w+\(", 0), (r"\bclass \w+[\(:]", 0),
    (r"=>|->|<\?php", 0), (r"\bSELECT\b.*\bFROM\b", re.I), (r"\bimport \w+\b", 0),
]]
_MATH = [re.compile(p, f) for p, f in [
    (r"\b\d+\s*[+\-*/]\s*\d+", 0), (r"[∫∑∏√∞≠≈≤≥]", 0), (r"\bequation\b|\bintegral\b|\bderivative\b", re.I), (r"\$[^$]+\$", 0),
]]
_MULTI = [re.compile(p, re.I) for p in [r"\bstep[- ]by[- ]step\b", r"\bfirst,?\s+then\b", r"\b\d+\.\s", r"\bexplain\b.*\bwhy\b"]]


def estimate_complexity(text: str) -> dict:
    """{score, signals, token_count}. Deterministic and cheap; runs before every call."""
    signals = {
        "has_code": any(p.search(text) for p in _CODE),
        "has_math": any(p.search(text) for p in _MATH),
        "is_multi_step": any(p.search(text) for p in _MULTI),
        "is_long": len(text) > 2000,
    }
    score = 1.0 * signals["has_code"] + 1.0 * signals["has_math"] + 1.5 * signals["is_multi_step"] + 0.5 * signals["is_long"]
    token_count = len(text.split())  # whitespace proxy; the caller's real count replaces it after the call
    if token_count > 500:
        score += 1.0
    return {"score": score, "signals": signals, "token_count": token_count}


def tier_for_complexity(score: float) -> str:
    return "cheap" if score < 1.0 else "medium" if score < 3.0 else "expensive"


# ------------------------------------------------------------ request / response
@dataclass
class RouteRequest:
    request_id: str
    text: str
    tenant: str = "default"
    force_tier: str | None = None
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
    cost_source: str = "registry"  # "provider" when the caller reported it


class BudgetExceeded(Exception):
    pass


# ------------------------------------------------------------ router
class ModelRouter:
    def __init__(self, models: dict[str, ModelInfo] | None = None, caller: Caller | None = None, spend_log_path: str = "./spend.jsonl"):
        self.models = models or DEFAULT_MODELS
        self.caller = caller or stub_caller
        self.spend_log_path = Path(spend_log_path)
        self.spend: dict[tuple[str, str], float] = defaultdict(float)
        self.requests: list[RouteResponse] = []

    def route(self, req: RouteRequest) -> RouteResponse:
        complexity = estimate_complexity(req.text)
        tier = req.force_tier or tier_for_complexity(complexity["score"])
        model = self._pick(tier, complexity["token_count"])
        if model is None:
            raise ValueError(f"No model available for tier={tier!r}")

        # Budget check before the call, on the estimate, so we never spend past the cap.
        if req.budget_cap_usd is not None:
            est = compute_cost(model, complexity["token_count"], 50)
            if est > req.budget_cap_usd:
                raise BudgetExceeded(f"Estimated cost ${est:.4f} for {model.name} exceeds cap ${req.budget_cap_usd:.4f}")

        t0 = time.perf_counter()
        result = self.caller(model, req.text)
        latency_ms = (time.perf_counter() - t0) * 1000
        if not isinstance(result, CallResult):  # tolerate legacy (text, out_tokens) callers
            text, out_tok = result
            result = CallResult(text, complexity["token_count"], out_tok)

        in_tok = result.input_tokens or complexity["token_count"]
        cost = result.cost_usd if result.cost_usd is not None else compute_cost(model, in_tok, result.output_tokens)
        source = "provider" if result.cost_usd is not None else "registry"

        self.spend[(req.tenant, model.name)] += cost
        self._append_log(req, model, in_tok, result.output_tokens, cost, source, complexity, latency_ms)
        resp = RouteResponse(req.request_id, model.name, tier, in_tok, result.output_tokens, cost, complexity, latency_ms, result.text, source)
        self.requests.append(resp)
        return resp

    def _pick(self, tier: str, input_tokens: int) -> ModelInfo | None:
        fits = [m for m in self.models.values() if m.tier == tier and m.max_input_tokens >= input_tokens]
        return min(fits, key=lambda m: m.input_cost_per_mtok + m.output_cost_per_mtok, default=None)

    def _append_log(self, req, model, in_tok, out_tok, cost, source, complexity, latency_ms) -> None:
        record = {
            "ts": int(time.time()), "request_id": req.request_id, "tenant": req.tenant,
            "model": model.name, "tier": model.tier, "input_tokens": in_tok, "output_tokens": out_tok,
            "cost_usd": cost, "cost_source": source, "complexity_score": complexity["score"], "latency_ms": round(latency_ms, 1),
        }
        with self.spend_log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def metrics(self) -> dict:
        by_tier: dict[str, dict] = defaultdict(lambda: {"count": 0, "cost": 0.0})
        for r in self.requests:
            by_tier[r.tier]["count"] += 1
            by_tier[r.tier]["cost"] += r.cost_usd
        tenants = {t for t, _ in self.spend}
        return {
            "total_cost_usd": sum(self.spend.values()),
            "by_tenant": {t: {"cost_usd": sum(c for (te, _), c in self.spend.items() if te == t)} for t in tenants},
            "by_tier": dict(by_tier),
            "n_requests": len(self.requests),
        }


# ------------------------------------------------------------ spend log analysis
def spend_report(log_path: str | Path) -> dict:
    """Aggregate a spend.jsonl: totals by tenant, model and tier, plus what the
    same traffic would have cost on the most expensive model alone."""
    p = Path(log_path)
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    by = {k: defaultdict(lambda: {"requests": 0, "cost_usd": 0.0}) for k in ("tenant", "model", "tier")}
    total = 0.0
    for r in rows:
        total += r["cost_usd"]
        for k in by:
            by[k][r[k]]["requests"] += 1
            by[k][r[k]]["cost_usd"] += r["cost_usd"]
    priciest = max(DEFAULT_MODELS.values(), key=lambda m: m.input_cost_per_mtok + m.output_cost_per_mtok)
    counterfactual = sum(compute_cost(priciest, r["input_tokens"], r["output_tokens"]) for r in rows)
    return {
        "n_requests": len(rows),
        "total_cost_usd": total,
        "by_tenant": {k: dict(v) for k, v in by["tenant"].items()},
        "by_model": {k: dict(v) for k, v in by["model"].items()},
        "by_tier": {k: dict(v) for k, v in by["tier"].items()},
        "all_on_most_expensive_usd": counterfactual,
        "most_expensive_model": priciest.name,
        "savings_pct": (1 - total / counterfactual) * 100 if counterfactual else 0.0,
    }
