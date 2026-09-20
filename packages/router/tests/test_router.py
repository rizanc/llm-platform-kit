"""Unit tests — model router with deterministic stub caller."""
import json
import os
import tempfile
from pathlib import Path

import pytest

from modelrouter.router import (
    BudgetExceeded,
    DEFAULT_MODELS,
    ModelInfo,
    ModelRouter,
    RouteRequest,
    compute_cost,
    estimate_complexity,
    tier_for_complexity,
)


# ---------------------- complexity classifier
def test_complexity_simple_text_is_cheap():
    c = estimate_complexity("What is the weather today?")
    assert c["score"] < 1.0
    assert c["signals"]["has_code"] is False
    assert c["signals"]["has_math"] is False


def test_complexity_code_question_is_medium():
    c = estimate_complexity("Write a Python function `def add(a, b):` that returns the sum.")
    assert c["signals"]["has_code"] is True
    assert c["score"] >= 1.0


def test_complexity_math_question():
    c = estimate_complexity("Compute the integral ∫ x² dx from 0 to 1.")
    assert c["signals"]["has_math"] is True


def test_complexity_multi_step_question():
    c = estimate_complexity("Explain step-by-step why the sky is blue.")
    assert c["signals"]["is_multi_step"] is True
    assert c["score"] >= 1.5


def test_complexity_long_input_bumps_score():
    short = estimate_complexity("hi")
    long_text = "word " * 1000
    long_c = estimate_complexity(long_text)
    assert long_c["signals"]["is_long"] is True
    assert long_c["score"] > short["score"]


def test_tier_thresholds():
    assert tier_for_complexity(0.0) == "cheap"
    assert tier_for_complexity(0.9) == "cheap"
    assert tier_for_complexity(1.0) == "medium"
    assert tier_for_complexity(2.9) == "medium"
    assert tier_for_complexity(3.0) == "expensive"
    assert tier_for_complexity(10.0) == "expensive"


# ---------------------- cost
def test_compute_cost_simple():
    m = ModelInfo("test", "cheap", 1.0, 2.0, 1000)
    # 1000 input + 1000 output = 1 * 0.001 + 2 * 0.001 = 0.003
    assert compute_cost(m, 1000, 1000) == pytest.approx(0.003)


# ---------------------- routing
def test_simple_query_routes_to_cheap():
    r = ModelRouter()
    req = RouteRequest(request_id="r1", text="What is the weather today?")
    resp = r.route(req)
    assert resp.tier == "cheap"
    assert resp.model == "gpt-4o-mini"  # cheapest in cheap tier
    assert resp.cost_usd >= 0


def test_code_query_routes_to_medium():
    r = ModelRouter()
    text = "Write a Python function `def add(a, b):` that returns the sum."
    req = RouteRequest(request_id="r1", text=text)
    resp = r.route(req)
    assert resp.tier in ("medium", "expensive")  # code bumps score


def test_complex_query_routes_to_expensive():
    """Stack every signal + long input → expensive tier."""
    r = ModelRouter()
    # long text with code, math, and multi-step phrasing
    text = (
        "step-by-step derivation\n\n"
        + ("```python\ndef foo(x): return x**2\n```\n" * 5)
        + ("Compute ∫ x² dx. Solve the equation $E=mc^2$.\n" * 5)
        + ("word " * 600)  # pad to > 500 tokens → bump score
    )
    req = RouteRequest(request_id="r1", text=text)
    resp = r.route(req)
    assert resp.tier == "expensive"


def test_force_tier_overrides_complexity():
    r = ModelRouter()
    req = RouteRequest(request_id="r1", text="hello", force_tier="expensive")
    resp = r.route(req)
    assert resp.tier == "expensive"
    assert resp.model in ("o1", "claude-opus")


def test_budget_cap_rejects_expensive_call():
    r = ModelRouter()
    text = "step-by-step derivation of ∫ x² dx plus equation $E=mc^2$"
    req = RouteRequest(request_id="r1", text=text, budget_cap_usd=0.0000001)
    with pytest.raises(BudgetExceeded):
        r.route(req)


def test_router_picks_cheapest_in_tier():
    r = ModelRouter()
    # All "cheap" tier — should pick gpt-4o-mini ($0.15/$0.60) over claude-haiku ($0.25/$1.25)
    req = RouteRequest(request_id="r1", text="hi", force_tier="cheap")
    resp = r.route(req)
    assert resp.model == "gpt-4o-mini"


def test_router_handles_no_model_for_tier():
    """If a tier has no model that fits the input length, raise."""
    # Only Opus in expensive, max 200k tokens; make a request too long
    tiny_registry = {"only-opus": ModelInfo("only-opus", "expensive", 1.0, 1.0, 10)}
    r = ModelRouter(models=tiny_registry)
    req = RouteRequest(request_id="r1", text="word " * 100, force_tier="expensive")
    with pytest.raises(ValueError, match="No model"):
        r.route(req)


# ---------------------- spend tracking
def test_spend_accumulates_per_tenant_model():
    r = ModelRouter()
    for i in range(3):
        r.route(RouteRequest(request_id=f"r{i}", text="hi", tenant="acme"))
    assert r.spend[("acme", "gpt-4o-mini")] > 0
    assert sum(r.spend.values()) > 0


def test_spend_log_appended():
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd()
        try:
            os.chdir(d)
            r = ModelRouter(spend_log_path="./spend.jsonl")
            r.route(RouteRequest(request_id="r1", text="hi"))
            r.route(RouteRequest(request_id="r2", text="hello world"))
            lines = Path("spend.jsonl").read_text().splitlines()
            assert len(lines) == 2
            entries = [json.loads(l) for l in lines]
            assert {e["request_id"] for e in entries} == {"r1", "r2"}
            assert all("cost_usd" in e for e in entries)
        finally:
            os.chdir(cwd)


def test_metrics_summary():
    r = ModelRouter()
    r.route(RouteRequest(request_id="r1", text="hi", tenant="acme"))
    r.route(RouteRequest(request_id="r2", text="hi", tenant="acme"))
    r.route(RouteRequest(request_id="r3", text="hi", tenant="globex"))
    m = r.metrics()
    assert m["n_requests"] == 3
    assert m["total_cost_usd"] > 0
    assert "acme" in m["by_tenant"]
    assert "globex" in m["by_tenant"]
    assert m["by_tenant"]["acme"]["cost_usd"] > 0


# ---------------------- registry sanity
def test_default_models_have_all_three_tiers():
    tiers = {m.tier for m in DEFAULT_MODELS.values()}
    assert tiers == {"cheap", "medium", "expensive"}


def test_default_models_have_positive_prices():
    for m in DEFAULT_MODELS.values():
        assert m.input_cost_per_mtok > 0
        assert m.output_cost_per_mtok > 0


def test_router_preserves_request_id():
    r = ModelRouter()
    resp = r.route(RouteRequest(request_id="my-id", text="hi"))
    assert resp.request_id == "my-id"