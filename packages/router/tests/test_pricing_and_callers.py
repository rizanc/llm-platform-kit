"""Registry prices, LiteLLM caller (with fakes), spend report. No network."""
import json
from types import SimpleNamespace

from modelrouter.callers import CallResult, litellm_caller, stub_caller
from modelrouter.pricing import SNAPSHOT_DATE, ModelInfo, build_registry, compute_cost
from modelrouter.router import DEFAULT_MODELS, ModelRouter, RouteRequest, spend_report


def test_snapshot_registry_has_all_tiers_and_positive_prices():
    reg = build_registry(prefer_litellm=False)
    assert {m.tier for m in reg.values()} == {"cheap", "medium", "expensive"}
    assert all(m.input_cost_per_mtok > 0 and m.output_cost_per_mtok > 0 for m in reg.values())
    assert SNAPSHOT_DATE.startswith("2026")


def test_live_registry_overrides_snapshot_when_litellm_knows_the_model():
    live = build_registry(prefer_litellm=True)
    snap = build_registry(prefer_litellm=False)
    assert live.keys() == snap.keys()
    for name in live:  # prices may differ, tiers never do
        assert live[name].tier == snap[name].tier


def fake_completion_factory(text="hi", prompt_tokens=12, completion_tokens=3, model="anthropic/claude-sonnet-5"):
    calls = []

    def completion(**kw):
        calls.append(kw)
        return SimpleNamespace(
            model=model,
            usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        )

    completion.calls = calls
    return completion


def test_litellm_caller_returns_provider_tokens_and_cost():
    completion = fake_completion_factory()
    call = litellm_caller(system="be brief", completion=completion, completion_cost=lambda completion_response: 0.00042,
                          model_map={"claude-sonnet-5": "anthropic/claude-sonnet-5"})
    model = ModelInfo("claude-sonnet-5", "medium", 2.0, 10.0, 1_000_000)
    res = call(model, "hello")
    assert isinstance(res, CallResult)
    assert (res.input_tokens, res.output_tokens, res.cost_usd) == (12, 3, 0.00042)
    req = completion.calls[0]
    assert req["model"] == "anthropic/claude-sonnet-5"
    assert req["messages"][0] == {"role": "system", "content": "be brief"}
    assert req["messages"][-1] == {"role": "user", "content": "hello"}


def test_litellm_caller_cost_failure_falls_back_to_registry(tmp_path):
    def broken_cost(completion_response):
        raise ValueError("model not in litellm price map")

    call = litellm_caller(completion=fake_completion_factory(prompt_tokens=1000, completion_tokens=100), completion_cost=broken_cost)
    r = ModelRouter(caller=call, spend_log_path=str(tmp_path / "s.jsonl"))
    resp = r.route(RouteRequest("r1", "What is 2+2?", tenant="t"))
    m = DEFAULT_MODELS[resp.model]
    assert resp.cost_source == "registry"
    assert resp.cost_usd == compute_cost(m, 1000, 100)


def test_router_prefers_provider_cost(tmp_path):
    call = litellm_caller(completion=fake_completion_factory(), completion_cost=lambda completion_response: 0.001)
    r = ModelRouter(caller=call, spend_log_path=str(tmp_path / "s.jsonl"))
    resp = r.route(RouteRequest("r1", "hello", tenant="t"))
    assert resp.cost_source == "provider" and resp.cost_usd == 0.001
    row = json.loads((tmp_path / "s.jsonl").read_text().splitlines()[0])
    assert row["cost_source"] == "provider" and row["input_tokens"] == 12


def test_legacy_tuple_caller_still_works(tmp_path):
    r = ModelRouter(caller=lambda m, t: ("ok", 5), spend_log_path=str(tmp_path / "s.jsonl"))
    resp = r.route(RouteRequest("r1", "hello there"))
    assert resp.text == "ok" and resp.output_tokens == 5


def test_spend_report_aggregates_and_counterfactual(tmp_path):
    log = tmp_path / "s.jsonl"
    r = ModelRouter(caller=stub_caller, spend_log_path=str(log))
    r.route(RouteRequest("a", "What is the capital of France?", tenant="acme"))
    r.route(RouteRequest("b", "```python\ndef f(): pass\n```", tenant="acme"))
    r.route(RouteRequest("c", "Explain step by step why the integral of 2*x is x squared", tenant="globex"))
    rep = spend_report(log)
    assert rep["n_requests"] == 3
    assert set(rep["by_tenant"]) == {"acme", "globex"}
    assert rep["all_on_most_expensive_usd"] >= rep["total_cost_usd"] > 0
    assert 0 <= rep["savings_pct"] < 100
