# modelrouter

Send each request to the cheapest model that can handle it, and keep a spend
log that matches the provider's bill.

```python
from modelrouter.router import ModelRouter, RouteRequest
from modelrouter.callers import litellm_caller

router = ModelRouter(caller=litellm_caller(system="Answer briefly."))
resp = router.route(RouteRequest("req-1", "What is 2+2?", tenant="acme"))
print(resp.model, resp.tier, f"${resp.cost_usd:.6f}", resp.cost_source)
```

```
python -m modelrouter route "Explain step by step why ..." --litellm
python -m modelrouter report spend.jsonl
```

## How routing works

1. `estimate_complexity(text)` scores the request with regex signals: code,
   math, multi-step phrasing, length. Deterministic and sub-millisecond.
2. `tier_for_complexity(score)` maps the score to `cheap` / `medium` /
   `expensive`. `force_tier` overrides it.
3. The cheapest model in that tier whose context window fits the input wins.
4. If `budget_cap_usd` is set, the estimated cost is checked before the call.
5. The caller runs. With `litellm_caller` the response's own token counts and
   LiteLLM's `completion_cost` are used; with the stub, registry prices are.
6. One JSON line goes to `spend.jsonl` with tenant, model, tier, tokens, cost
   and `cost_source` (`provider` or `registry`).

## Default registry

| Model | Tier |
|---|---|
| gpt-5-mini, claude-haiku-4-5 | cheap |
| gpt-5, claude-sonnet-5 | medium |
| claude-opus-5 | expensive |

Prices are read from `litellm.model_cost` at import time when litellm is
installed, so they track LiteLLM's price table. Without litellm a snapshot
dated in `pricing.py` is used. Pass your own `models=` dict to change any of it.

## Spend report

`spend_report(path)` aggregates by tenant, model and tier and computes what
the same tokens would have cost on the most expensive model alone, which is
the number that justifies the router's existence.

## Layout

```
modelrouter/
  router.py     complexity heuristics, ModelRouter, spend_report
  pricing.py    ModelInfo, registry, litellm price lookup with snapshot fallback
  callers.py    CallResult, stub_caller, litellm_caller
  metrics.py    optional Prometheus counters/histogram
tests/          27 tests, no network (LiteLLM is faked)
```
