"""Model callers. A caller turns (model, text) into a CallResult.

  stub_caller      deterministic, no network; unit tests and dry runs
  litellm_caller   real completions through LiteLLM; returns the provider's
                   own token counts and LiteLLM's computed cost when available

The router prefers the caller's reported cost and token counts over its own
estimates, so the spend log reflects what the provider will actually bill.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from modelrouter.pricing import ModelInfo


@dataclass
class CallResult:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = None     # None -> router computes from registry prices
    provider_model: str | None = None


Caller = Callable[[ModelInfo, str], CallResult]


def stub_caller(model: ModelInfo, text: str) -> CallResult:
    """Echo a prefix of the input. Output tokens: min(50, ceil(len/100))."""
    out_len = max(1, min(50, len(text) // 100 + 1))
    return CallResult(f"[{model.name}] echo: {text[:40]}", input_tokens=len(text.split()), output_tokens=out_len)


def litellm_caller(
    system: str | None = None,
    max_tokens: int = 1024,
    model_map: dict[str, str] | None = None,
    completion: Callable | None = None,
    completion_cost: Callable | None = None,
) -> Caller:
    """Build a caller backed by litellm.completion.

    model_map lets registry names differ from LiteLLM model strings, e.g.
    {"claude-sonnet-5": "anthropic/claude-sonnet-5"}. `completion` and
    `completion_cost` are injectable for tests.
    """
    if completion is None or completion_cost is None:
        import litellm

        completion = completion or litellm.completion
        completion_cost = completion_cost or litellm.completion_cost
    model_map = model_map or {}

    def call(model: ModelInfo, text: str) -> CallResult:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        resp = completion(model=model_map.get(model.name, model.name), messages=messages, max_tokens=max_tokens)
        usage = getattr(resp, "usage", None) or {}
        get = (lambda k, d=0: getattr(usage, k, None) if not isinstance(usage, dict) else usage.get(k)) 
        in_tok = int(get("prompt_tokens") or 0)
        out_tok = int(get("completion_tokens") or 0)
        try:
            cost = float(completion_cost(completion_response=resp))
        except Exception:  # noqa: BLE001 - unknown model in litellm's table; router falls back to registry prices
            cost = None
        choice = resp.choices[0]
        content = choice.message.content if hasattr(choice, "message") else choice["message"]["content"]
        return CallResult(content or "", in_tok, out_tok, cost, getattr(resp, "model", None))

    return call
