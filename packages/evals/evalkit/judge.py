"""LLM-as-judge scorers on the Anthropic API.

One judge call per case returns three verdicts in a single JSON object:

  faithfulness      every claim in the answer is supported by the retrieved context
  answer_relevancy  the answer actually addresses the question
  correctness       the answer agrees with expected_answer (only when one is given)

Design notes
  - Structured output (`output_config.format`) so parsing never depends on
    the model remembering to emit clean JSON.
  - The rubric lives in the system prompt with a cache breakpoint; across a
    golden set of a few hundred cases that is most of the input tokens.
  - Verdicts are cached per case_id so the three scorers share one request.
  - The judge is a second, cheaper model by default (claude-sonnet-5). The
    system under test can be anything; the judge only sees text.
  - Any API failure or refusal becomes a scored 0.0 with a reason attached,
    so a flaky judge fails loudly in the report instead of crashing the run.

Usage
    from evalkit.harness import EvalHarness
    from evalkit.judge import JudgeScorers

    h = EvalHarness()
    h.scorers.update(JudgeScorers().as_scorers())      # adds judge_* metrics
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from evalkit.harness import CaseResult, GoldenCase, Scorer

DEFAULT_MODEL = os.environ.get("EVALKIT_JUDGE_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = """You are grading the output of a retrieval-augmented question answering system.

You will receive the question, the context passages the system retrieved, the system's answer, and sometimes a reference answer. Score each dimension from 0.0 to 1.0:

faithfulness: 1.0 if every factual claim in the answer is supported by the retrieved context. Deduct in proportion to the share of claims that are unsupported or contradicted. An answer that says it cannot find the information is faithful (1.0) if the context truly lacks it.

answer_relevancy: 1.0 if the answer directly addresses what was asked, with no padding or off-topic material. Deduct for partial answers or answers to a different question.

correctness: only when a reference answer is given. 1.0 if the answer conveys the same facts as the reference (wording may differ), 0.0 if it contradicts it, in between for partial. If no reference answer is given, return null.

Be strict and literal. Do not reward confident tone. Keep the reason to one sentence."""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "number"},
        "answer_relevancy": {"type": "number"},
        "correctness": {"type": ["number", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["faithfulness", "answer_relevancy", "correctness", "reason"],
    "additionalProperties": False,
}


@dataclass
class Verdict:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    correctness: float | None = None
    reason: str = ""
    error: str | None = None
    usage: dict = field(default_factory=dict)


def _clamp(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


class JudgeScorers:
    """Bundle of scorers backed by one judge request per case."""

    def __init__(self, client: Any | None = None, model: str = DEFAULT_MODEL, max_context_chars: int = 12_000):
        if client is None:
            import anthropic  # local import so the package works without the SDK installed

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_context_chars = max_context_chars
        self._cache: dict[str, Verdict] = {}
        self.calls = 0

    # --- scorers (all share one cached verdict per case)
    def faithfulness(self, case: GoldenCase, result: CaseResult) -> float:
        return _clamp(self.verdict(case, result).faithfulness)

    def answer_relevancy(self, case: GoldenCase, result: CaseResult) -> float:
        return _clamp(self.verdict(case, result).answer_relevancy)

    def correctness(self, case: GoldenCase, result: CaseResult) -> float:
        v = self.verdict(case, result)
        if v.correctness is None:
            return 1.0 if v.error is None else 0.0  # nothing to compare against
        return _clamp(v.correctness)

    def as_scorers(self, prefix: str = "judge_") -> dict[str, Scorer]:
        return {
            f"{prefix}faithfulness": self.faithfulness,
            f"{prefix}answer_relevancy": self.answer_relevancy,
            f"{prefix}correctness": self.correctness,
        }

    # --- the one request
    def verdict(self, case: GoldenCase, result: CaseResult) -> Verdict:
        key = case.case_id
        if key in self._cache:
            return self._cache[key]
        v = self._judge(case, result)
        self._cache[key] = v
        return v

    def _judge(self, case: GoldenCase, result: CaseResult) -> Verdict:
        if result.error:
            return Verdict(error=f"system error, not judged: {result.error}")
        prompt = self._build_prompt(case, result)
        self.calls += 1
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": prompt}],
                output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
            )
        except Exception as e:  # noqa: BLE001 - surfaced in the report, never raised
            return Verdict(error=f"{type(e).__name__}: {e}")
        if getattr(response, "stop_reason", None) == "refusal":
            return Verdict(error="judge refused")
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return Verdict(error=f"judge returned non-JSON: {e}")
        usage = getattr(response, "usage", None)
        return Verdict(
            faithfulness=_clamp(data.get("faithfulness")),
            answer_relevancy=_clamp(data.get("answer_relevancy")),
            correctness=None if data.get("correctness") is None else _clamp(data.get("correctness")),
            reason=str(data.get("reason", "")),
            usage={
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
            } if usage else {},
        )

    def _build_prompt(self, case: GoldenCase, result: CaseResult) -> str:
        ctx_parts, budget = [], self.max_context_chars
        for i, c in enumerate(result.retrieved_chunks, 1):
            piece = f"[{i}] ({c.get('doc_id')}, p.{c.get('page')}) {c.get('text', '')}"
            if len(piece) > budget:
                piece = piece[:budget] + " ...[truncated]"
            ctx_parts.append(piece)
            budget -= len(piece)
            if budget <= 0:
                break
        parts = [
            f"<question>\n{case.question}\n</question>",
            "<context>\n" + ("\n\n".join(ctx_parts) or "(no context retrieved)") + "\n</context>",
            f"<answer>\n{result.answer or '(empty answer)'}\n</answer>",
        ]
        if case.expected_answer:
            parts.append(f"<reference_answer>\n{case.expected_answer}\n</reference_answer>")
        return "\n\n".join(parts)
