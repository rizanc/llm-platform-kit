"""Judge scorers with a fake Anthropic client. No network."""
import json
from types import SimpleNamespace

import pytest

from evalkit.harness import CaseResult, EvalHarness, GoldenCase, HarnessConfig
from evalkit.judge import SYSTEM_PROMPT, JudgeScorers, Verdict


class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kw):
        self.requests.append(kw)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            stop_reason=reply.get("stop_reason", "end_turn"),
            content=[SimpleNamespace(type="text", text=reply.get("text", ""))],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=80),
        )


def fake_client(*replies):
    return SimpleNamespace(messages=FakeMessages(replies))


def verdict_text(f=0.9, r=0.8, c=None, reason="ok"):
    return {"text": json.dumps({"faithfulness": f, "answer_relevancy": r, "correctness": c, "reason": reason})}


CASE = GoldenCase("c1", "What is the capital of France?", expected_answer="Paris")
RESULT = CaseResult("c1", CASE.question, "Paris is the capital of France.", [], [{"doc_id": "geo", "page": 1, "text": "Paris is the capital of France."}])


def test_one_request_serves_all_three_scorers():
    j = JudgeScorers(client=fake_client(verdict_text(0.9, 0.8, 1.0)))
    scorers = j.as_scorers()
    assert scorers["judge_faithfulness"](CASE, RESULT) == 0.9
    assert scorers["judge_answer_relevancy"](CASE, RESULT) == 0.8
    assert scorers["judge_correctness"](CASE, RESULT) == 1.0
    assert j.calls == 1


def test_request_shape_uses_structured_output_and_cached_rubric():
    j = JudgeScorers(client=fake_client(verdict_text()), model="claude-sonnet-5")
    j.faithfulness(CASE, RESULT)
    req = j.client.messages.requests[0]
    assert req["model"] == "claude-sonnet-5"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert req["system"][0]["text"] == SYSTEM_PROMPT
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    user = req["messages"][0]["content"]
    assert "<question>" in user and "<context>" in user and "<reference_answer>" in user
    assert "Paris" in user


def test_scores_are_clamped():
    j = JudgeScorers(client=fake_client(verdict_text(1.7, -3, 0.5)))
    assert j.faithfulness(CASE, RESULT) == 1.0
    assert j.answer_relevancy(CASE, RESULT) == 0.0


def test_missing_reference_makes_correctness_neutral():
    j = JudgeScorers(client=fake_client(verdict_text(1, 1, None)))
    assert j.correctness(CASE, RESULT) == 1.0


def test_api_error_scores_zero_and_keeps_reason():
    j = JudgeScorers(client=fake_client(RuntimeError("boom")))
    assert j.faithfulness(CASE, RESULT) == 0.0
    assert "RuntimeError" in j.verdict(CASE, RESULT).error


def test_refusal_scores_zero():
    j = JudgeScorers(client=fake_client({"stop_reason": "refusal", "text": ""}))
    assert j.answer_relevancy(CASE, RESULT) == 0.0
    assert j.verdict(CASE, RESULT).error == "judge refused"


def test_non_json_scores_zero():
    j = JudgeScorers(client=fake_client({"text": "not json"}))
    assert j.faithfulness(CASE, RESULT) == 0.0


def test_system_errors_are_not_sent_to_the_judge():
    j = JudgeScorers(client=fake_client(verdict_text()))
    broken = CaseResult("c1", "q", "", [], [], error="ValueError: pipeline exploded")
    assert j.faithfulness(CASE, broken) == 0.0
    assert j.calls == 0


def test_context_is_truncated_to_budget():
    j = JudgeScorers(client=fake_client(verdict_text()), max_context_chars=200)
    big = CaseResult("c1", "q", "a", [], [{"doc_id": "d", "page": 1, "text": "x" * 5000}])
    j.faithfulness(CASE, big)
    user = j.client.messages.requests[0]["messages"][0]["content"]
    assert "[truncated]" in user and len(user) < 1000


def test_judge_plugs_into_harness(tmp_path):
    replies = [verdict_text(0.9, 0.9, 1.0), verdict_text(0.2, 0.9, 0.0)]
    j = JudgeScorers(client=fake_client(*replies))
    h = EvalHarness(HarnessConfig(history_path=str(tmp_path / "h.jsonl")))
    h.scorers.update(j.as_scorers())
    h.config.thresholds["judge_faithfulness"] = 0.7
    cases = [GoldenCase("a", "q1", expected_answer="x"), GoldenCase("b", "q2", expected_answer="y")]
    results, summary = h.run(cases, lambda c: CaseResult(c.case_id, c.question, "ans " + c.question, [], [{"doc_id": "d", "page": 1, "text": "ans " + c.question}]))
    assert "judge_faithfulness" in summary["averages"]
    assert results[0].metrics["judge_faithfulness"] == 0.9
    assert results[1].passed is False
    assert j.calls == 2
