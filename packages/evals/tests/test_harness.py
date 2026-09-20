"""Unit tests — eval harness with deterministic scorers. No model downloads."""
import json
import os
import tempfile
from pathlib import Path

import pytest

from evalkit.harness import (
    CaseResult,
    EvalHarness,
    GoldenCase,
    HarnessConfig,
    METRICS,
    score_answer_relevancy,
    score_citation_accuracy,
    score_context_precision,
    score_context_recall,
    score_faithfulness,
    score_grounding_rate,
    report_console,
    report_json,
    report_markdown,
)
from evalkit.system_stub import stub_rag_system


def make_result(case_id, question, answer, citations=None, chunks=None):
    return CaseResult(
        case_id=case_id,
        question=question,
        answer=answer,
        citations=citations or [],
        retrieved_chunks=chunks or [],
        metrics={},
        passed=False,
    )


# ---------------------- individual scorers
def test_faithfulness_full_overlap():
    case = GoldenCase("c1", "What is X?")
    result = make_result("c1", "What is X?", "alpha beta gamma",
                         chunks=[{"text": "alpha beta gamma delta"}])
    assert score_faithfulness(case, result) == pytest.approx(1.0)


def test_faithfulness_no_context():
    case = GoldenCase("c1", "What is X?")
    result = make_result("c1", "What is X?", "alpha beta")
    assert score_faithfulness(case, result) == 0.0


def test_faithfulness_partial():
    case = GoldenCase("c1", "What is X?")
    result = make_result("c1", "What is X?", "alpha beta",
                         chunks=[{"text": "alpha gamma delta"}])
    assert score_faithfulness(case, result) == pytest.approx(0.5)


def test_answer_relevancy_overlap():
    case = GoldenCase("c1", "What is the capital of France?")
    result = make_result("c1", case.question, "The capital is Paris.")
    score = score_answer_relevancy(case, result)
    assert 0.3 < score < 1.0


def test_context_precision_with_expected_pages():
    case = GoldenCase("c1", "Q", expected_pages=[5, 7])
    result = make_result(
        "c1", "Q", "A",
        chunks=[
            {"doc_id": "d", "page": 5, "text": "x"},
            {"doc_id": "d", "page": 99, "text": "y"},
            {"doc_id": "d", "page": 7, "text": "z"},
        ],
    )
    assert score_context_precision(case, result) == pytest.approx(2 / 3)


def test_context_recall_all_expected_found():
    case = GoldenCase("c1", "Q", expected_doc_ids=["dA"], expected_pages=[3, 4])
    result = make_result(
        "c1", "Q", "A",
        chunks=[
            {"doc_id": "dA", "page": 3, "text": ""},
            {"doc_id": "dA", "page": 4, "text": ""},
            {"doc_id": "dA", "page": 99, "text": ""},
        ],
    )
    assert score_context_recall(case, result) == pytest.approx(1.0)


def test_context_recall_keyword_fallback():
    case = GoldenCase("c1", "Q", expected_keywords=["Paris", "France", "Seine"])
    result = make_result(
        "c1", "Q", "A",
        chunks=[{"doc_id": "d", "page": 1, "text": "Paris is the capital of France"}],
    )
    assert score_context_recall(case, result) == pytest.approx(2 / 3)


def test_citation_accuracy_all_correct():
    case = GoldenCase("c1", "Q", expected_doc_ids=["dA"], expected_pages=[5])
    result = make_result(
        "c1", "Q", "A",
        citations=[{"doc_id": "dA", "page": 5}, {"doc_id": "dA", "page": 5}],
    )
    assert score_citation_accuracy(case, result) == pytest.approx(1.0)


def test_citation_accuracy_no_citations():
    case = GoldenCase("c1", "Q", expected_pages=[5])
    result = make_result("c1", "Q", "A")
    assert score_citation_accuracy(case, result) == 0.0


def test_grounding_rate_alias():
    case = GoldenCase("c1", "Q", expected_pages=[5])
    result = make_result("c1", "Q", "A", citations=[{"doc_id": "d", "page": 5}])
    assert score_grounding_rate(case, result) == score_citation_accuracy(case, result)


# ---------------------- runner
def test_harness_runs_all_cases_and_summarizes():
    cases = [
        GoldenCase("c1", "What is X?", expected_keywords=["X"], expected_pages=[1]),
        GoldenCase("c2", "What is Y?", expected_keywords=["Y"], expected_pages=[2]),
    ]
    h = EvalHarness(HarnessConfig())
    results, summary = h.run(cases, stub_rag_system)
    assert len(results) == 2
    assert summary["n_cases"] == 2
    assert "faithfulness" in summary["averages"]


def test_harness_marks_pass_fail_per_threshold():
    cases = [
        # Perfect: expected_keywords cover the question words → high relevancy
        GoldenCase("good", "What is alpha?", expected_keywords=["alpha", "what", "is"], expected_pages=[1], expected_doc_ids=["d"]),
        GoldenCase("bad", "What is beta?", expected_keywords=["unrelated"], expected_pages=[99], expected_doc_ids=["d"]),
    ]
    h = EvalHarness(HarnessConfig())
    results, summary = h.run(cases, stub_rag_system)
    by_id = {r.case_id: r for r in results}
    assert by_id["good"].passed is True
    assert by_id["bad"].passed is False
    assert summary["n_passed"] == 1


def test_harness_handles_system_exceptions():
    """A broken system_fn should produce a failed result, not crash the harness."""
    def broken(_case):
        raise RuntimeError("boom")
    cases = [GoldenCase("c1", "Q", expected_keywords=["alpha"])]
    h = EvalHarness(HarnessConfig())
    results, summary = h.run(cases, broken)
    assert len(results) == 1
    assert results[0].passed is False
    assert "boom" in results[0].error


def test_harness_history_is_appended():
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd()
        try:
            os.chdir(d)
            cases = [GoldenCase("c1", "Q", expected_keywords=["alpha"])]
            h = EvalHarness(HarnessConfig())
            h.run(cases, stub_rag_system)
            h.run(cases, stub_rag_system)
            lines = Path("eval_history.jsonl").read_text().splitlines()
            assert len(lines) == 2
            entries = [json.loads(l) for l in lines]
            assert all("averages" in e for e in entries)
        finally:
            os.chdir(cwd)


def test_harness_regression_detection_runs_without_baseline():
    h = EvalHarness(HarnessConfig())
    assert h.check_regression({"averages": {"faithfulness": 0.5}}) == []


def test_harness_regression_detects_drop():
    with tempfile.TemporaryDirectory() as d:
        baseline = Path(d) / "baseline.json"
        baseline.write_text(json.dumps({"faithfulness": 0.9, "answer_relevancy": 0.9}))
        h = EvalHarness(HarnessConfig(baseline_path=str(baseline), max_regression=0.05))
        regs = h.check_regression({"averages": {"faithfulness": 0.7, "answer_relevancy": 0.9}})
        assert any("faithfulness" in r for r in regs)
        assert not any("answer_relevancy" in r for r in regs)


def test_harness_trend_returns_history():
    with tempfile.TemporaryDirectory() as d:
        h = EvalHarness(HarnessConfig())
        h.history_path = Path(d) / "history.jsonl"
        for val in (0.5, 0.6, 0.7):
            h._append_history({
                "ts": 1, "averages": {"faithfulness": val}, "stdevs": {},
                "n_cases": 1, "n_passed": 1, "pass_rate": 1.0, "regressions": []
            })
        trend = h.trend("faithfulness")
        assert [t["value"] for t in trend] == [0.5, 0.6, 0.7]


def test_set_baseline_writes_file():
    with tempfile.TemporaryDirectory() as d:
        bp = Path(d) / "base.json"
        h = EvalHarness(HarnessConfig(baseline_path=str(bp)))
        h.set_baseline({"averages": {"faithfulness": 0.85}})
        assert json.loads(bp.read_text()) == {"faithfulness": 0.85}


# ---------------------- reporters
def test_report_console_contains_pass_rate():
    cases = [GoldenCase("c", "Q", expected_keywords=["alpha"])]
    h = EvalHarness(HarnessConfig())
    results, summary = h.run(cases, stub_rag_system)
    out = report_console(results, summary)
    assert "passed" in out
    assert "faithfulness" in out


def test_report_markdown_table():
    cases = [GoldenCase("c", "Q", expected_keywords=["alpha"])]
    h = EvalHarness(HarnessConfig())
    results, summary = h.run(cases, stub_rag_system)
    md = report_markdown(results, summary)
    assert "## Averages" in md
    assert "| Metric | Score |" in md


def test_report_json_round_trip():
    cases = [GoldenCase("c", "Q", expected_keywords=["alpha"])]
    h = EvalHarness(HarnessConfig())
    results, summary = h.run(cases, stub_rag_system)
    j = report_json(results, summary)
    parsed = json.loads(j)
    assert "summary" in parsed and "results" in parsed
    assert parsed["summary"]["n_cases"] == 1


# ---------------------- all metrics produce a score in [0,1]
def test_all_metrics_bounded():
    case = GoldenCase("c", "What is the capital of France?",
                      expected_keywords=["Paris"], expected_pages=[5], expected_doc_ids=["dA"])
    result = make_result(
        "c", case.question, "Paris is the capital",
        citations=[{"doc_id": "dA", "page": 5}],
        chunks=[{"doc_id": "dA", "page": 5, "text": "Paris"}],
    )
    for name, scorer in METRICS.items():
        v = scorer(case, result)
        assert 0.0 <= v <= 1.0, f"{name} returned {v}"