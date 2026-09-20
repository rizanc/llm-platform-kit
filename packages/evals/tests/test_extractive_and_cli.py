"""End to end: bundled golden set -> extractive system -> harness -> CLI exit codes. No network."""
import json

import pytest

from evalkit.cli import EXIT_OK, EXIT_PASS_RATE, EXIT_REGRESSION, main
from evalkit.golden import load_cases, load_corpus
from evalkit.harness import EvalHarness, HarnessConfig
from evalkit.systems.extractive import build


def test_bundled_golden_set_loads():
    cases, corpus = load_cases(), load_corpus()
    assert len(cases) == 40
    assert len(corpus["pages"]) == 12
    assert len({c.case_id for c in cases}) == 40


def test_extractive_system_answers_with_citation(tmp_path):
    answer = build(load_corpus(), db_path=tmp_path / "c.db")
    case = next(c for c in load_cases() if c.case_id == "largest-moon")
    r = answer(case)
    assert "Ganymede" in r.answer
    assert r.citations == [{"doc_id": "solar-system-notes", "page": 6}]
    assert any(ch["page"] == 6 for ch in r.retrieved_chunks)


def test_extractive_retrieval_quality_on_golden_set(tmp_path):
    """Floor, not a target: retrieval must find the expected page for most cases."""
    answer = build(load_corpus(), db_path=tmp_path / "c.db")
    h = EvalHarness(HarnessConfig(history_path=str(tmp_path / "h.jsonl")))
    _, summary = h.run(load_cases(), answer)
    assert summary["averages"]["context_recall"] >= 0.85
    assert summary["averages"]["citation_accuracy"] >= 0.80


def test_cli_run_writes_reports_and_baseline(tmp_path, capsys):
    base, hist, md = tmp_path / "b.json", tmp_path / "h.jsonl", tmp_path / "r.md"
    rc = main(["run", "--baseline", str(base), "--history", str(hist), "--report-md", str(md), "--update-baseline"])
    assert rc == EXIT_OK
    assert base.exists() and "context_recall" in json.loads(base.read_text())
    assert "| Metric | Score |" in md.read_text()
    assert "EVAL:" in capsys.readouterr().out


def test_cli_blocks_on_regression(tmp_path):
    base, hist = tmp_path / "b.json", tmp_path / "h.jsonl"
    # Baseline with an unreachable score so any real run is a regression.
    base.write_text(json.dumps({"context_recall": 1.5}))
    rc = main(["run", "--baseline", str(base), "--history", str(hist)])
    assert rc == EXIT_REGRESSION


def test_cli_blocks_on_min_pass_rate(tmp_path):
    rc = main(["run", "--history", str(tmp_path / "h.jsonl"), "--baseline", str(tmp_path / "none.json"), "--min-pass-rate", "1.01"])
    assert rc == EXIT_PASS_RATE


def test_cli_trend(tmp_path, capsys):
    hist = tmp_path / "h.jsonl"
    main(["run", "--history", str(hist), "--baseline", str(tmp_path / "none.json")])
    main(["run", "--history", str(hist), "--baseline", str(tmp_path / "none.json")])
    assert main(["trend", "context_recall", "--history", str(hist)]) == EXIT_OK
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    assert len(lines) == 2 and "value" in json.loads(lines[0])


def test_cli_judge_requires_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rc = main(["run", "--judge", "--history", str(tmp_path / "h.jsonl"), "--baseline", str(tmp_path / "none.json")])
    assert rc == 1
