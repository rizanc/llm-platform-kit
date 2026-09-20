"""Automated Eval Harness — gates deploys, tracks quality trends.

Implements:
  - Golden test cases (Q + expected answer + expected citations)
  - 6 metrics: faithfulness, answer_relevancy, context_precision,
               context_recall, citation_accuracy, grounding_rate
  - Regression block: fails if any metric drops > threshold vs baseline
  - Trend tracking: appends results to history.jsonl, computes deltas
  - Reporters: console / JSON / markdown

In production: swap _score_* for DeepEval / RAGAS / LangSmith calls.
Unit tests use deterministic scorers so the harness is testable
without GPU, model downloads, or network.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Callable


# --------------------------------------------------------- golden test cases
@dataclass
class GoldenCase:
    case_id: str
    question: str
    expected_answer: str | None = None
    expected_keywords: list[str] = field(default_factory=list)
    expected_pages: list[int] = field(default_factory=list)
    expected_doc_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    case_id: str
    question: str
    answer: str
    citations: list[dict]   # [{"doc_id": ..., "page": ...}]
    retrieved_chunks: list[dict]
    metrics: dict[str, float]
    passed: bool
    error: str | None = None


# --------------------------------------------------------- deterministic scorers
# In prod these call out to DeepEval / RAGAS / LangSmith. Here they are pure
# functions of (case, result) so the harness logic is testable.

def score_faithfulness(case: GoldenCase, result: CaseResult) -> float:
    """Are claims in the answer supported by retrieved chunks?

    Score = |overlapping unigrams(answer, chunks)| / |unigrams(answer)|.
    Penalizes hallucinated tokens.
    """
    ans_tokens = set(_tokens(result.answer))
    if not ans_tokens:
        return 1.0
    ctx_tokens = set()
    for c in result.retrieved_chunks:
        ctx_tokens.update(_tokens(c.get("text", "")))
    if not ctx_tokens:
        return 0.0
    return len(ans_tokens & ctx_tokens) / len(ans_tokens)


def score_answer_relevancy(case: GoldenCase, result: CaseResult) -> float:
    """Does the answer address the question?

    Score = |unigrams(answer) ∩ unigrams(question)| / |unigrams(question)|.
    """
    q = set(_tokens(case.question))
    if not q:
        return 1.0
    a = set(_tokens(result.answer))
    return len(q & a) / len(q)


def score_context_precision(case: GoldenCase, result: CaseResult) -> float:
    """What fraction of retrieved chunks are actually relevant?

    A retrieved chunk is relevant if it shares >=1 expected keyword with the question
    or if its page/doc is in the expected set.
    """
    if not result.retrieved_chunks:
        return 0.0
    expected_pages = set(case.expected_pages)
    expected_docs = set(case.expected_doc_ids)
    keywords = {k.lower() for k in case.expected_keywords}
    hits = 0
    for c in result.retrieved_chunks:
        if expected_pages and c.get("page") in expected_pages:
            hits += 1
        elif expected_docs and c.get("doc_id") in expected_docs:
            hits += 1
        elif keywords and any(k in c.get("text", "").lower() for k in keywords):
            hits += 1
        else:
            # weak fallback: any token overlap with question
            q_tokens = set(_tokens(case.question))
            c_tokens = set(_tokens(c.get("text", "")))
            if q_tokens & c_tokens:
                hits += 1
    return hits / len(result.retrieved_chunks)


def score_context_recall(case: GoldenCase, result: CaseResult) -> float:
    """Were all expected chunks retrieved?"""
    if not case.expected_pages and not case.expected_doc_ids and not case.expected_keywords:
        return 1.0
    retrieved_keys = {(c.get("doc_id"), c.get("page")) for c in result.retrieved_chunks}
    expected_keys = {(d, p) for d, p in zip(case.expected_doc_ids, case.expected_pages)}
    if not expected_keys and case.expected_keywords:
        # no expected doc/page; fall back to keyword coverage
        ctx_text = " ".join(c.get("text", "") for c in result.retrieved_chunks).lower()
        hits = sum(1 for k in case.expected_keywords if k.lower() in ctx_text)
        return hits / len(case.expected_keywords)
    if not expected_keys:
        return 1.0
    hits = sum(1 for k in expected_keys if k in retrieved_keys)
    return hits / len(expected_keys)


def score_citation_accuracy(case: GoldenCase, result: CaseResult) -> float:
    """What fraction of citations in the answer map to expected sources?"""
    if not result.citations:
        return 0.0
    if not case.expected_pages and not case.expected_doc_ids:
        return 1.0  # nothing to check against
    hits = 0
    for cit in result.citations:
        if case.expected_pages and cit.get("page") in case.expected_pages:
            hits += 1
        elif case.expected_docs and cit.get("doc_id") in case.expected_doc_ids:
            hits += 1
    return hits / len(result.citations)


def score_grounding_rate(case: GoldenCase, result: CaseResult) -> float:
    """Same as citation_accuracy but in the framework of project #1."""
    return score_citation_accuracy(case, result)


# default registry — keys are public metric names
METRICS: dict[str, Callable[[GoldenCase, CaseResult], float]] = {
    "faithfulness": score_faithfulness,
    "answer_relevancy": score_answer_relevancy,
    "context_precision": score_context_precision,
    "context_recall": score_context_recall,
    "citation_accuracy": score_citation_accuracy,
    "grounding_rate": score_grounding_rate,
}


def _tokens(text: str) -> list[str]:
    import re
    return re.findall(r"[A-Za-z0-9]+", text.lower())


# --------------------------------------------------------- runner
@dataclass
class HarnessConfig:
    """Pass/fail thresholds. Below the threshold → regression → block deploy."""
    thresholds: dict[str, float] = field(default_factory=lambda: {
        "faithfulness": 0.6,
        "answer_relevancy": 0.5,
        "context_precision": 0.5,
        "context_recall": 0.7,
        "citation_accuracy": 0.8,
        "grounding_rate": 0.8,
    })
    # Absolute drop vs baseline that triggers a regression block.
    max_regression: float = 0.05
    baseline_path: str | None = None


class EvalHarness:
    def __init__(self, config: HarnessConfig | None = None):
        self.config = config or HarnessConfig()
        self.history_path = Path("./eval_history.jsonl")
        self.baseline = self._load_baseline()

    # --- run
    def run(self, cases: list[GoldenCase], system_fn: Callable) -> tuple[list[CaseResult], dict]:
        """system_fn(case) → CaseResult. Caller wires this to their RAG."""
        results: list[CaseResult] = []
        for case in cases:
            t0 = time.time()
            try:
                result = system_fn(case)
                result.error = None
            except Exception as e:
                result = CaseResult(
                    case_id=case.case_id,
                    question=case.question,
                    answer="",
                    citations=[],
                    retrieved_chunks=[],
                    metrics={},
                    passed=False,
                    error=str(e),
                )
            result.metrics = {
                name: scorer(case, result) for name, scorer in METRICS.items()
            }
            result.passed = all(
                result.metrics.get(m, 0.0) >= thr
                for m, thr in self.config.thresholds.items()
                if m in result.metrics
            ) and result.error is None
            results.append(result)

        summary = self._summarize(results)
        self._append_history(summary)
        return results, summary

    # --- baseline / regression
    def _load_baseline(self) -> dict[str, float] | None:
        if not self.config.baseline_path:
            return None
        p = Path(self.config.baseline_path)
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def check_regression(self, current: dict) -> list[str]:
        """Return list of metric names that regressed beyond max_regression."""
        if not self.baseline:
            return []
        regressions = []
        for metric, value in current.get("averages", {}).items():
            base = self.baseline.get(metric)
            if base is None:
                continue
            if base - value > self.config.max_regression:
                regressions.append(f"{metric}: {base:.3f} → {value:.3f} (Δ={base - value:.3f})")
        return regressions

    def set_baseline(self, summary: dict) -> None:
        if not self.config.baseline_path:
            return
        Path(self.config.baseline_path).write_text(json.dumps(summary["averages"], indent=2))

    # --- trend
    def trend(self, metric: str) -> list[dict]:
        """Return [{ts, value}, ...] for the given metric across history."""
        if not self.history_path.exists():
            return []
        out = []
        for line in self.history_path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if metric in entry.get("averages", {}):
                out.append({"ts": entry["ts"], "value": entry["averages"][metric]})
        return out

    # --- helpers
    def _summarize(self, results: list[CaseResult]) -> dict:
        per_metric: dict[str, list[float]] = defaultdict(list)
        for r in results:
            for m, v in r.metrics.items():
                per_metric[m].append(v)
        averages = {m: mean(v) for m, v in per_metric.items() if v}
        stdevs = {m: stdev(v) if len(v) > 1 else 0.0 for m, v in per_metric.items() if v}
        n_pass = sum(1 for r in results if r.passed)
        return {
            "ts": int(time.time()),
            "n_cases": len(results),
            "n_passed": n_pass,
            "pass_rate": n_pass / len(results) if results else 0.0,
            "averages": averages,
            "stdevs": stdevs,
            "regressions": self.check_regression({"averages": averages}),
        }

    def _append_history(self, summary: dict) -> None:
        with self.history_path.open("a") as f:
            f.write(json.dumps(summary) + "\n")


# --------------------------------------------------------- reporters
def report_console(results: list[CaseResult], summary: dict) -> str:
    lines = [f"\n{'='*60}", f"EVAL: {summary['n_passed']}/{summary['n_cases']} passed ({summary['pass_rate']:.0%})", "="*60]
    for m, v in summary["averages"].items():
        sd = summary["stdevs"].get(m, 0.0)
        lines.append(f"  {m:24s} {v:.3f}  σ={sd:.3f}")
    if summary["regressions"]:
        lines.append("\n⚠ REGRESSIONS:")
        for r in summary["regressions"]:
            lines.append(f"  - {r}")
    lines.append("="*60)
    for r in results:
        flag = "✓" if r.passed else "✗"
        lines.append(f"  {flag} {r.case_id:30s} {r.metrics}")
        if r.error:
            lines.append(f"      ERROR: {r.error}")
    return "\n".join(lines)


def report_markdown(results: list[CaseResult], summary: dict) -> str:
    lines = ["# Eval Report", "", f"**Pass rate:** {summary['pass_rate']:.0%} ({summary['n_passed']}/{summary['n_cases']})", "", "## Averages", ""]
    lines.append("| Metric | Score | σ |")
    lines.append("|---|---|---|")
    for m, v in summary["averages"].items():
        lines.append(f"| {m} | {v:.3f} | {summary['stdevs'].get(m, 0.0):.3f} |")
    if summary["regressions"]:
        lines.append("\n## Regressions")
        for r in summary["regressions"]:
            lines.append(f"- {r}")
    return "\n".join(lines)


def report_json(results: list[CaseResult], summary: dict) -> str:
    return json.dumps(
        {
            "summary": summary,
            "results": [asdict(r) for r in results],
        },
        indent=2,
    )