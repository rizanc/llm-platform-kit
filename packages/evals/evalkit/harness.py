"""Eval harness: golden cases, pluggable scorers, baseline regression gate.

The harness is deliberately small and dependency-free so it can sit in a CI
job. What it gives you:

  - GoldenCase / CaseResult dataclasses (question, expected evidence, answer,
    citations, retrieved chunks)
  - A registry of scorers. The built-in ones are deterministic lexical
    proxies (token overlap, expected-page hit rate). They are cheap and
    stable, which is what you want for a merge gate. `evalkit.judge` adds
    LLM-as-judge scorers for faithfulness and relevancy when an API key is
    available; both kinds plug into the same registry.
  - Per-case pass/fail against thresholds
  - Regression detection against a committed baseline (any metric average
    dropping by more than `max_regression` blocks the run)
  - Append-only history for trend queries
  - Console / Markdown / JSON reporters

Scores from the lexical scorers are proxies, not ground truth. Treat them as
a tripwire: a sudden drop means something changed, and a human should look.
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Callable

_WORD = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


# ------------------------------------------------------------------ data model
@dataclass
class GoldenCase:
    case_id: str
    question: str
    expected_answer: str | None = None
    expected_keywords: list[str] = field(default_factory=list)
    expected_pages: list[int] = field(default_factory=list)
    expected_doc_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenCase":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CaseResult:
    case_id: str
    question: str
    answer: str
    citations: list[dict]        # [{"doc_id": ..., "page": ...}]
    retrieved_chunks: list[dict]  # [{"doc_id": ..., "page": ..., "text": ...}]
    metrics: dict[str, float] = field(default_factory=dict)
    passed: bool = False
    error: str | None = None
    latency_ms: float = 0.0


Scorer = Callable[[GoldenCase, CaseResult], float]


# ------------------------------------------------------ deterministic scorers
def score_faithfulness(case: GoldenCase, result: CaseResult) -> float:
    """Fraction of answer unigrams that appear somewhere in the retrieved context.

    Lexical proxy for "is the answer supported by the context". Penalises
    tokens that came from nowhere. For a semantic version use
    `evalkit.judge.JudgeScorers.faithfulness`.
    """
    ans = set(_tokens(result.answer))
    if not ans:
        return 1.0
    ctx: set[str] = set()
    for c in result.retrieved_chunks:
        ctx.update(_tokens(c.get("text", "")))
    if not ctx:
        return 0.0
    return len(ans & ctx) / len(ans)


def score_answer_relevancy(case: GoldenCase, result: CaseResult) -> float:
    """Fraction of question unigrams echoed in the answer. Lexical proxy."""
    q = set(_tokens(case.question))
    if not q:
        return 1.0
    return len(q & set(_tokens(result.answer))) / len(q)


def score_context_precision(case: GoldenCase, result: CaseResult) -> float:
    """Fraction of retrieved chunks that are relevant to the case.

    A chunk counts as relevant if its page or doc is expected, or it contains
    an expected keyword. Falls back to any token overlap with the question.
    """
    if not result.retrieved_chunks:
        return 0.0
    pages, docs = set(case.expected_pages), set(case.expected_doc_ids)
    kws = {k.lower() for k in case.expected_keywords}
    q_tokens = set(_tokens(case.question))
    hits = 0
    for c in result.retrieved_chunks:
        text = c.get("text", "").lower()
        if pages and c.get("page") in pages:
            hits += 1
        elif docs and c.get("doc_id") in docs:
            hits += 1
        elif kws and any(k in text for k in kws):
            hits += 1
        elif q_tokens & set(_tokens(text)):
            hits += 1
    return hits / len(result.retrieved_chunks)


def score_context_recall(case: GoldenCase, result: CaseResult) -> float:
    """Were the expected (doc, page) pairs retrieved? Keyword coverage as fallback."""
    if not (case.expected_pages or case.expected_doc_ids or case.expected_keywords):
        return 1.0
    retrieved = {(c.get("doc_id"), c.get("page")) for c in result.retrieved_chunks}
    expected = set(zip(case.expected_doc_ids, case.expected_pages))
    if not expected and case.expected_keywords:
        ctx = " ".join(c.get("text", "") for c in result.retrieved_chunks).lower()
        return sum(k.lower() in ctx for k in case.expected_keywords) / len(case.expected_keywords)
    if not expected:
        return 1.0
    return sum(k in retrieved for k in expected) / len(expected)


def score_citation_accuracy(case: GoldenCase, result: CaseResult) -> float:
    """Fraction of citations that point at an expected page or doc."""
    if not result.citations:
        return 0.0
    if not case.expected_pages and not case.expected_doc_ids:
        return 1.0
    hits = 0
    for cit in result.citations:
        if case.expected_pages and cit.get("page") in case.expected_pages:
            hits += 1
        elif case.expected_doc_ids and cit.get("doc_id") in case.expected_doc_ids:
            hits += 1
    return hits / len(result.citations)


def score_grounding_rate(case: GoldenCase, result: CaseResult) -> float:
    """Fraction of citations whose (doc, page) is actually in the retrieved set.

    Different from citation_accuracy: this catches citations the model
    invented, regardless of whether the golden case expected that page.
    """
    if not result.citations:
        return 0.0
    retrieved = {(c.get("doc_id"), c.get("page")) for c in result.retrieved_chunks}
    return sum((c.get("doc_id"), c.get("page")) in retrieved for c in result.citations) / len(result.citations)


DEFAULT_SCORERS: dict[str, Scorer] = {
    "faithfulness": score_faithfulness,
    "answer_relevancy": score_answer_relevancy,
    "context_precision": score_context_precision,
    "context_recall": score_context_recall,
    "citation_accuracy": score_citation_accuracy,
    "grounding_rate": score_grounding_rate,
}
METRICS = DEFAULT_SCORERS  # backwards-compatible alias


# --------------------------------------------------------------------- runner
@dataclass
class HarnessConfig:
    """Thresholds are per-case pass/fail. max_regression is the run-level gate."""
    thresholds: dict[str, float] = field(default_factory=lambda: {
        "faithfulness": 0.6,
        "answer_relevancy": 0.5,
        "context_precision": 0.5,
        "context_recall": 0.7,
        "citation_accuracy": 0.8,
        "grounding_rate": 0.8,
    })
    max_regression: float = 0.05
    baseline_path: str | None = None
    history_path: str = "./eval_history.jsonl"


class EvalHarness:
    def __init__(self, config: HarnessConfig | None = None, scorers: dict[str, Scorer] | None = None):
        self.config = config or HarnessConfig()
        self.scorers = dict(scorers) if scorers is not None else dict(DEFAULT_SCORERS)
        self.history_path = Path(self.config.history_path)
        self.baseline = self._load_baseline()

    def add_scorer(self, name: str, fn: Scorer, threshold: float | None = None) -> None:
        self.scorers[name] = fn
        if threshold is not None:
            self.config.thresholds[name] = threshold

    def run(self, cases: list[GoldenCase], system_fn: Callable[[GoldenCase], CaseResult]) -> tuple[list[CaseResult], dict]:
        """system_fn(case) -> CaseResult. Exceptions become failed cases, not crashes."""
        results: list[CaseResult] = []
        for case in cases:
            t0 = time.perf_counter()
            try:
                result = system_fn(case)
                result.error = None
            except Exception as e:  # noqa: BLE001 - we want every failure recorded
                result = CaseResult(case.case_id, case.question, "", [], [], error=f"{type(e).__name__}: {e}")
            result.latency_ms = (time.perf_counter() - t0) * 1000
            result.metrics = {}
            for name, scorer in self.scorers.items():
                try:
                    result.metrics[name] = float(scorer(case, result))
                except Exception as e:  # noqa: BLE001
                    result.metrics[name] = 0.0
                    result.error = (result.error or "") + f" [scorer {name}: {type(e).__name__}: {e}]"
            result.passed = result.error is None and all(
                result.metrics.get(m, 0.0) >= thr
                for m, thr in self.config.thresholds.items()
                if m in result.metrics
            )
            results.append(result)
        summary = self._summarize(results)
        self._append_history(summary)
        return results, summary

    # --- baseline / regression
    def _load_baseline(self) -> dict[str, float] | None:
        if not self.config.baseline_path:
            return None
        p = Path(self.config.baseline_path)
        return json.loads(p.read_text()) if p.exists() else None

    def check_regression(self, current: dict) -> list[str]:
        """Metric names (with deltas) whose average dropped more than max_regression vs baseline."""
        if not self.baseline:
            return []
        out = []
        for metric, value in current.get("averages", {}).items():
            base = self.baseline.get(metric)
            if base is not None and base - value > self.config.max_regression:
                out.append(f"{metric}: {base:.3f} -> {value:.3f} (delta={base - value:.3f})")
        return out

    def set_baseline(self, summary: dict) -> None:
        if self.config.baseline_path:
            Path(self.config.baseline_path).write_text(json.dumps(summary["averages"], indent=2, sort_keys=True) + "\n")

    # --- trend
    def trend(self, metric: str) -> list[dict]:
        if not self.history_path.exists():
            return []
        out = []
        for line in self.history_path.read_text().splitlines():
            if line.strip():
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
        stdevs = {m: (stdev(v) if len(v) > 1 else 0.0) for m, v in per_metric.items() if v}
        n_pass = sum(r.passed for r in results)
        return {
            "ts": int(time.time()),
            "n_cases": len(results),
            "n_passed": n_pass,
            "pass_rate": n_pass / len(results) if results else 0.0,
            "averages": averages,
            "stdevs": stdevs,
            "p50_latency_ms": _percentile([r.latency_ms for r in results], 0.5),
            "regressions": self.check_regression({"averages": averages}),
        }

    def _append_history(self, summary: dict) -> None:
        with self.history_path.open("a") as f:
            f.write(json.dumps(summary) + "\n")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


# ------------------------------------------------------------------ reporters
def report_console(results: list[CaseResult], summary: dict) -> str:
    bar = "=" * 60
    lines = [bar, f"EVAL: {summary['n_passed']}/{summary['n_cases']} passed ({summary['pass_rate']:.0%})", bar]
    for m, v in summary["averages"].items():
        lines.append(f"  {m:24s} {v:.3f}  sd={summary['stdevs'].get(m, 0.0):.3f}")
    if summary["regressions"]:
        lines.append("\nREGRESSIONS:")
        lines.extend(f"  - {r}" for r in summary["regressions"])
    lines.append(bar)
    for r in results:
        flag = "PASS" if r.passed else "FAIL"
        lines.append(f"  {flag} {r.case_id:30s} " + " ".join(f"{k}={v:.2f}" for k, v in r.metrics.items()))
        if r.error:
            lines.append(f"       ERROR: {r.error}")
    return "\n".join(lines)


def report_markdown(results: list[CaseResult], summary: dict, baseline: dict | None = None) -> str:
    lines = [
        "# Eval report",
        "",
        f"**Pass rate:** {summary['pass_rate']:.0%} ({summary['n_passed']}/{summary['n_cases']})",
        "",
        "| Metric | Score | sd | Baseline | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for m, v in summary["averages"].items():
        b = (baseline or {}).get(m)
        b_s = f"{b:.3f}" if b is not None else "-"
        d_s = f"{v - b:+.3f}" if b is not None else "-"
        lines.append(f"| {m} | {v:.3f} | {summary['stdevs'].get(m, 0.0):.3f} | {b_s} | {d_s} |")
    if summary["regressions"]:
        lines += ["", "## Regressions", ""] + [f"- {r}" for r in summary["regressions"]]
    failed = [r for r in results if not r.passed]
    if failed:
        lines += ["", "## Failed cases", ""]
        lines += [f"- `{r.case_id}`: " + (r.error or ", ".join(f"{k}={v:.2f}" for k, v in r.metrics.items())) for r in failed]
    return "\n".join(lines) + "\n"


def report_json(results: list[CaseResult], summary: dict) -> str:
    return json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2)
