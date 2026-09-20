"""evalkit command line.

    evalkit run  [--system MODULE:FN] [--golden cases.jsonl] [--corpus corpus.json]
                 [--baseline baseline.json] [--history eval_history.jsonl]
                 [--judge] [--judge-model MODEL] [--report-md FILE] [--report-json FILE]
                 [--update-baseline] [--max-regression 0.05]
    evalkit trend METRIC [--history eval_history.jsonl]

Exit codes: 0 ok, 1 usage/system error, 2 regression against baseline,
3 pass rate below --min-pass-rate. CI keys off these.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from evalkit.golden import load_cases, load_corpus
from evalkit.harness import EvalHarness, HarnessConfig, report_console, report_json, report_markdown

EXIT_OK, EXIT_ERROR, EXIT_REGRESSION, EXIT_PASS_RATE = 0, 1, 2, 3


def _resolve(spec: str):
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn or "build")


def cmd_run(a: argparse.Namespace) -> int:
    cases = load_cases(a.golden)
    corpus = load_corpus(a.corpus)
    system = _resolve(a.system)(corpus)
    cfg = HarnessConfig(baseline_path=a.baseline, history_path=a.history, max_regression=a.max_regression)
    h = EvalHarness(cfg)
    if a.judge:
        if not os.environ.get("ANTHROPIC_API_KEY") and not a.allow_no_key:
            print("--judge needs ANTHROPIC_API_KEY (or pass --allow-no-key to let the SDK resolve credentials)", file=sys.stderr)
            return EXIT_ERROR
        from evalkit.judge import JudgeScorers

        judge = JudgeScorers(model=a.judge_model) if a.judge_model else JudgeScorers()
        h.scorers.update(judge.as_scorers())
        h.config.thresholds.setdefault("judge_faithfulness", 0.7)
        h.config.thresholds.setdefault("judge_answer_relevancy", 0.7)
    results, summary = h.run(cases, system)
    print(report_console(results, summary))
    if a.report_md:
        Path(a.report_md).write_text(report_markdown(results, summary, h.baseline))
    if a.report_json:
        Path(a.report_json).write_text(report_json(results, summary))
    if a.update_baseline:
        h.set_baseline(summary)
        print(f"baseline written to {a.baseline}")
        return EXIT_OK
    if summary["regressions"]:
        print("\nREGRESSION: blocking.", file=sys.stderr)
        return EXIT_REGRESSION
    if summary["pass_rate"] < a.min_pass_rate:
        print(f"\npass rate {summary['pass_rate']:.0%} below minimum {a.min_pass_rate:.0%}", file=sys.stderr)
        return EXIT_PASS_RATE
    return EXIT_OK


def cmd_trend(a: argparse.Namespace) -> int:
    h = EvalHarness(HarnessConfig(history_path=a.history))
    for row in h.trend(a.metric):
        print(json.dumps(row))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalkit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the golden set against a system and gate on regressions")
    r.add_argument("--system", default="evalkit.systems.extractive:build", help="MODULE:FN returning a callable(GoldenCase)->CaseResult")
    r.add_argument("--golden", default=None, help="JSONL of GoldenCase dicts (default: bundled 40 cases)")
    r.add_argument("--corpus", default=None, help="corpus JSON handed to the system builder (default: bundled)")
    r.add_argument("--baseline", default="baseline.json")
    r.add_argument("--history", default="eval_history.jsonl")
    r.add_argument("--max-regression", type=float, default=0.05)
    r.add_argument("--min-pass-rate", type=float, default=0.0)
    r.add_argument("--judge", action="store_true", help="add LLM-as-judge scorers (Anthropic API)")
    r.add_argument("--judge-model", default=None)
    r.add_argument("--allow-no-key", action="store_true")
    r.add_argument("--report-md", default=None)
    r.add_argument("--report-json", default=None)
    r.add_argument("--update-baseline", action="store_true", help="write averages to --baseline instead of gating")
    r.set_defaults(fn=cmd_run)
    t = sub.add_parser("trend", help="print history for one metric")
    t.add_argument("metric")
    t.add_argument("--history", default="eval_history.jsonl")
    t.set_defaults(fn=cmd_trend)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
