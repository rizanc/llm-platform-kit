"""Bundled golden set: a 12-page corpus of Solar System notes and 40 questions.

Small on purpose. It exists so the harness, the CLI and the CI gate can be
exercised end to end without any external service. Point `--golden` and
`--corpus` at your own files for real work.
"""
from __future__ import annotations

import json
from pathlib import Path

from evalkit.harness import GoldenCase

_HERE = Path(__file__).parent
DEFAULT_CASES = _HERE / "cases.jsonl"
DEFAULT_CORPUS = _HERE / "corpus.json"


def load_cases(path: str | Path | None = None) -> list[GoldenCase]:
    p = Path(path) if path else DEFAULT_CASES
    cases = []
    for line in p.read_text().splitlines():
        if line.strip():
            cases.append(GoldenCase.from_dict(json.loads(line)))
    return cases


def load_corpus(path: str | Path | None = None) -> dict:
    """{"doc_id": str, "pages": [{"page": int, "text": str}, ...]}"""
    p = Path(path) if path else DEFAULT_CORPUS
    return json.loads(p.read_text())
