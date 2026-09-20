# evalkit

Eval harness for LLM features with a merge gate. Zero dependencies for the
core; `anthropic` for the judge, `citerag` for the bundled reference system.

```
uv run evalkit run                      # 40 bundled cases vs the extractive baseline
uv run evalkit run --judge              # + LLM-as-judge scorers (ANTHROPIC_API_KEY)
uv run evalkit run --system my.pipeline:build --golden my/cases.jsonl --corpus my/corpus.json
uv run evalkit run --update-baseline    # promote the current run to the committed baseline
uv run evalkit trend context_recall     # history of one metric
```

Exit code 2 means a metric average dropped more than `--max-regression`
(default 0.05) below the baseline. The `eval-gate` workflow in this repo runs
exactly that on every pull request and posts the report as a PR comment.

## What is measured

| Metric | Kind | What it measures |
|---|---|---|
| `context_recall` | deterministic | expected (doc, page) pairs present in the retrieved set |
| `context_precision` | deterministic | share of retrieved chunks that are relevant to the case |
| `citation_accuracy` | deterministic | share of citations pointing at an expected page or doc |
| `grounding_rate` | deterministic | share of citations pointing at a page that was actually retrieved |
| `faithfulness` | deterministic | share of answer tokens present in the context (lexical proxy) |
| `answer_relevancy` | deterministic | share of question tokens echoed in the answer (lexical proxy) |
| `judge_faithfulness` | LLM judge | every claim supported by the context |
| `judge_answer_relevancy` | LLM judge | the answer addresses the question |
| `judge_correctness` | LLM judge | agreement with `expected_answer` when one is given |

The lexical scorers are proxies. They are stable and free, which makes them a
good tripwire in CI. The judge scorers are the ones to read when deciding
whether an answer is good. One judge request per case returns all three
verdicts as structured JSON; the rubric sits behind a prompt-cache breakpoint.

## Results on the bundled set

Reference system: `evalkit.systems.extractive` (BM25 over the 12-page
corpus, top 3 chunks, best-overlap sentence, page citation). Run locally with
`uv run evalkit run` on 2026-09-20:

| Metric | Score |
|---|---:|
| context_recall | 1.000 |
| context_precision | 1.000 |
| citation_accuracy | 1.000 |
| grounding_rate | 1.000 |
| faithfulness | 1.000 |
| answer_relevancy | 0.554 |
| pass rate (all thresholds) | 62% (25/40) |

Read this as: retrieval finds the right page on every case, the extractive
"answer" is copied from context so it is trivially faithful, and the lexical
relevancy score is low because a copied sentence rarely repeats the question's
words. That last number is why the judge scorers exist. Judge results are not
published here because they require an API key at run time; the CI workflow
adds them when the `ANTHROPIC_API_KEY` secret is set.

The committed baseline is `evalkit/golden/baseline.json`.

## Wiring your own system

```python
# my/pipeline.py
from evalkit.harness import CaseResult, GoldenCase

def build(corpus: dict):
    index = ...  # build once
    def answer(case: GoldenCase) -> CaseResult:
        out = index.ask(case.question)
        return CaseResult(case.case_id, case.question, out.text,
                          citations=[{"doc_id": d, "page": p} for d, p in out.cites],
                          retrieved_chunks=[{"doc_id": c.doc, "page": c.page, "text": c.text} for c in out.chunks])
    return answer
```

Golden cases are one JSON object per line with `case_id`, `question`, and any
of `expected_answer`, `expected_keywords`, `expected_pages`,
`expected_doc_ids`, `tags`. Include some unanswerable questions.

## Layout

```
evalkit/
  harness.py        GoldenCase, CaseResult, scorers, EvalHarness, reporters
  judge.py          JudgeScorers (Anthropic API, structured output, cached verdicts)
  cli.py            evalkit run / trend
  golden/           corpus.json, cases.jsonl, baseline.json
  systems/          extractive.py (reference), stub.py (harness self-test)
tests/              42 tests, no network
```
