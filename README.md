# llm-platform-kit

[![ci](https://github.com/rizanc/llm-platform-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/rizanc/llm-platform-kit/actions/workflows/ci.yml)
[![eval-gate](https://github.com/rizanc/llm-platform-kit/actions/workflows/eval-gate.yml/badge.svg)](https://github.com/rizanc/llm-platform-kit/actions/workflows/eval-gate.yml)

Small, tested building blocks for running LLM features in production. Each
package is a few hundred lines with its own tests and README, and they
compose: the eval harness measures the RAG package, the router feeds the
spend log, the vector store swaps in behind the RAG store.

| Package | What it does | Tests |
|---|---|---|
| [`evalkit`](packages/evals) | Golden cases, deterministic and LLM-as-judge scorers, baseline regression gate wired into CI | 42 |
| [`modelrouter`](packages/router) | Complexity-tiered routing to the cheapest capable model, LiteLLM caller, spend log that matches the bill | 27 |
| [`citerag`](packages/rag) | SQLite FTS5 + cosine retrieval fused with RRF, page-level citations, grounding check | 11 |
| [`vectorkit`](packages/vectorstore) | One interface over in-memory and Qdrant backends: hybrid search, filters, embed cache, snapshots | 12 |
| [`localrag`](packages/local-dev) | Ollama + LanceDB + FastAPI in docker compose for zero-cost local development | 4 |

## The merge gate

The part worth reading first is `.github/workflows/eval-gate.yml`. Every pull
request runs the 40-case golden set against the reference system and fails
if any metric average drops more than 5 points below the committed baseline.
With an `ANTHROPIC_API_KEY` secret, LLM-as-judge scores are added; without
one, the deterministic scorers still gate. The report is posted on the PR.

```
uv sync --all-packages --all-extras
uv run pytest -q                       # 89 tests; 84 run offline, 5 integration tests skip
uv run evalkit run                     # golden set vs the extractive baseline
uv run evalkit run --judge             # add judge scorers (needs ANTHROPIC_API_KEY)
python -m modelrouter report spend.jsonl
```

Unit tests never touch the network. Integration tests (Ollama, Qdrant, the
compose stack, a live judge) are opt-in with `RUN_INTEGRATION=1`.

## What this is and is not

This is the generic, publishable layer of patterns I run in a private
delivery pipeline: gate on evals, route by cost, log spend per tenant, verify
citations. It is not a framework. Copy the piece you need.

Scores in the eval README are from the bundled extractive baseline and are
there to show the harness works end to end, not to claim retrieval quality
on real data. Judge scores are not published because they need a key at run
time.

## History

Started as five separate repositories scaffolded in August 2026. Consolidated
here in September 2026 with the pieces that were missing: an actual judge
scorer, a real caller with provider-reported costs, a CI gate, and two
correctness fixes found on the way (an inverted BM25 ordering and a citation
scorer referencing a field that did not exist).

MIT licensed.
