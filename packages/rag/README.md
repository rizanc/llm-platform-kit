# citerag

Retrieval-augmented answering that cites page numbers and checks its own
citations. SQLite only; no vector database needed below ~10k chunks.

```python
from citerag.rag import HybridStore, Chunk, build_graph, tokenize
from citerag.embedders import ollama_embedder
from citerag.llm import stub_llm  # replace with your model call

store = HybridStore("./rag.db")
chunks = [Chunk("c1", "docA", 5, "the capital of France is Paris", tokenize("..."))]
store.add(chunks, ollama_embedder()([c.text for c in chunks]))

graph = build_graph(store, ollama_embedder(), stub_llm)
result = graph.invoke({"query": "What is the capital of France?"})
result["answer"]     # "... [docA, p.5]"
result["grounding"]  # {"grounding_rate": 1.0, "ungrounded": []}
```

## Pipeline

```
query -> embed -> retrieve (FTS5 BM25 + cosine, fused with RRF)
      -> fetch -> rerank (cross-encoder, or token-overlap fallback)
      -> generate (prompt demands "[doc_id, p.N]" markers)
      -> verify (every cited page must be in the retrieved set)
```

Orchestrated as a LangGraph state machine so each step is inspectable and
individually testable.

## Notes

- `bm25_search` takes FTS5 MATCH syntax. Sanitise user input (e.g. join
  tokens with `OR`) before calling it.
- FTS5's `bm25()` is lower-is-better. The store orders ascending and negates
  the score so callers see higher-is-better. This was inverted in the first
  version; `test_bm25_ranks_more_relevant_chunk_first` guards it.
- `hash_embedder` is a deterministic stand-in for tests, not a semantic model.
  Fusing its rankings with BM25 makes retrieval worse, not better; use a real
  embedder (Ollama `nomic-embed-text` is wired) for hybrid search.
- `grounding_check` is the piece worth copying: it turns "did the model cite
  something it never saw" into a number you can gate on.

```
citerag/
  rag.py         Chunk, HybridStore, reciprocal_rank_fusion, rerank, grounding_check, build_graph
  embedders.py   hash (tests) and ollama (real)
  llm.py         stub_llm
tests/           10 unit tests + 1 integration (needs Ollama)
```
