# vectorkit

A vector store interface with two backends and the operational pieces around
it: hybrid search, embedding cache, metadata filters, snapshot and restore.

```python
from vectorkit.vecstore import Chunk, HybridSearch, EmbedCache, InMemoryVectorStore
from vectorkit.qdrant_store import QdrantVectorStore

store = QdrantVectorStore(url="http://localhost:6333", collection="docs")   # or InMemoryVectorStore()
hs = HybridSearch(store, embedder, cache=EmbedCache("./cache"))
hs.index([Chunk("c1", "docA", 5, "...", {"tenant": "acme", "lang": "en"})])
hs.query("user question", k=5, filters={"tenant": "acme"})
hs.snapshot("./snapshots/2026-09-20")
```

| Concern | Implementation |
|---|---|
| Hybrid search | dense cosine + BM25-style sparse weights, fused with reciprocal rank fusion |
| Metadata filters | `eq`, `in`, `gte`, `lte`, `contains`; translated to Qdrant's filter AST server-side |
| Embedding cache | LRU in memory plus a JSON file keyed by sha256(text) |
| Index settings | HNSW m=16, ef_construct=128, int8 scalar quantisation on the Qdrant backend |
| Backup | Qdrant snapshots plus a local JSON snapshot of payloads; `restore()` into a fresh collection |
| Tenancy | `tenant` in the payload, enforced by filter at query time |

The in-memory backend runs the same code paths so the unit tests cover the
logic without a Qdrant container. `RUN_INTEGRATION=1` with
`docker run -p 6333:6333 qdrant/qdrant` runs the Qdrant test.

```
vectorkit/
  vecstore.py      VectorStore ABC, InMemoryVectorStore, HybridSearch, EmbedCache
  qdrant_store.py  QdrantVectorStore
  embedders.py     hash (tests) and ollama (real)
tests/             10 unit tests + 1 integration
```
