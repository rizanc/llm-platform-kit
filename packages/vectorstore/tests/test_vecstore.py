"""Unit tests — zero external deps. Integration gated on RUN_INTEGRATION=1."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vectorkit.vecstore import (
    Chunk,
    EmbedCache,
    HybridSearch,
    InMemoryVectorStore,
    bm25_sparse,
    reciprocal_rank_fusion,
)
from vectorkit.embedders import hash_embedder


# ---------------------- embed cache
def test_embed_cache_round_trip():
    with tempfile.TemporaryDirectory() as d:
        c = EmbedCache(d)
        assert len(c) == 0
        c.put("hello", [0.1, 0.2, 0.3])
        c.put("world", [0.4, 0.5, 0.6])
        assert c.get("hello") == [0.1, 0.2, 0.3]
        # LRU touch
        assert c.get("world") == [0.4, 0.5, 0.6]

        # disk persistence — explicit flush required
        c.flush()
        c2 = EmbedCache(d)
        assert c2.get("hello") == [0.1, 0.2, 0.3]


def test_embed_cache_get_many_partial():
    with tempfile.TemporaryDirectory() as d:
        c = EmbedCache(d)
        c.put("a", [1.0])
        vecs, missing = c.get_many(["a", "b", "c"])
        assert vecs[0] == [1.0]
        assert vecs[1] is None
        assert vecs[2] is None
        assert missing == [1, 2]


def test_embed_cache_lru_eviction():
    with tempfile.TemporaryDirectory() as d:
        c = EmbedCache(d, maxsize=2)
        c.put("a", [1.0])
        c.put("b", [2.0])
        c.put("c", [3.0])  # evicts "a"
        assert c.get("a") is None
        assert c.get("b") == [2.0]
        assert c.get("c") == [3.0]


# ---------------------- sparse + RRF
def test_bm25_sparse_basic():
    s = bm25_sparse("apple banana apple")
    assert s["apple"] > s["banana"]  # repeated term → higher weight


def test_rrf_merges_rankings():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], top_n=10)
    # a and b appear in both → highest fused scores
    top_ids = [cid for cid, _ in fused]
    assert top_ids[0] in ("a", "b")
    assert "d" in top_ids
    assert "c" in top_ids


# ---------------------- hybrid search end-to-end
def test_hybrid_search_returns_top_k():
    chunks = [
        Chunk(f"c{i}", "doc1", i, t, {"tenant": "acme"})
        for i, t in enumerate([
            "apples are red and delicious",
            "the capital of France is Paris",
            "machine learning models need data",
            "Paris is a beautiful city in Europe",
        ])
    ]
    store = InMemoryVectorStore(dim=768)
    hs = HybridSearch(store, hash_embedder)
    hs.index(chunks)
    assert store.count() == 4

    results = hs.query("Paris France", k=2)
    assert len(results) == 2
    # top results should mention Paris
    texts = [r["chunk"]["text"] for r in results]
    assert any("Paris" in t for t in texts)


def test_hybrid_search_metadata_filter_eq():
    chunks = [
        Chunk("c1", "d1", 1, "public doc", {"visibility": "public"}),
        Chunk("c2", "d2", 1, "private doc", {"visibility": "private"}),
        Chunk("c3", "d3", 1, "public again", {"visibility": "public"}),
    ]
    store = InMemoryVectorStore(dim=768)
    hs = HybridSearch(store, hash_embedder)
    hs.index(chunks)

    res = hs.query("doc", k=10, filters={"visibility": "public"})
    assert all(r["chunk"]["metadata"]["visibility"] == "public" for r in res)
    assert len(res) == 2


def test_hybrid_search_metadata_filter_in():
    chunks = [
        Chunk("c1", "d1", 1, "x", {"lang": "en"}),
        Chunk("c2", "d2", 1, "x", {"lang": "fr"}),
        Chunk("c3", "d3", 1, "x", {"lang": "es"}),
        Chunk("c4", "d4", 1, "x", {"lang": "de"}),
    ]
    store = InMemoryVectorStore(dim=768)
    hs = HybridSearch(store, hash_embedder)
    hs.index(chunks)

    res = hs.query("x", k=10, filters={"lang": {"op": "in", "value": ["en", "fr"]}})
    ids = sorted(r["chunk"]["chunk_id"] for r in res)
    assert ids == ["c1", "c2"]


def test_hybrid_search_metadata_filter_range():
    chunks = [
        Chunk(f"c{i}", "d", i, "x", {"score": i}) for i in range(5)
    ]
    store = InMemoryVectorStore(dim=768)
    hs = HybridSearch(store, hash_embedder)
    hs.index(chunks)

    res = hs.query("x", k=10, filters={"score": {"op": "gte", "value": 3}})
    scores = [r["chunk"]["metadata"]["score"] for r in res]
    assert scores == [3, 4]


# ---------------------- embed cache hits
def test_hybrid_search_uses_embed_cache():
    chunks = [
        Chunk("c1", "d", 1, "alpha bravo charlie", {}),
        Chunk("c2", "d", 2, "delta echo foxtrot", {}),
    ]
    store = InMemoryVectorStore(dim=768)
    cache = EmbedCache(tempfile.mkdtemp(), maxsize=100)
    hs = HybridSearch(store, hash_embedder, cache=cache)

    # First index → cold cache, 2 misses
    hs.index(chunks)
    assert len(cache) == 2

    # Second query → "alpha" is new (different from indexed texts)
    hs.query("alpha")
    # cache grew by 1 (the query text was embedded)
    assert len(cache) == 3


# ---------------------- snapshot / restore
def test_snapshot_round_trip():
    chunks = [
        Chunk("c1", "doc1", 5, "alpha bravo charlie", {"tenant": "acme"}),
        Chunk("c2", "doc2", 12, "delta echo foxtrot", {"tenant": "acme"}),
    ]
    store = InMemoryVectorStore(dim=768)
    hs = HybridSearch(store, hash_embedder)
    hs.index(chunks)

    with tempfile.TemporaryDirectory() as d:
        hs.snapshot(d)

        # verify files exist
        assert (Path(d) / "chunks.json").exists()
        assert (Path(d) / "vectors.npz").exists()
        assert (Path(d) / "sparse.json").exists()
        assert (Path(d) / "meta.json").exists()

        # restore into a fresh store
        store2 = InMemoryVectorStore(dim=768)
        hs2 = HybridSearch(store2, hash_embedder)
        hs2.restore(d)
        assert store2.count() == 2
        # and query works
        res = hs2.query("alpha bravo", k=2)
        assert any(r["chunk"]["chunk_id"] == "c1" for r in res)


# ---------------------- integration (Qdrant)
@pytest.mark.integration
def test_qdrant_end_to_end():
    """Requires `docker run -p 6333:6333 qdrant/qdrant` + RUN_INTEGRATION=1."""
    from vectorkit.qdrant_store import QdrantVectorStore
    store = QdrantVectorStore(collection="test_chunks_12")
    chunks = [
        Chunk("c1", "d", 1, "Paris is in France", {}),
        Chunk("c2", "d", 2, "Tokyo is in Japan", {}),
    ]
    hs = HybridSearch(store, hash_embedder)
    hs.index(chunks)
    assert store.count() >= 2
    res = hs.query("France Paris", k=1, filters={})
    assert len(res) == 1