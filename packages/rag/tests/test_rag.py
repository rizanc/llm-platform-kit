"""Unit tests — no model downloads, no GPU, no LangGraph runtime required
for the pure-logic tests. Integration tests gated on RUN_INTEGRATION=1."""
import tempfile
import numpy as np
import pytest

from citerag.rag import (
    Chunk,
    HybridStore,
    chunk_text,
    tokenize,
    reciprocal_rank_fusion,
    cross_encoder_rerank,
    extract_citations,
    grounding_check,
    build_graph,
)
from citerag.embedders import hash_embedder
from citerag.llm import stub_llm


# ---------------------- pure logic (always run)
def test_chunk_text_overlap_and_bounds():
    text = "abcde" * 300  # 1500 chars
    chunks = chunk_text(text, chunk_size=400, overlap=50)
    assert 3 <= len(chunks) <= 5
    assert all(len(c) <= 400 for c in chunks)
    assert "abcde" in chunks[0]


def test_tokenize_lowercase_alnum():
    assert tokenize("Hello, World! 42") == ["hello", "world", "42"]


def test_rrf_merges_rankings():
    a = [("x", 1.0), ("y", 0.5), ("z", 0.1)]
    b = [("y", 0.9), ("x", 0.4), ("w", 0.2)]
    merged = reciprocal_rank_fusion([a, b], top_n=10)
    # x and y are in both → highest fused score
    top_ids = [cid for cid, _ in merged]
    assert top_ids[0] in ("x", "y")
    assert set(top_ids[:4]) == {"x", "y", "z", "w"}


def test_cross_encoder_fallback_is_deterministic():
    chunks = [
        Chunk(chunk_id="a", doc_id="d1", page=1, text="orange grape kiwi"),
        Chunk(chunk_id="b", doc_id="d1", page=2, text="apple banana cherry apple banana"),
        Chunk(chunk_id="c", doc_id="d1", page=3, text="orange"),
    ]
    out = cross_encoder_rerank("apple banana", chunks)
    # "b" has both query tokens → highest Jaccard → first
    assert out[0].chunk_id == "b"


def test_extract_citations_handles_both_forms():
    text = "Foo [p.5] and bar [docA, p.12]."
    assert extract_citations(text) == [(None, 5), ("docA", 12)]


def test_grounding_check_counts_correct():
    retrieved = {
        "a": Chunk("a", "docA", 5, "x"),
        "b": Chunk("b", "docA", 12, "y"),
    }
    res = grounding_check("Answer [docA, p.5] and [docA, p.99].", retrieved)
    assert res["n_citations"] == 2
    assert res["n_grounded"] == 1
    assert res["grounding_rate"] == 0.5
    assert ("docA", 99) in res["ungrounded"]


# ---------------------- store (always run, uses tempdir)
def test_hybrid_store_round_trip_bm25():
    with tempfile.TemporaryDirectory() as d:
        s = HybridStore(f"{d}/h.db")
        chunks = [
            Chunk("c1", "doc1", 1, "the quick brown fox", tokenize("the quick brown fox")),
            Chunk("c2", "doc1", 2, "jumps over the lazy dog", tokenize("jumps over the lazy dog")),
        ]
        emb = hash_embedder([c.text for c in chunks])
        s.add(chunks, emb)
        assert s.count() == 2
        hits = s.bm25_search("fox", k=5)
        assert hits and hits[0][0] == "c1"


def test_hybrid_store_round_trip_dense():
    """Hash embedder is a stub — verify the dense path works structurally
    (returns ranked results, top hit is query's own chunk when text matches)."""
    with tempfile.TemporaryDirectory() as d:
        s = HybridStore(f"{d}/h.db")
        # Use the exact same text → hash embedder yields identical vector → cosine=1.0
        chunks = [
            Chunk("c1", "doc1", 1, "alpha beta gamma delta", tokenize("alpha beta gamma delta")),
            Chunk("c2", "doc1", 2, "echo foxtrot golf hotel", tokenize("echo foxtrot golf hotel")),
        ]
        emb = hash_embedder([c.text for c in chunks])
        s.add(chunks, emb)
        # Query exactly matches c1's text → c1 must be top hit
        qv = hash_embedder(["alpha beta gamma delta"])[0]
        hits = s.dense_search(qv, k=2)
        assert hits[0][0] == "c1"
        assert hits[0][1] == pytest.approx(1.0, abs=1e-3)


# ---------------------- LangGraph end-to-end (always run, no Ollama)
def test_graph_full_pipeline_grounds_answer():
    """Run the whole LangGraph with stub LLM + hash embedder. Verifies:
    1. retrieval works
    2. answer is generated
    3. grounding check verifies the citation
    """
    from langgraph.graph import END, StateGraph

    with tempfile.TemporaryDirectory() as d:
        store = HybridStore(f"{d}/h.db")
        chunks = [
            Chunk("c1", "doc1", 5, "the capital of France is Paris", tokenize("the capital of France is Paris")),
            Chunk("c2", "doc1", 6, "the Eiffel Tower is in Paris", tokenize("the Eiffel Tower is in Paris")),
        ]
        emb = hash_embedder([c.text for c in chunks])
        store.add(chunks, emb)

        embedder = lambda texts: hash_embedder(texts)
        llm = stub_llm

        graph = build_graph(store, embedder, llm, reranker=None)
        result = graph.invoke({"query": "What is the capital of France?"})
        assert "answer" in result
        assert result["grounding"]["grounding_rate"] == 1.0
        assert result["grounding"]["n_citations"] >= 1


@pytest.mark.integration
def test_graph_with_real_ollama():
    """Integration: requires `docker compose up` from project #7 + Ollama."""
    from citerag.embedders import ollama_embedder
    import httpx, tempfile

    with tempfile.TemporaryDirectory() as d:
        store = HybridStore(f"{d}/h.db")
        # Add a chunk via real embedder
        emb = ollama_embedder()(["The capital of France is Paris."])
        store.add([Chunk("c1", "doc1", 1, "The capital of France is Paris.",
                         tokenize("The capital of France is Paris."))], emb)

        graph = build_graph(store, ollama_embedder(), stub_llm, reranker=None)
        r = graph.invoke({"query": "capital France"})
        assert r["grounding"]["grounding_rate"] == 1.0