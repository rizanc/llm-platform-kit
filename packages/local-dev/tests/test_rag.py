"""Smoke tests — unit tests run anywhere; integration requires docker compose up."""
import pytest

from localrag.ingest import chunk_text


def test_chunk_text_overlap():
    text = "abcdefghij" * 100
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 3
    assert all(len(c) <= 500 for c in chunks)


def test_chunk_text_strips_empty():
    assert chunk_text("\n\n\n\n\n") == []


@pytest.mark.integration
def test_rag_retrieve_empty():
    """Empty-table retrieval returns []. Needs LanceDB only."""
    import tempfile
    from localrag.rag import RAG
    with tempfile.TemporaryDirectory() as d:
        rag = RAG(uri=d, ollama_host="http://localhost:1")
        assert rag.retrieve("anything") == []
        assert rag.stats() == {"chunks": 0, "table": "chunks"}


@pytest.mark.integration
def test_health_endpoint():
    """End-to-end — needs full docker compose stack."""
    import httpx
    r = httpx.get("http://localhost:8000/health", timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"