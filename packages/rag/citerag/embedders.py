"""Embedding helpers.

In production: swap for any 768-dim model (nomic-embed-text, BGE, OpenAI ada-002 dim=1536).
The unit tests use a deterministic hash-based embedder so they're reproducible
and need no model download.
"""
import hashlib
import numpy as np


def hash_embedder(texts: list[str], dim: int = 768) -> np.ndarray:
    """Deterministic, dependency-free embedder for tests.

    Same input → same vector. Different inputs → orthogonal-ish vectors.
    Not useful for real retrieval — proves the pipeline works."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        # 12 hash buckets × 64 dims = 768
        h = hashlib.sha256(t.encode()).digest()
        for j in range(12):
            chunk = int.from_bytes(h[j * 2 : j * 2 + 2], "big")
            start = (j * 64) % dim
            for k in range(64):
                if (chunk >> (k % 16)) & 1:
                    out[i, (start + k) % dim] += 1.0
        out[i] -= out[i].mean()
        n = np.linalg.norm(out[i]) + 1e-9
        out[i] /= n
    return out


def ollama_embedder(host: str = "http://localhost:11434", model: str = "nomic-embed-text"):
    """Real embedder via Ollama. Lazy-imported to avoid hard dep in tests."""
    import ollama

    def _embed(texts: list[str]) -> np.ndarray:
        client = ollama.Client(host=host)
        out = []
        for i in range(0, len(texts), 16):
            batch = texts[i : i + 16]
            resp = client.embed(model=model, input=batch)
            out.extend(resp["embeddings"])
        return np.asarray(out, dtype=np.float32)

    return _embed