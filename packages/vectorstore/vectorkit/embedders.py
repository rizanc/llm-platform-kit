"""Embedding helpers — hash embedder (deterministic, zero-dep) + ollama (real)."""
import numpy as np


def hash_embedder(texts: list[str], dim: int = 768) -> np.ndarray:
    """Same as project #1 — keeps tests reproducible without model downloads."""
    import hashlib
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
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
    """Real embedder. Lazy import keeps tests light."""
    import ollama
    def _embed(texts):
        client = ollama.Client(host=host)
        out = []
        for i in range(0, len(texts), 16):
            resp = client.embed(model=model, input=texts[i:i + 16])
            out.extend(resp["embeddings"])
        return np.asarray(out, dtype=np.float32)
    return _embed