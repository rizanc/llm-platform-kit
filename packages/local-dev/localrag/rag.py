"""RAG core: LanceDB storage + Ollama embeddings + Ollama generation."""
import logging
import lancedb
import ollama
import pyarrow as pa
from typing import Any

log = logging.getLogger(__name__)

SCHEMA = pa.schema([
    ("vector", pa.list_(pa.float32(), 768)),  # nomic-embed-text dim
    ("text", pa.string()),
    ("page", pa.int32()),
    ("source", pa.string()),
])


class RAG:
    def __init__(self, uri: str, ollama_host: str, embed_model: str = "nomic-embed-text", llm_model: str = "llama3.1:8b"):
        self.db = lancedb.connect(uri)
        self.table_name = "chunks"
        self.ollama_host = ollama_host
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.table = self.db.open_table(self.table_name) if self.table_name in self.db.list_tables() else None

    def _embed(self, texts: list[str]) -> list[list[float]]:
        client = ollama.Client(host=self.ollama_host)
        out = []
        # batch in groups of 16 to avoid Ollama OOM on tiny models
        for i in range(0, len(texts), 16):
            batch = texts[i : i + 16]
            resp = client.embed(model=self.embed_model, input=batch)
            out.extend(e for e in resp["embeddings"])
        return out

    def add(self, rows: list[dict[str, Any]]) -> None:
        vecs = self._embed([r["text"] for r in rows])
        data = [{"vector": v, **r} for v, r in zip(vecs, rows)]
        if self.table is None:
            self.table = self.db.create_table(self.table_name, data, mode="overwrite")
        else:
            self.table.add(data)

    def retrieve(self, q: str, k: int = 4) -> list[dict]:
        if self.table is None or self.table.count_rows() == 0:
            return []
        qvec = self._embed([q])[0]
        hits = self.table.search(qvec).limit(k).to_list()
        return [{"text": h["text"], "page": h["page"], "source": h["source"], "_distance": h["_distance"]} for h in hits]

    def generate(self, q: str, hits: list[dict]) -> str:
        ctx = "\n\n---\n\n".join(f"[page {h['page']}]\n{h['text']}" for h in hits)
        prompt = f"Answer using ONLY the context. Cite pages like [p.X].\n\nContext:\n{ctx}\n\nQ: {q}\nA:"
        client = ollama.Client(host=self.ollama_host)
        resp = client.generate(model=self.llm_model, prompt=prompt, stream=False)
        return resp["response"]

    def stats(self) -> dict:
        return {"chunks": 0 if self.table is None else self.table.count_rows(), "table": self.table_name}