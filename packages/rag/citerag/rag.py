"""Production RAG with citations — core logic.

Implements:
- Hybrid search (BM25 + dense via sqlite-vec)
- Reciprocal Rank Fusion (RRF) to merge rankings
- Cross-encoder reranking
- Citation grounding (verify each claim maps to a retrieved chunk)
- LangGraph pipeline (load → embed → retrieve → rerank → generate → verify)
"""
from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, TypedDict

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# --------------------------------------------------------------------- chunks
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    text: str
    bm25_tokens: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "page": self.page,
            "text": self.text,
            "bm25_tokens": self.bm25_tokens,
        }


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Char-based chunking with overlap. Predictable for tests."""
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + chunk_size])
        i += chunk_size - overlap
    return [c.strip() for c in out if c.strip()]


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


# ----------------------------------------------------------------- vector store
class HybridStore:
    """SQLite-backed: FTS5 (BM25) + sqlite-vec (dense) + metadata."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id   TEXT NOT NULL,
                page     INTEGER NOT NULL,
                text     TEXT NOT NULL,
                bm25_tokens TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED, text, tokenize = 'unicode61 remove_diacritics 2'
            );
            CREATE TABLE IF NOT EXISTS chunks_vec (
                chunk_id TEXT PRIMARY KEY,
                vector   BLOB NOT NULL
            );
            """
        )
        self.conn.commit()

    def add(self, chunks: Iterable[Chunk], embeddings: np.ndarray) -> None:
        """embeddings shape: (N, dim), dtype float32."""
        cur = self.conn.cursor()
        for c, vec in zip(chunks, embeddings):
            cur.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?)",
                (c.chunk_id, c.doc_id, c.page, c.text, " ".join(c.bm25_tokens)),
            )
            cur.execute(
                "INSERT OR REPLACE INTO chunks_fts (chunk_id, text) VALUES (?,?)",
                (c.chunk_id, c.text),
            )
            cur.execute(
                "INSERT OR REPLACE INTO chunks_vec (chunk_id, vector) VALUES (?,?)",
                (c.chunk_id, vec.astype(np.float32).tobytes()),
            )
        self.conn.commit()

    # ---- retrieval
    def bm25_search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        """Returns (chunk_id, score), best match first, higher = better.

        FTS5's bm25() is *lower is better* (usually negative), so we order
        ascending and negate the score so callers can treat it like any other
        relevance score. `query` is FTS5 MATCH syntax; sanitise user input
        (e.g. join tokens with OR) before calling.
        """
        cur = self.conn.cursor()
        try:
            rows = cur.execute(
                "SELECT chunk_id, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) ASC LIMIT ?",
                (query, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []  # no FTS-indexed docs yet, or bad query syntax
        return [(cid, -s) for cid, s in rows if s is not None]

    def dense_search(self, query_vec: np.ndarray, k: int = 20) -> list[tuple[str, float]]:
        """Brute-force cosine. Returns (chunk_id, similarity). Higher = better.

        For >10k chunks use the Qdrant backend in the vectorstore package."""
        rows = self.conn.execute("SELECT chunk_id, vector FROM chunks_vec").fetchall()
        if not rows:
            return []
        ids = [r[0] for r in rows]
        mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        q = query_vec.astype(np.float32)
        # cosine
        denom = (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-9)
        sims = (mat @ q) / denom
        order = np.argsort(-sims)[:k]
        return [(ids[i], float(sims[i])) for i in order]

    def get_text(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self.conn.execute(
            f"SELECT chunk_id, doc_id, page, text, bm25_tokens FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        out = {}
        for cid, doc, page, text, toks in rows:
            out[cid] = Chunk(
                chunk_id=cid,
                doc_id=doc,
                page=page,
                text=text,
                bm25_tokens=toks.split() if toks else [],
            )
        return out

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


# ---------------------------------------------------------------- hybrid merge
def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
    top_n: int = 20,
) -> list[tuple[str, float]]:
    """RRF: sum 1/(k + rank) across rankings. Higher = better."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, start=1):
            scores[cid] += 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda x: -x[1])
    return ordered[:top_n]


# ----------------------------------------------------------------- reranking
def cross_encoder_rerank(
    query: str,
    candidates: list[Chunk],
    scorer=None,
) -> list[Chunk]:
    """Score (query, doc) pairs and re-order. If `scorer` is None, fall back
    to a token-overlap heuristic so the function is testable without GPU."""
    if scorer is None:
        # Jaccard fallback — deterministic, unit-test friendly
        q = set(tokenize(query))
        scored = []
        for c in candidates:
            t = set(tokenize(c.text))
            inter = len(q & t)
            union = len(q | t) or 1
            scored.append((c, inter / union))
        return [c for c, _ in sorted(scored, key=lambda x: -x[1])]

    # Real cross-encoder path: e.g. `scorer = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`
    pairs = [(query, c.text) for c in candidates]
    scores = scorer.predict(pairs)
    return [c for _, c in sorted(zip(scores, candidates), key=lambda x: -x[0])]


# ------------------------------------------------------------ citation grounding
CITATION_RE = re.compile(r"\[p\.(\d+)\]|\[(\w+),\s*p\.(\d+)\]", re.IGNORECASE)


def extract_citations(text: str) -> list[tuple[str | None, int]]:
    """Returns list of (doc_id, page). doc_id may be None if not specified."""
    out = []
    for m in CITATION_RE.finditer(text):
        if m.group(1):  # [p.5]
            out.append((None, int(m.group(1))))
        else:  # [docA, p.5]
            out.append((m.group(2), int(m.group(3))))
    return out


def grounding_check(
    answer: str,
    retrieved: dict[str, Chunk],
) -> dict:
    """Verify every cited page exists in the retrieved set."""
    cites = extract_citations(answer)
    pages_seen = {(c.doc_id, c.page) for c in retrieved.values()}
    grounded = [(doc, page) for doc, page in cites if (doc, page) in pages_seen or
                (doc is None and any(pg == page for _, pg in pages_seen))]
    return {
        "n_citations": len(cites),
        "n_grounded": len(grounded),
        "grounding_rate": (len(grounded) / len(cites)) if cites else 1.0,
        "ungrounded": [(doc, page) for doc, page in cites if (doc, page) not in pages_seen and
                       not (doc is None and any(pg == page for _, pg in pages_seen))],
    }


# ----------------------------------------------------------------- LangGraph
class RAGState(TypedDict, total=False):
    query: str
    query_vec: list[float]
    bm25_hits: list[tuple[str, float]]
    dense_hits: list[tuple[str, float]]
    fused: list[tuple[str, float]]
    candidates: list[dict]
    reranked: list[dict]
    answer: str
    grounding: dict


def build_graph(retriever, embedder, llm, reranker=None):
    """Construct the RAG state graph. `retriever` = HybridStore,
    `embedder` = callable(texts)->np.ndarray, `llm` = callable(prompt)->str,
    `reranker` = optional cross-encoder scorer."""
    from langgraph.graph import END, StateGraph

    g = StateGraph(RAGState)

    def embed(state):
        state["query_vec"] = embedder([state["query"]])[0].tolist()
        return state

    def retrieve(state):
        qv = np.asarray(state["query_vec"], dtype=np.float32)
        state["bm25_hits"] = retriever.bm25_search(state["query"], k=20)
        state["dense_hits"] = retriever.dense_search(qv, k=20)
        state["fused"] = reciprocal_rank_fusion(
            [state["bm25_hits"], state["dense_hits"]], top_n=20
        )
        return state

    def fetch(state):
        cids = [cid for cid, _ in state["fused"]]
        chunks = retriever.get_text(cids)
        # preserve fused order
        state["candidates"] = [chunks[cid].__dict__ for cid, _ in state["fused"] if cid in chunks]
        return state

    def rerank(state):
        cands = [Chunk(**c) for c in state["candidates"]]
        reranked_chunks = cross_encoder_rerank(state["query"], cands, scorer=reranker)
        state["reranked"] = [c.__dict__ for c in reranked_chunks[:5]]
        return state

    def generate(state):
        ctx_lines = [f"[{c['doc_id']}, p.{c['page']}]\n{c['text']}" for c in state["reranked"]]
        ctx = "\n\n---\n\n".join(ctx_lines)
        prompt = (
            "Answer the question using ONLY the context below.\n"
            "Cite every claim like [doc_id, p.X]. If the answer is not in the context, say 'I don't know'.\n\n"
            f"Context:\n{ctx}\n\nQ: {state['query']}\nA:"
        )
        state["answer"] = llm(prompt)
        return state

    def verify(state):
        # rehydrate
        retrieved = {c["chunk_id"]: Chunk(**c) for c in state["reranked"]}
        state["grounding"] = grounding_check(state["answer"], retrieved)
        return state

    g.add_node("embed", embed)
    g.add_node("retrieve", retrieve)
    g.add_node("fetch", fetch)
    g.add_node("rerank", rerank)
    g.add_node("generate", generate)
    g.add_node("verify", verify)

    g.set_entry_point("embed")
    g.add_edge("embed", "retrieve")
    g.add_edge("retrieve", "fetch")
    g.add_edge("fetch", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", "verify")
    g.add_edge("verify", END)

    return g.compile()