"""Extractive reference system: BM25 retrieval + sentence extraction, no LLM.

Retrieval uses `citerag.HybridStore`'s SQLite FTS5 BM25 index. The dense
side of the store is deliberately not used here: without a real embedding
model the hash embedder is a test double, and fusing its rankings in adds
noise (measured: recall@3 on the bundled set dropped from 1.00 to 0.80).
Pass `embedder=` to enable hybrid retrieval with a real model.

The "generation" step picks the sentence, across the retrieved chunks, with
the most question-term overlap and cites the page it came from.

Why it exists: it is a deterministic, honest baseline. Retrieval metrics
(context_recall, context_precision, citation_accuracy, grounding_rate)
measure something real here. faithfulness is trivially high because the
answer is copied from the context, which is exactly what you would expect
an extractive system to score. Replace it with your pipeline via
`evalkit run --system your.module:build`.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Callable

from evalkit.harness import CaseResult, GoldenCase

_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z0-9]+")
_STOP = {"the", "a", "an", "of", "is", "are", "was", "were", "what", "which", "who", "how", "when", "where", "why",
         "does", "do", "did", "in", "on", "to", "and", "it", "its", "for", "from", "by", "that", "this", "with", "as"}


def _terms(text: str) -> set[str]:
    return {t.lower() for t in _WORD.findall(text)} - _STOP


def build(corpus: dict, k: int = 3, db_path: str | Path | None = None, embedder: Callable | None = None) -> Callable[[GoldenCase], CaseResult]:
    from citerag.embedders import hash_embedder
    from citerag.rag import Chunk, HybridStore, reciprocal_rank_fusion, tokenize

    embed = embedder or hash_embedder

    db_path = db_path or Path(tempfile.mkdtemp(prefix="evalkit-")) / "corpus.db"
    store = HybridStore(db_path)
    doc_id = corpus["doc_id"]
    chunks = [Chunk(f"{doc_id}:p{p['page']}", doc_id, p["page"], p["text"], tokenize(p["text"])) for p in corpus["pages"]]
    store.add(chunks, embed([c.text for c in chunks]))

    def answer(case: GoldenCase) -> CaseResult:
        terms = _terms(case.question)
        fts_query = " OR ".join(sorted(terms)) or case.question
        bm25 = store.bm25_search(fts_query, k=10)
        if embedder is not None:
            dense = store.dense_search(embed([case.question])[0], k=10)
            ranked = reciprocal_rank_fusion([bm25, dense], top_n=k)
        else:
            ranked = bm25[:k]
        ids = [cid for cid, _ in ranked]
        by_id = store.get_text(ids)
        top = [by_id[c] for c in ids if c in by_id]
        retrieved = [{"doc_id": c.doc_id, "page": c.page, "text": c.text} for c in top]
        if not top:
            return CaseResult(case.case_id, case.question, "No relevant passage found.", [], retrieved)
        best_sentence, best_chunk, best_score = "", top[0], -1
        for c in top:
            for sent in _SENT.split(c.text):
                score = len(terms & _terms(sent))
                if score > best_score:
                    best_sentence, best_chunk, best_score = sent.strip(), c, score
        text = f"{best_sentence} [{best_chunk.doc_id}, p.{best_chunk.page}]"
        return CaseResult(case.case_id, case.question, text, [{"doc_id": best_chunk.doc_id, "page": best_chunk.page}], retrieved)

    return answer
