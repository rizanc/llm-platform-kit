"""Stub RAG system for testing the harness.

Swap for your real pipeline; see evalkit.systems for reference adapters.
"""
from evalkit.harness import CaseResult


def stub_rag_system(case) -> CaseResult:
    """Synthesize a deterministic CaseResult based on the question keywords.

    This lets us drive the harness with a known-quality "system" without
    requiring a real RAG pipeline. Swap for `build_graph(...).invoke({...})`
    in production.
    """
    keywords = case.expected_keywords or case.question.split()

    # Pretend retrieval: chunks include all expected keywords
    chunks = []
    for i, page in enumerate(case.expected_pages or [1, 2]):
        chunks.append({
            "doc_id": (case.expected_doc_ids or ["docA"])[min(i, len(case.expected_doc_ids or ["docA"]) - 1)],
            "page": page,
            "text": " ".join(keywords),  # all keywords appear in context → high faithfulness
        })

    # Pretend citations: include them only when expected_pages is non-empty
    citations = []
    if case.expected_pages:
        for page in case.expected_pages:
            citations.append({
                "doc_id": (case.expected_doc_ids or ["docA"])[0],
                "page": page,
            })

    # Pretend answer: just the keywords, with no filler
    if case.expected_answer:
        answer = case.expected_answer
    elif keywords:
        answer = " ".join(keywords)
    else:
        answer = case.question

    return CaseResult(
        case_id=case.case_id,
        question=case.question,
        answer=answer,
        citations=citations,
        retrieved_chunks=chunks,
        metrics={},
        passed=False,
    )