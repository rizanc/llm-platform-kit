"""Tiny stub LLM for tests — echoes a citation-rich answer based on chunks.

Production swap: any chat model (ollama, openai, anthropic).
"""
from .rag import Chunk


def stub_llm(prompt: str) -> str:
    """Extract the first doc_id + page from the prompt and emit one citation.
    Deterministic so tests are stable."""
    import re
    m = re.search(r"\[(\w+),\s*p\.(\d+)\]", prompt)
    if not m:
        return "I don't know."
    doc, page = m.group(1), m.group(2)
    return f"Based on the document, the answer is X [{doc}, p.{page}]."