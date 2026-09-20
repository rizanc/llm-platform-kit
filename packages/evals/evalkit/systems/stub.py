"""Deterministic stub that always answers from the expected evidence. Useful for testing the harness itself."""
from evalkit.system_stub import stub_rag_system


def build(corpus=None):
    return stub_rag_system
