"""Skip integration tests unless RUN_INTEGRATION=1.

Unit tests never touch the network, a GPU, or an API key. Integration tests
need a live service (Ollama, Qdrant, docker compose, or ANTHROPIC_API_KEY)
and are opt-in so CI stays deterministic.
"""
import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="integration test; set RUN_INTEGRATION=1 and start the service it needs")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
