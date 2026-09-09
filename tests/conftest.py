# conftest.py — Shared pytest fixtures and configuration.
# Responsibility: provide reusable test fixtures (API clients, configs,
# sample data) for all test modules in the test suite.

from __future__ import annotations

import json

import pytest


@pytest.fixture
def ontology() -> dict:
    """Load shared anchor ontology for tests."""
    with open("data/anchor_ontology.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def embeddings_data() -> dict:
    """Load shared embedding data for tests."""
    with open("data/anchor_ontology_embeddings.json", encoding="utf-8") as f:
        return json.load(f)