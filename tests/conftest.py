# conftest.py — Shared pytest fixtures and configuration.
# Responsibility: provide reusable test fixtures (ontology, configs, sample
# data) for all test modules in the test suite.

from __future__ import annotations

import pytest

from ard.core.ontology import OntologyV4, load_ontology_v4

#: The ontology the production pipeline reads (see ``configs/config.toml``).
ONTOLOGY_PATH = "ontology/anchor_ontology.v4.json"


@pytest.fixture(scope="module")
def ontology() -> OntologyV4:
    """Load the shared v4 ontology for tests."""
    return load_ontology_v4(ONTOLOGY_PATH)
