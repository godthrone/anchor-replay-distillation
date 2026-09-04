"""Ontology loading for ARD anchor generation.

Loads anchor_ontology.json and provides typed access to its sections.
No external dependencies beyond stdlib json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_ontology(path: str | Path) -> dict[str, Any]:
    """Load anchor ontology from a JSON file.

    Args:
        path: Path to anchor_ontology.json.

    Returns:
        Parsed ontology dict with keys: languages, knowledge_domains,
        capabilities, conversation_types, language_features, visual_domains.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)
