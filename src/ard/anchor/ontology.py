from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ONTOLOGY_PATH = "configs/anchor_ontology.json"

ANCHOR_ONTOLOGY_SECTIONS = {
    "languages",
    "knowledge_domains",
    "capabilities",
    "conversation_types",
    "safety_boundaries",
    "language_features",
}


@dataclass(slots=True)
class Ontology:
    leaves: list[dict[str, Any]]


@dataclass(slots=True)
class AnchorOntology:
    languages: list[str]
    knowledge: Ontology
    language_features: Ontology
    capabilities: Ontology
    conversation_types: Ontology
    safety_boundaries: Ontology


def _extract_leaves(node: Any, path: list[str] | None = None) -> list[dict[str, Any]]:
    path = path or []
    leaves: list[dict[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            leaves.extend(_extract_leaves(value, path + [str(key)]))
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, str):
                full_path = path + [item]
                leaves.append(
                    {
                        "full_path": " -> ".join(full_path),
                        "top_level": path[0] if path else item,
                        "path": path,
                        "leaf": item,
                    }
                )
            else:
                leaves.extend(_extract_leaves(item, path))
    elif isinstance(node, str):
        full_path = path + [node]
        leaves.append(
            {
                "full_path": " -> ".join(full_path),
                "top_level": path[0] if path else node,
                "path": path,
                "leaf": node,
            }
        )
    return leaves


def _section_as_ontology(data: dict[str, Any], section: str) -> Ontology:
    if section not in data:
        raise ValueError(f"anchor ontology is missing required section: {section}")
    ontology = Ontology(leaves=_extract_leaves(data[section]))
    if not ontology.leaves:
        raise ValueError(f"anchor ontology section has no leaves: {section}")
    return ontology


def _section_as_languages(data: dict[str, Any]) -> list[str]:
    raw = data.get("languages")
    if not isinstance(raw, list) or not raw:
        raise ValueError("anchor ontology section must be a non-empty string list: languages")
    languages = [item for item in raw if isinstance(item, str) and item.strip()]
    if len(languages) != len(raw):
        raise ValueError("anchor ontology section must contain only strings: languages")
    return languages


def load_anchor_ontology(path: str | Path = DEFAULT_ONTOLOGY_PATH) -> AnchorOntology:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("anchor ontology root must be a JSON object")

    missing = sorted(ANCHOR_ONTOLOGY_SECTIONS - set(data))
    if missing:
        raise ValueError("anchor ontology is missing required sections: " + ", ".join(missing))

    return AnchorOntology(
        languages=_section_as_languages(data),
        knowledge=_section_as_ontology(data, "knowledge_domains"),
        language_features=_section_as_ontology(data, "language_features"),
        capabilities=_section_as_ontology(data, "capabilities"),
        conversation_types=_section_as_ontology(data, "conversation_types"),
        safety_boundaries=_section_as_ontology(data, "safety_boundaries"),
    )
