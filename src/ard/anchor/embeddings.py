from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ard.anchor.ontology import _extract_leaves


DEFAULT_ONTOLOGY_EMBEDDINGS_PATH = "configs/anchor_ontology_embeddings.json"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
BOOTSTRAP_EMBEDDING_MODEL = "research-neutral-lexical-bootstrap-v1"

EMBEDDABLE_SECTIONS = (
    "languages",
    "knowledge_domains",
    "capabilities",
    "conversation_types",
    "language_features",
)


@dataclass(slots=True)
class OntologyEmbeddingItem:
    section: str
    path: list[str]
    leaf: str
    text: str
    embedding: list[float]

    @property
    def full_path(self) -> str:
        return " -> ".join([*self.path, self.leaf])


@dataclass(slots=True)
class OntologyEmbeddings:
    ontology_sha256: str
    embedding_model: str
    embedding_dimension: int
    distance: str
    items: list[OntologyEmbeddingItem]

    def items_for_section(self, section: str) -> list[OntologyEmbeddingItem]:
        return [item for item in self.items if item.section == section]


EmbedFn = Callable[[list[str]], list[list[float]]]


def ontology_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ontology_records(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("anchor ontology root must be a JSON object")
    records: list[dict[str, Any]] = []
    for section in EMBEDDABLE_SECTIONS:
        if section not in data:
            raise ValueError(f"anchor ontology is missing required section: {section}")
        if section == "languages":
            raw_languages = data[section]
            if not isinstance(raw_languages, list):
                raise ValueError("languages must be a list")
            for language in raw_languages:
                records.append(
                    {
                        "section": section,
                        "path": [],
                        "leaf": str(language),
                    }
                )
            continue
        for leaf in _extract_leaves(data[section]):
            records.append(
                {
                    "section": section,
                    "path": list(leaf["path"]),
                    "leaf": str(leaf["leaf"]),
                }
            )
    return records


def ontology_embedding_text(record: dict[str, Any]) -> str:
    path = " -> ".join([*record["path"], record["leaf"]])
    return (
        "ARD ontology concept\n"
        f"section: {record['section']}\n"
        f"path: {path}\n"
        f"leaf: {record['leaf']}\n"
        "purpose: research-neutral anchor replay distillation coverage"
    )


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def hash_embed_texts(texts: list[str], dimensions: int = 64) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * dimensions
        for token in text.lower().replace("\n", " ").split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, min(len(digest), dimensions), 2):
                index = digest[offset] % dimensions
                sign = 1.0 if digest[offset + 1] % 2 == 0 else -1.0
                vector[index] += sign
        vectors.append(_normalize(vector))
    return vectors


def build_ontology_embeddings(
    ontology_path: str | Path,
    embed_fn: EmbedFn,
    embedding_model: str,
    distance: str = "cosine",
) -> OntologyEmbeddings:
    records = _ontology_records(ontology_path)
    texts = [ontology_embedding_text(record) for record in records]
    vectors = embed_fn(texts)
    if len(vectors) != len(records):
        raise ValueError("embedding backend returned the wrong number of vectors")
    dimensions = len(vectors[0]) if vectors else 0
    items = [
        OntologyEmbeddingItem(
            section=str(record["section"]),
            path=list(record["path"]),
            leaf=str(record["leaf"]),
            text=text,
            embedding=[float(value) for value in vector],
        )
        for record, text, vector in zip(records, texts, vectors)
    ]
    return OntologyEmbeddings(
        ontology_sha256=ontology_sha256(ontology_path),
        embedding_model=embedding_model,
        embedding_dimension=dimensions,
        distance=distance,
        items=items,
    )


def load_ontology_embeddings(
    path: str | Path,
    ontology_path: str | Path | None = None,
    validate_hash: bool = True,
) -> OntologyEmbeddings:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ontology embeddings root must be a JSON object")
    if ontology_path is not None and validate_hash:
        expected_hash = ontology_sha256(ontology_path)
        if payload.get("ontology_sha256") != expected_hash:
            raise ValueError(
                "ontology embeddings are stale for this ontology. "
                "Run `ard ontology-embed --ontology <path> --output <embeddings>` "
                "or use `--sampling-strategy balanced`."
            )
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("ontology embeddings must contain a non-empty items list")
    items = [
        OntologyEmbeddingItem(
            section=str(item["section"]),
            path=list(item.get("path", [])),
            leaf=str(item["leaf"]),
            text=str(item.get("text", "")),
            embedding=[float(value) for value in item["embedding"]],
        )
        for item in raw_items
    ]
    dimensions = int(payload.get("embedding_dimension", len(items[0].embedding)))
    if any(len(item.embedding) != dimensions for item in items):
        raise ValueError("ontology embeddings contain vectors with inconsistent dimensions")
    return OntologyEmbeddings(
        ontology_sha256=str(payload.get("ontology_sha256", "")),
        embedding_model=str(payload.get("embedding_model", "")),
        embedding_dimension=dimensions,
        distance=str(payload.get("distance", "cosine")),
        items=items,
    )


def write_ontology_embeddings(embeddings: OntologyEmbeddings, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ontology_sha256": embeddings.ontology_sha256,
        "embedding_model": embeddings.embedding_model,
        "embedding_dimension": embeddings.embedding_dimension,
        "distance": embeddings.distance,
        "items": [
            {
                "section": item.section,
                "path": item.path,
                "leaf": item.leaf,
                "text": item.text,
                "embedding": item.embedding,
            }
            for item in embeddings.items
        ],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
