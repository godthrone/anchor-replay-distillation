#!/usr/bin/env python3
"""Generate ontology embeddings for ARD anchor replay distillation.

Calls the embedding API to produce 1024-dim embeddings for:
  - 18 knowledge domain categories (Layer 1)
  - 20 AI capabilities (Layer 2)
  - 4 languages (Layer 2)
  - 7 conversation types (Layer 2)

Usage:
    python scripts/generate_ontology_embeddings.py \
        --ontology ontology/anchor_ontology.json \
        --output ontology/anchor_ontology_embeddings.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ontology embeddings for ARD anchor replay distillation."
    )
    parser.add_argument(
        "--ontology",
        required=True,
        type=Path,
        help="Path to anchor_ontology.json (e.g. ontology/anchor_ontology.json)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for the embeddings JSON file.",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="Embedding API URL (OpenAI-compatible). Required.",
    )
    parser.add_argument(
        "--model",
        default="embedding",
        help="Model name to pass to the API.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of texts per API call (default: 10).",
    )
    return parser.parse_args()


def load_ontology(path: Path) -> dict[str, Any]:
    """Load and return the ontology JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def flatten_leaf_topics(domain: dict[str, list[str]]) -> list[str]:
    """Flatten all leaf topics from all sub-categories of a knowledge domain."""
    topics: list[str] = []
    for sub_cat_values in domain.values():
        topics.extend(sub_cat_values)
    return topics


def build_texts(ontology: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Build the text descriptions for every item that needs an embedding.

    Returns a dict with sections: knowledge_domains, capabilities, languages,
    conversation_types. Each value is a {key: text} mapping.
    """
    result: dict[str, dict[str, str]] = {
        "knowledge_domains": {},
        "capabilities": {},
        "languages": {},
        "conversation_types": {},
    }

    # 18 knowledge domain categories
    for domain_name, domain_data in ontology["knowledge_domains"].items():
        leaf_topics = flatten_leaf_topics(domain_data)
        topics_str = ", ".join(leaf_topics[:12])  # cap to avoid overly long texts
        if len(leaf_topics) > 12:
            topics_str += f", ... ({len(leaf_topics)} topics total)"
        text = (
            f"Knowledge domain in ARD ontology: {domain_name}. "
            f"This domain covers topics like: {topics_str}. "
            f"Used for anchor replay distillation to ensure coverage diversity."
        )
        result["knowledge_domains"][domain_name] = text

    # 20 capabilities (leaf values under each capability group)
    for group_name, leaf_list in ontology["capabilities"].items():
        for cap_name in leaf_list:
            text = (
                f"AI capability: {cap_name}. "
                f"Belongs to capability group: {group_name}. "
                f"Used in ARD anchor replay distillation."
            )
            result["capabilities"][cap_name] = text

    # 4 languages
    for lang in ontology["languages"]:
        text = (
            f"Language: {lang}. "
            f"Used in ARD anchor replay distillation for multilingual coverage."
        )
        result["languages"][lang] = text

    # 7 conversation types (leaf values under each conversation type group)
    for ct_group, leaf_list in ontology["conversation_types"].items():
        for ct_name in leaf_list:
            text = (
                f"Conversation type: {ct_name}. "
                f"Belongs to conversation group: {ct_group}. "
                f"Used in ARD anchor replay distillation for interaction diversity."
            )
            result["conversation_types"][ct_name] = text

    return result


def call_embedding_api(
    api_url: str,
    model: str,
    texts: list[str],
    max_retries: int = 3,
) -> list[list[float]]:
    """Call the OpenAI-compatible embedding API and return embeddings.

    Raises RuntimeError on failure after all retries.
    """
    payload = {"model": model, "input": texts}
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = httpx.post(api_url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            # Sort by index to preserve order
            embeddings = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in embeddings]
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  API call failed (attempt {attempt}/{max_retries}), "
                      f"retrying in {wait}s: {e}", file=sys.stderr)
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Embedding API call failed after {max_retries} attempts: {e}"
                ) from e

    # Should not reach here
    raise RuntimeError(f"Embedding API call failed: {last_error}")


def batch_call_api(
    api_url: str,
    model: str,
    items: dict[str, str],
    batch_size: int,
) -> dict[str, list[float]]:
    """Call the embedding API in batches and return {key: embedding}."""
    keys = list(items.keys())
    texts = list(items.values())
    all_embeddings: dict[str, list[float]] = {}

    total = len(keys)
    for i in range(0, total, batch_size):
        batch_keys = keys[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"  Batch {batch_num}/{total_batches}: "
              f"embedding {len(batch_keys)} items ({batch_keys[0]} ... {batch_keys[-1]})")

        embeddings = call_embedding_api(api_url, model, batch_texts)
        for key, emb in zip(batch_keys, embeddings):
            all_embeddings[key] = emb

    return all_embeddings


def main() -> None:
    args = parse_args()

    if args.api_url is None:
        print(
            "ERROR: --api-url is required. Please provide the embedding API URL.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading ontology from {args.ontology}")
    ontology = load_ontology(args.ontology)

    print("Building text descriptions...")
    text_items = build_texts(ontology)

    # Count items
    n_domains = len(text_items["knowledge_domains"])
    n_caps = len(text_items["capabilities"])
    n_langs = len(text_items["languages"])
    n_ctypes = len(text_items["conversation_types"])
    total = n_domains + n_caps + n_langs + n_ctypes
    print(f"  knowledge_domains: {n_domains}")
    print(f"  capabilities:      {n_caps}")
    print(f"  languages:         {n_langs}")
    print(f"  conversation_types:{n_ctypes}")
    print(f"  total:             {total}")

    if total != 49:
        print(f"WARNING: expected 49 items, got {total}", file=sys.stderr)

    # Generate embeddings for each section
    print(f"\nCalling embedding API at {args.api_url} (model={args.model})")

    embeddings: dict[str, dict[str, list[float]]] = {}
    for section_name, section_items in text_items.items():
        if not section_items:
            embeddings[section_name] = {}
            continue
        print(f"\n[Section: {section_name}] ({len(section_items)} items)")
        embeddings[section_name] = batch_call_api(
            args.api_url, args.model, section_items, args.batch_size
        )

    # Verify dimensions
    for section_name, section_embs in embeddings.items():
        for key, emb in section_embs.items():
            if len(emb) != 1024:
                raise RuntimeError(
                    f"Embedding dimension mismatch for {section_name}/{key}: "
                    f"expected 1024, got {len(emb)}"
                )

    # Compute ontology SHA256
    ontology_sha256 = compute_sha256(args.ontology)

    # Build output
    generated_at = datetime.now(timezone.utc).isoformat()
    output = {
        "model": args.model,
        "embedding_dimension": 1024,
        "ontology_sha256": ontology_sha256,
        "generated_at": generated_at,
        "items": embeddings,
    }

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    file_size = args.output.stat().st_size
    print(f"\n✓ Embeddings written to {args.output}")
    print(f"  File size: {file_size:,} bytes")
    print(f"  Ontology SHA256: {ontology_sha256}")
    print(f"  Generated at: {generated_at}")
    print(f"  Items: {n_domains} domains + {n_caps} capabilities "
          f"+ {n_langs} languages + {n_ctypes} conversation types = {total}")


if __name__ == "__main__":
    main()