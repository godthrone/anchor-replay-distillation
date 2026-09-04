"""Anchor bank storage — unified format aligned with graspo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ard.core.types import GeneratedAnchor


def anchor_to_dict(anchor: GeneratedAnchor) -> dict[str, Any]:
    """Convert a :class:`GeneratedAnchor` to a dict in the unified JSONL format.

    Output format (aligned with graspo):
    ```json
    {
      "id": "anchor_<sha256_hex16>",
      "source": "ard",
      "messages": [...],
      "targets": [{
        "id": "primary",
        "output": {
          "content": "<target_answer>",
          "logprobs": {"token_ids": [...], "log_probs": [...]}
        }
      }],
      "anchor_meta": {...},
      "teacher_id": "..."
    }
    ```
    """
    return {
        "id": anchor.id,
        "source": "ard",
        "messages": anchor.messages,
        "targets": [
            {
                "id": "primary",
                "output": {
                    "content": anchor.target_answer,
                    "logprobs": anchor.logprobs or {"token_ids": [], "log_probs": []},
                },
            }
        ],
        "anchor_meta": anchor.anchor_meta,
        "teacher_id": anchor.target_model,
    }


def write_anchor_bank(anchors: list[GeneratedAnchor], output_path: str | Path) -> None:
    """Write anchors to a JSONL file in the unified format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for a in anchors:
            f.write(json.dumps(anchor_to_dict(a), ensure_ascii=False) + "\n")


def append_anchor(anchor: GeneratedAnchor, path: Path) -> None:
    """Append a single anchor to a JSONL file (thread-safe).

    Uses ``O_APPEND`` (via ``open(..., "a")``) which POSIX guarantees
    is atomic for writes up to ``PIPE_BUF`` bytes.  ``flush()`` ensures
    the line is immediately written to disk so that an interrupted run
    can resume from the last committed anchor.
    """
    line = json.dumps(anchor_to_dict(anchor), ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()


def count_existing_anchors(path: Path) -> int:
    """Count existing anchors in a JSONL file.

    Returns 0 if the file does not exist.
    """
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def read_anchor_bank(path: str | Path) -> list[dict[str, Any]]:
    """Read anchors from a JSONL file.

    Returns:
        List of parsed record dicts.
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_manifest(
    anchors: list[GeneratedAnchor],
    output_dir: str | Path,
    config_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest dict summarizing the anchor bank."""
    domains: dict[str, int] = {}
    languages: dict[str, int] = {}
    capabilities: dict[str, int] = {}
    for a in anchors:
        m = a.anchor_meta
        d = m.get("knowledge_domain", m.get("visual_domain", "unknown"))
        domains[d] = domains.get(d, 0) + 1
        lang = m.get("language", "unknown")
        languages[lang] = languages.get(lang, 0) + 1
        c = m.get("capability", m.get("question_type", "unknown"))
        capabilities[c] = capabilities.get(c, 0) + 1
    manifest: dict[str, Any] = {
        "total_anchors": len(anchors),
        "domains": domains,
        "languages": languages,
        "capabilities": capabilities,
        "output_dir": str(output_dir),
    }
    if config_info:
        manifest["config"] = config_info
    return manifest


def build_manifest_from_records(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    config_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest dict from raw JSONL records (for checkpoint/resume).

    This is the record-based counterpart of :func:`build_manifest` — it
    reads ``anchor_meta`` directly from the parsed dicts rather than
    from :class:`GeneratedAnchor` objects, so the full anchor bank can be
    summarised without re-materialising every ``GeneratedAnchor``.
    """
    domains: dict[str, int] = {}
    languages: dict[str, int] = {}
    capabilities: dict[str, int] = {}
    for r in records:
        m = r.get("anchor_meta", {})
        d = m.get("knowledge_domain", m.get("visual_domain", "unknown"))
        domains[d] = domains.get(d, 0) + 1
        lang = m.get("language", "unknown")
        languages[lang] = languages.get(lang, 0) + 1
        c = m.get("capability", m.get("question_type", "unknown"))
        capabilities[c] = capabilities.get(c, 0) + 1
    manifest: dict[str, Any] = {
        "total_anchors": len(records),
        "domains": domains,
        "languages": languages,
        "capabilities": capabilities,
        "output_dir": str(output_dir),
    }
    if config_info:
        manifest["config"] = config_info
    return manifest


def write_manifest(manifest: dict[str, Any], output_path: str | Path) -> None:
    """Write a manifest dict to a JSON file."""
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
