"""Anchor bank storage — unified format aligned with graspo."""

import json
from pathlib import Path
from typing import Any

from ard.core.types import Anchor


def write_anchor_bank(anchors: list[Anchor], output_path: str | Path) -> None:
    """Write anchors to a JSONL file in the unified format.

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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for a in anchors:
            record = {
                "id": a.id,
                "source": "ard",
                "messages": a.messages,
                "targets": [
                    {
                        "id": "primary",
                        "output": {
                            "content": a.target_answer,
                            "logprobs": a.logprobs or {"token_ids": [], "log_probs": []},
                        },
                    }
                ],
                "anchor_meta": a.anchor_meta,
                "teacher_id": a.target_model,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    anchors: list[Anchor],
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
        l = m.get("language", "unknown")
        languages[l] = languages.get(l, 0) + 1
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


def write_manifest(manifest: dict[str, Any], output_path: str | Path) -> None:
    """Write a manifest dict to a JSON file."""
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)