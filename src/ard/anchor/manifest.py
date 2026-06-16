from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ard.anchor.bank import FilterStats, TargetAnswerAnchor


def _filter_stats_payload(
    filter_stats: FilterStats | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if filter_stats is None:
        return None
    if isinstance(filter_stats, FilterStats):
        return filter_stats.to_dict()
    return filter_stats


def build_anchor_manifest(
    target_model: str,
    generation_config: dict[str, Any],
    anchors: list[TargetAnswerAnchor],
    filter_stats: FilterStats | dict[str, Any] | None = None,
    seed: int | None = None,
    requested_count: int | None = None,
    attempted_count: int | None = None,
    input_generation_stats: dict[str, Any] | None = None,
    target_answer_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    domains = Counter(
        anchor.anchor_meta.get("knowledge_top_level", "unknown") for anchor in anchors
    )
    capabilities = Counter(
        anchor.anchor_meta.get("capability", anchor.anchor_meta.get("task_type", "unknown"))
        for anchor in anchors
    )
    tasks = Counter(anchor.anchor_meta.get("task_type", "unknown") for anchor in anchors)
    languages = Counter(anchor.anchor_meta.get("language", "unknown") for anchor in anchors)
    conversation_types = Counter(
        anchor.anchor_meta.get("conversation_type", "unknown") for anchor in anchors
    )
    system_personas = Counter(
        anchor.anchor_meta.get("system_persona", "none") for anchor in anchors
    )
    filter_stats_payload = _filter_stats_payload(filter_stats)
    filter_input_count = (
        int(filter_stats_payload.get("input_count", 0)) if filter_stats_payload else 0
    )
    kept_count = int(filter_stats_payload.get("kept_count", 0)) if filter_stats_payload else 0
    dropped_count = filter_input_count - kept_count
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_generator_model": (
            anchors[0].input_generator_model
            or str(anchors[0].anchor_meta.get("input_generator_model", ""))
            if anchors
            else ""
        ),
        "target_model": target_model,
        "requested_count": requested_count,
        "attempted_count": attempted_count,
        "final_count": len(anchors),
        "filter_rate": dropped_count / filter_input_count if filter_input_count else 0,
        "generation_config": generation_config,
        "seed": seed,
        "counts": {
            "total": len(anchors),
            "by_domain": dict(domains),
            "by_capability": dict(capabilities),
            "by_task_type": dict(tasks),
            "by_language": dict(languages),
            "by_conversation_type": dict(conversation_types),
            "by_system_persona": dict(system_personas),
        },
        "filter_stats": filter_stats_payload,
        "input_generation_stats": input_generation_stats,
        "target_answer_stats": target_answer_stats,
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
