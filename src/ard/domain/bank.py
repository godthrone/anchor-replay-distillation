"""Anchor bank storage — unified format aligned with graspo.

This module owns the *exit boundary* of anchor production.  Everything that
reaches the anchor bank has passed three gates here:

1. **Message shape** (:func:`ard.domain.anchor_shape.message_shape_error`) —
   a conversation must start with ``user``, end with ``user`` and alternate
   roles.  ``AnchorSpec`` enforces that on the way in; enforcing it here means
   a malformed anchor can never be published even if a turn generator
   regresses (that is exactly how ``UAUAU``-shaped anchors once reached a
   bank: entry gate without exit gate).
2. **Data source vocabulary** (:func:`data_source_error`) — ``data_source`` is
   the OPD routing key, and a value outside
   :class:`~ard.core.types.DataSource` is a record no consumer routes.  It used
   to be an unconstrained string, so a typo was persisted silently.
3. **Id uniqueness** — one record per anchor id.  ``anchor id`` is derived
   from 5-dimensional metadata (see :func:`ard.core.sampler.generate_anchor_id`),
   so two specs can legitimately carry the same id; writing both inflates the
   bank while the resume logic in :mod:`ard.pipeline` counts lines, not ids.

All gates reject loudly: the caller receives an :class:`AppendOutcome` and
logs it (§2.3 边界校验即防呆, §3.2 透明退路).
"""

from __future__ import annotations

import json
import logging
import threading
from enum import Enum
from pathlib import Path
from typing import Any

from ard.core.types import DataSource, GeneratedAnchor
from ard.domain.anchor_shape import message_shape_error

logger = logging.getLogger(__name__)

# Output-schema version of every record persisted through this module (v3.0.0).
# Single source of truth (§1.4): the anchor format's version is written here,
# once, as a top-level ``schema_version`` field on each record.
SCHEMA_VERSION = "3.0.0"


class AppendOutcome(Enum):
    """What happened when an anchor was offered to the bank."""

    APPENDED = "appended"
    """The anchor was written to the bank."""

    DUPLICATE_SKIPPED = "duplicate_skipped"
    """An anchor with this id is already in the bank — nothing was written."""

    INVALID_SHAPE_SKIPPED = "invalid_shape_skipped"
    """The anchor's messages violated the conversation-shape contract."""

    INVALID_DATA_SOURCE_SKIPPED = "invalid_data_source_skipped"
    """The anchor's ``data_source`` was outside the controlled vocabulary."""


# Ids already present per bank file.  The value is ``(fingerprint, ids)`` where
# the fingerprint is the file's ``(st_mtime_ns, st_size)`` at read time, so an
# externally truncated/rewritten bank invalidates the cache instead of silently
# suppressing legitimate anchors.
_seen_ids_cache: dict[str, tuple[tuple[int, int] | None, set[str]]] = {}
_seen_ids_lock = threading.Lock()


def _bank_fingerprint(path: Path) -> tuple[int, int] | None:
    """Return ``(mtime_ns, size)`` for *path*, or ``None`` if it is absent."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _read_ids(path: Path) -> set[str]:
    """Collect the anchor ids already stored in *path* (tolerating bad lines)."""
    ids: set[str] = set()
    if not path.exists():
        return ids
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Ignoring unreadable line %d while scanning ids in %s",
                    line_no, path,
                )
                continue
            record_id = record.get("id")
            if isinstance(record_id, str):
                ids.add(record_id)
    return ids


def _known_ids(path: Path) -> set[str]:
    """Return (and cache) the ids present in *path*.

    Callers must hold :data:`_seen_ids_lock`.  The cache is keyed by path and
    invalidated whenever the file's mtime/size changed, which happens exactly
    when a row was appended (by us) or the bank was rewritten (by someone
    else / a new run).
    """
    key = str(path)
    fingerprint = _bank_fingerprint(path)
    cached = _seen_ids_cache.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    ids = _read_ids(path)
    _seen_ids_cache[key] = (fingerprint, ids)
    return ids


def data_source_error(anchor: GeneratedAnchor) -> str | None:
    """Validate an anchor's ``data_source`` against the controlled vocabulary.

    ``data_source`` is the OPD routing key: the training side routes records by
    it, so a value outside the vocabulary is not "an unusual label", it is a
    record no consumer will ever pick up.  The contract is enforced by the
    :class:`~ard.core.types.DataSource` enum at construction; this gate exists
    so the **exit boundary** owns it too (§2.3 边界校验即防呆) — a record that
    is written to the bank must satisfy both gates, whoever built it.

    Args:
        anchor: The anchor about to be persisted.

    Returns:
        ``None`` when the value is in the vocabulary, otherwise a
        human-readable description of the violation.
    """
    if not isinstance(anchor.data_source, DataSource):
        return (
            f"data_source {anchor.data_source!r} is not a DataSource "
            f"(allowed: {[member.value for member in DataSource]})"
        )
    return None


def append_anchor(anchor: GeneratedAnchor, path: Path) -> AppendOutcome:
    """Validate and append a single anchor to a JSONL file (thread-safe).

    The shape check, the ``data_source`` check and the id-uniqueness check
    happen under one lock together with the write, so two threads racing on the
    same id cannot both win.

    Uses ``O_APPEND`` (via ``open(..., "a")``) which POSIX guarantees
    is atomic for writes up to ``PIPE_BUF`` bytes.  ``flush()`` ensures
    the line is immediately written to disk so that an interrupted run
    can resume from the last committed anchor.

    Args:
        anchor: The anchor to persist.
        path: Target JSONL file (created if absent; parent dir must exist).

    Returns:
        :class:`AppendOutcome` describing whether anything was written.
    """
    shape_error = message_shape_error(anchor.messages)
    if shape_error is not None:
        # Rejected *before* touching the file: a structurally broken
        # conversation is not distillable data and must never be published.
        logger.warning(
            "Anchor %s rejected by the output shape gate (%s) — not written to %s",
            anchor.id, shape_error, path,
        )
        return AppendOutcome.INVALID_SHAPE_SKIPPED

    routing_error = data_source_error(anchor)
    if routing_error is not None:
        # Same reasoning as the shape gate: an unrouteable record must not be
        # published, and it must be refused *before* the file is opened.
        logger.warning(
            "Anchor %s rejected by the data_source gate (%s) — not written to %s",
            anchor.id, routing_error, path,
        )
        return AppendOutcome.INVALID_DATA_SOURCE_SKIPPED

    line = json.dumps(anchor_to_dict(anchor), ensure_ascii=False) + "\n"
    with _seen_ids_lock:
        known = _known_ids(path)
        if anchor.id in known:
            logger.warning(
                "Anchor %s already present in %s — duplicate not written",
                anchor.id, path,
            )
            return AppendOutcome.DUPLICATE_SKIPPED
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
        known.add(anchor.id)
    return AppendOutcome.APPENDED


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


def count_unique_anchor_ids(path: Path | str) -> int:
    """Count *distinct* anchor ids in a JSONL file.

    The line count (:func:`count_existing_anchors`) is what the resume logic
    in :mod:`ard.pipeline` compares against ``target_count``; this function
    makes the difference between lines and real anchors visible.
    """
    return len(_read_ids(Path(path)))


def anchor_to_dict(anchor: GeneratedAnchor) -> dict[str, Any]:
    """Convert a :class:`GeneratedAnchor` to a dict in the unified JSONL format.

    Output format (aligned with graspo + OPD v3.0.0 additions):
    ```json
    {
      "id": "anchor_<sha256_hex16>",
      "source": "ard",
      "data_source": "ard_text",
      "schema_version": "3.0.0",
      "messages": [...],
      "targets": [{
        "id": "primary",
        "output": {
          "content": "<target_answer>",
          "reasoning": "<teacher reasoning text or null>"
        }
      }],
      "anchor_meta": {...},
      "input_generator_model": "...",
      "teacher_id": "..."
    }
    ```

    ``content`` and ``reasoning`` are separate keys on purpose (§1.2 契约 2):
    thinking is not the answer, and downstream decides on its own whether it
    wants the reasoning.  ``reasoning`` is ``null`` when the teacher did not
    think (``enable_thinking = false``) — never ``""``.

    ``data_source`` (per-record OPD routing key, set at anchor construction),
    ``schema_version`` (the single format version, §1.4) and
    ``input_generator_model`` are top-level fields so downstream can route and
    version records without re-deriving them.
    """
    return {
        "id": anchor.id,
        "source": "ard",
        "data_source": anchor.data_source,
        "schema_version": SCHEMA_VERSION,
        "messages": anchor.messages,
        "targets": [
            {
                "id": "primary",
                "output": {
                    "content": anchor.target_answer,
                    "reasoning": anchor.reasoning,
                },
            }
        ],
        "anchor_meta": anchor.anchor_meta,
        "input_generator_model": anchor.input_generator_model,
        "teacher_id": anchor.target_model,
    }


def write_anchor_bank(anchors: list[GeneratedAnchor], output_path: str | Path) -> None:
    """Rewrite a whole JSONL bank, applying the same output gates as
    :func:`append_anchor`.

    A malformed conversation or an off-vocabulary ``data_source`` is never
    written — because this function writes the *entire* file at once, silently
    dropping rows would hide the problem, so it raises instead.  Duplicate ids
    are dropped (first occurrence wins) and reported.

    Args:
        anchors: Anchors to persist.
        output_path: Target JSONL file (parent directories are created).

    Raises:
        ValueError: If any anchor violates the message-shape contract or
            carries a ``data_source`` outside the controlled vocabulary.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rejected = [
        (a.id, message_shape_error(a.messages))
        for a in anchors
        if message_shape_error(a.messages) is not None
    ]
    if rejected:
        raise ValueError(
            "refusing to write a bank containing malformed conversations: "
            + "; ".join(f"{anchor_id}: {reason}" for anchor_id, reason in rejected)
        )

    rejected_sources = [
        (a.id, data_source_error(a))
        for a in anchors
        if data_source_error(a) is not None
    ]
    if rejected_sources:
        raise ValueError(
            "refusing to write a bank containing unrouteable data_source values: "
            + "; ".join(f"{anchor_id}: {reason}" for anchor_id, reason in rejected_sources)
        )

    unique: dict[str, GeneratedAnchor] = {}
    duplicates: list[str] = []
    for anchor in anchors:
        if anchor.id in unique:
            duplicates.append(anchor.id)
            continue
        unique[anchor.id] = anchor
    if duplicates:
        logger.warning(
            "write_anchor_bank: dropped %d duplicate id(s) for %s: %s",
            len(duplicates), output_path, ", ".join(sorted(set(duplicates))),
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for anchor in unique.values():
            f.write(json.dumps(anchor_to_dict(anchor), ensure_ascii=False) + "\n")


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


def _manifest_breakdown(
    meta: dict[str, Any],
    data_source: str,
) -> tuple[str, str, str, str, str]:
    """Extract the (domain, language, capability, mode, data_source) of a record.

    The sampling dimensions that are most useful to see *after* a run, plus the
    routing key.  The system-prompt mode is included because it became a
    sampling dimension in v3.0.0 (B3), and ``data_source`` because it decides
    which sub-corpus the training side routes the record to: a distribution
    quietly stuck on one value looks exactly like a healthy one (§3.2).
    """
    return (
        meta.get("knowledge_domain", meta.get("visual_domain", "unknown")),
        meta.get("language", "unknown"),
        meta.get("capability", meta.get("question_type", "unknown")),
        meta.get("system_prompt_mode", "unknown"),
        data_source,
    )


def _tally(
    breakdowns: list[tuple[str, str, str, str, str]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    """Count every position of the breakdown over all records."""
    domains: dict[str, int] = {}
    languages: dict[str, int] = {}
    capabilities: dict[str, int] = {}
    system_prompt_modes: dict[str, int] = {}
    data_sources: dict[str, int] = {}
    for domain, language, capability, mode, data_source in breakdowns:
        domains[domain] = domains.get(domain, 0) + 1
        languages[language] = languages.get(language, 0) + 1
        capabilities[capability] = capabilities.get(capability, 0) + 1
        system_prompt_modes[mode] = system_prompt_modes.get(mode, 0) + 1
        data_sources[data_source] = data_sources.get(data_source, 0) + 1
    return domains, languages, capabilities, system_prompt_modes, data_sources


def build_manifest(
    anchors: list[GeneratedAnchor],
    output_dir: str | Path,
    config_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest dict summarizing the anchor bank."""
    domains, languages, capabilities, system_prompt_modes, data_sources = _tally(
        [_manifest_breakdown(a.anchor_meta, a.data_source.value) for a in anchors]
    )
    manifest: dict[str, Any] = {
        "total_anchors": len(anchors),
        "domains": domains,
        "languages": languages,
        "capabilities": capabilities,
        "system_prompt_modes": system_prompt_modes,
        "data_sources": data_sources,
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

    Bank composition only: run health (abandoned anchors, reasoning failures,
    cooldowns) is not derivable from the surviving records and is attached
    separately by :func:`with_generation_report`.
    """
    domains, languages, capabilities, system_prompt_modes, data_sources = _tally(
        [
            _manifest_breakdown(r.get("anchor_meta", {}), r.get("data_source", "unknown"))
            for r in records
        ]
    )
    manifest: dict[str, Any] = {
        "total_anchors": len(records),
        "domains": domains,
        "languages": languages,
        "capabilities": capabilities,
        "system_prompt_modes": system_prompt_modes,
        "data_sources": data_sources,
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


def with_generation_report(
    manifest: dict[str, Any],
    *,
    stats: dict[str, Any] | None = None,
    failures: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Attach generation health counters to *manifest* (in place) and return it.

    The manifest used to describe only the anchors that *survived*
    (``total_anchors`` / ``domains`` / …), so a run that dropped a third of its
    anchors produced a manifest that looked perfectly healthy.  That is the
    failure mode this project kept paying for, so the dropped-anchor accounting
    now travels with the output (§3.2 透明退路).

    The result is one nested ``"generation"`` object with two sub-objects:

    * ``counters`` — what happened to each requested anchor:
      ``requested`` / ``succeeded`` / ``abandoned_total`` /
      ``abandoned_by_reason`` / ``written`` / ``rejected_invalid_shape`` /
      ``duplicate_ids`` / ``backpressure_events``.
    * ``failures`` — process-level failure counters, keyed by their own
      machine-readable tags: reasoning/empty-content counters
      (``responses``, ``empty_content``, ``reasoning_only_responses``,
      ``truncated_empty``, …).

    Both sub-objects are written **only when non-empty**, and zero-valued
    entries are dropped, so a healthy run gains no noise while an unhealthy one
    can never be mistaken for a healthy one.

    Args:
        manifest: Manifest dict built by :func:`build_manifest` or
            :func:`build_manifest_from_records`.
        stats: :meth:`ard.domain.text_anchor.AnchorGenerationStats.to_manifest_dict`
            output, or ``None`` when the run did not generate anchors
            (checkpoint-resume path).
        failures: Process-level counters (reasoning / empty-content stats).

    Returns:
        The same *manifest* object, enriched.
    """
    counters = {key: value for key, value in (stats or {}).items() if value}
    observed_failures = {
        key: value for key, value in (failures or {}).items() if value
    }
    if not counters and not observed_failures:
        return manifest
    generation: dict[str, Any] = {}
    if counters:
        generation["counters"] = counters
    if observed_failures:
        generation["failures"] = observed_failures
    manifest["generation"] = generation
    return manifest
