"""Main pipeline orchestrator for ARD anchor generation.

Responsibility: turn a validated :class:`~ard.config.ARDConfig` into one run
directory — plan the anchors, refuse an unusable boundary before any side
effect, resume a previous bank, generate the anchors, and write the artifact
(``anchor_bank.jsonl``, ``config.toml``, ``manifest.json``, ``results/``,
``images/``).

Boundaries:

* **N and every other content parameter come from the config** (``[generation]
  count``); this module adds no CLI knob of its own (§10.1).  ``--smoke`` is the
  one run-boundary flag and only changes the plan's *size*, never a second
  construction rule.
* **Identity is not content.**  An anchor id is the plan *position*
  (``sampling.format_anchor_id``), so the pipeline never derives a coordinate
  from an id and never treats a repeated coordinate as a duplicate.  The one
  place ids and coordinates meet is the resume guard, and it compares them
  *per id* (§2.3).
* **The sampling rule lives in** :mod:`ard.core.sampling`; the image-addressing
  rule in :mod:`ard.domain.image_store`.  This module wires them together and
  owns only the run-level decisions (boundary checks, resume, manifest).

Phase 3 unified flow: all anchors (text + multimodal) are generated from
:class:`AnchorSpec` objects via a single code path.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w

from ard.backends.api_client import (
    ChatAPIClient,
    ChatAPIConfig,
)
from ard.backends.api_client import (
    reasoning_stats as api_client_reasoning_stats,
)
from ard.backends.ontology_loader import load_ontology_v4
from ard.config import ARDConfig, ConfigError
from ard.core import acceptance
from ard.core.ontology import OntologyV4
from ard.core.quota import allocate_images, stamp_image_bookkeeping
from ard.core.sampling import (
    PLAN_IDENTITY_ALGORITHM,
    PLAN_IDENTITY_AXES,
    PlanIdentity,
    ontology_sha256,
    plan_rounds,
    sample_anchors,
    unit_total,
)
from ard.core.types import (
    AnchorGenerationConfig,
    AnchorSpec,
    AnchorSpecList,
    JsonObject,
    JsonObjectList,
    StringList,
)
from ard.domain.bank import (
    build_manifest_from_records,
    read_anchor_bank,
    with_generation_report,
    write_manifest,
)
from ard.domain.image_store import (
    CONVERTABLE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    VISUAL_DOMAIN_LAYOUT,
    DomainImageResolution,
    convert_and_copy_images,
    copy_images_to_output,
    domain_directory,
    list_domain_images,
    list_pool_images,
    resolve_domain_images,
    select_domain_image,
)
from ard.domain.text_anchor import AnchorGenerationStats, generate_text_anchors
from ard.logging import configure_file_logging

logger = logging.getLogger(__name__)

# ── Secret redaction for the config snapshot (§2.3 boundary check) ────────
#
# The merged config carries credentials (``input_generator.api_key``,
# ``target_model.api_key``).  Those values must never reach the output
# directory: users share/pack output dirs, so a snapshot written verbatim
# leaks the keys.  Redaction is applied once, to the single ``config_info``
# dict that feeds **both** the archived ``config.toml`` and the manifest's
# ``config`` section — one definition, both sinks (§1.4).
#
# Endpoints (``api_base``) and model names are **not** masked: §7.1 classes
# them as environment fields, not secrets, and they are part of what the
# snapshot exists to record.  The residual risk is accepted and stated here:
# sharing an output directory also shares the endpoint it records.

REDACTED_PLACEHOLDER = "***REDACTED***"
"""Stand-in written in place of any secret value in an output snapshot."""

REDACTED_KEY_WORDS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "key",
    "token",
    "secret",
    "password",
    "passwd",
    "bearer",
    "credential",
    "auth",
)
"""Credential words that mark a config key as secret-bearing.

A key matches when **any of its words** is one of these, or when its flattened
form equals one of them.  Splitting on ``_``/``-``/spaces and camelCase
boundaries, ``api_key``, ``API-KEY``, ``apiKey``, ``some_api_key``,
``aws_access_key_id``, ``client_secret``, ``access_token``, ``hf_token``,
``DB_PASSWORD`` and ``Authorization-Bearer`` all match.

This is deliberately **word** matching, not substring matching: ``max_tokens``
splits into ``max`` + ``tokens``, and ``tokens`` is not ``token``, so it does
not match.  Same for ``max_tokens_with_image`` and ``first_token_timeout``.
Those fields are sizes/durations — masking them would destroy the snapshot's
ability to reproduce the run (§2.3: redact secrets, do not corrupt the record).

``api_base`` is deliberately absent — an endpoint is a location, not a
credential; the residual risk of publishing it with the snapshot is accepted.
"""


def _split_key(key: str) -> list[str]:
    """Split a key name into lower-cased words on separators and camelCase.

    ``"api_key"`` → ``["api", "key"]``, ``"clientSecret"`` →
    ``["client", "secret"]``, ``"max_tokens"`` → ``["max", "tokens"]``.
    """
    with_boundaries = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return [word for word in re.split(r"[^a-zA-Z0-9]+", with_boundaries.lower()) if word]


def _flatten_key(key: str) -> str:
    """Key words joined without separators: ``"api_key"`` → ``"apikey"``."""
    return "".join(_split_key(key))


REDACTED_KEY_WORD_SET: frozenset[str] = frozenset(REDACTED_KEY_WORDS)
"""Word-level comparison set — the single definition of "is a secret key"."""

REDACTED_KEY_FLAT_SET: frozenset[str] = frozenset(_flatten_key(word) for word in REDACTED_KEY_WORDS)
"""Flattened names, so single-token spellings like ``apikey`` also match."""

NOT_SECRET_KEY_WORDS: frozenset[str] = frozenset({"timeout", "timeouts"})
"""Words that look like credentials but name a duration, never a value.

``first_token_timeout`` / ``inter_token_timeout`` contain the word ``token``
yet hold seconds.  Providing/omitting a timeout cannot leak a key, so these are
exempt — the snapshot must keep them to reproduce the run.  This list is the
explicit, auditable place where "looks like a secret but is not one" is
recorded (§2.2 显式即防呆); it is not a heuristic escape hatch.
"""


def _is_secret_key(key: object) -> bool:
    """Whether *key* names a credential field that must be masked.

    Matches when any word of the key is a credential word, or when the key's
    flattened spelling is one — see :data:`REDACTED_KEY_WORDS` for why this is
    word matching rather than substring matching (``max_tokens`` must survive).
    """
    if not isinstance(key, str):
        return False
    words = _split_key(key)
    if any(word in NOT_SECRET_KEY_WORDS for word in words):
        return False
    if any(word in REDACTED_KEY_WORD_SET for word in words):
        return True
    return "".join(words) in REDACTED_KEY_FLAT_SET


def _redact_mapping(mapping: dict[str, Any], found: set[str]) -> dict[str, Any]:
    """Return *mapping* rebuilt with credential values masked.

    The dict case is its own function so a redacted mapping's type is stated
    where it is produced (§2.2 显式即防呆).  ``_redact_value`` cannot express
    "dict in, dict out" — it must accept and return ``object`` to recurse into
    arbitrary JSON — so a caller that hands in a mapping would otherwise have to
    re-assert the result's type.
    """
    redacted: dict[str, Any] = {}
    for key, item in mapping.items():
        if _is_secret_key(key) and item is not None and item != "":
            redacted[key] = REDACTED_PLACEHOLDER
            found.add(str(key))
        else:
            redacted[key] = _redact_value(item, found)
    return redacted


def _redact_value(value: object, found: set[str]) -> object:
    """Recursively copy *value*, masking credential values, collecting keys hit.

    Containers are rebuilt (never mutated in place) so the caller's config
    dict is left untouched.
    """
    if isinstance(value, dict):
        return _redact_mapping(value, found)
    if isinstance(value, list):
        return [_redact_value(item, found) for item in value]
    return value


def _redact_secrets(config_info: dict[str, Any]) -> dict[str, Any]:
    """Return *config_info* with every credential value replaced by a mask.

    Recurses through nested dicts and lists, so structured config sections
    (``[input_generator]``, ``[target_model]``) are covered as well as any
    future nesting.  A value that is ``None`` or ``""`` is left as-is: those
    mean "no key configured" (§2.2 — ``None`` is the only empty value), and
    masking them would make an absent credential indistinguishable from a
    redacted one.  Every other value under a matching key is masked, so the
    key name never vouches for a secret's absence.

    The hit keys are logged at WARNING (§3.2 — redaction is not silent).
    """
    found: set[str] = set()
    redacted_info = _redact_mapping(config_info, found)
    if found:
        logger.warning(
            "Redacted %d credential field(s) from the config snapshot: %s. "
            "Values are replaced with %r so output directories can be shared "
            "safely; real values stay in the config file / override TOML.",
            len(found),
            ", ".join(sorted(found)),
            REDACTED_PLACEHOLDER,
        )
    return redacted_info


CONFIG_SNAPSHOT_HEADER = """\
# Run configuration snapshot (§8.5).  This file is the merged base + override
# configuration that produced the artifact next to it, so the run can be
# described (and repeated) after the fact.
#
# Credential values are masked as ***REDACTED***; the run's log names the
# masked fields.  Endpoints and model names are kept verbatim: they are
# environment fields, not secrets (§7.1), and the snapshot exists to record
# where the run pointed.
#
# An unset optional field is written as "" — TOML has no null literal, and ""
# is the representation load_config turns back into "not provided" (§2.2).
#
# Feed this file back as the base config, supplying the real credentials in an
# override (§7.1), e.g.:
#   python -m ard --config <this file> --override config.override.toml
"""
"""Explanatory header prepended to the archived config snapshot."""


def _none_as_empty_string_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    """Return *mapping* rebuilt with every ``None`` replaced by ``""`` for TOML.

    The mapping case is its own function for the same reason as
    :func:`_redact_mapping`: :func:`_none_as_empty_string` must accept and
    return ``object`` to recurse into arbitrary JSON, so a caller that knows it
    holds a mapping should not have to re-assert the result's type (§2.2).
    """
    return {key: _none_as_empty_string(item) for key, item in mapping.items()}


def _none_as_empty_string(value: object) -> object:
    """Return *value* with every ``None`` replaced by ``""`` for TOML.

    TOML has no null literal.  ``""`` is the serialisation of "not provided"
    and :func:`ard.config.load_config` normalises it straight back to ``None``
    (§2.2 serialization-boundary exception, the same rule the shipped
    ``configs/config.toml`` uses).  Without this the snapshot of a run that
    leaves e.g. ``output.directory`` unset could not be written at all.
    """
    if isinstance(value, dict):
        return _none_as_empty_string_mapping(value)
    if isinstance(value, list):
        return [_none_as_empty_string(item) for item in value]
    if value is None:
        return ""
    return value


def _config_snapshot_toml(config_info: dict[str, Any]) -> str:
    """Render the merged config as a TOML document ``--config`` can read back.

    The archive is written in the project's own config format so §8.5's
    promise ("the output directory describes the run by itself") holds
    literally: the file can be passed to ``--config`` again, which a JSON
    snapshot could not, because the loader reads TOML only (§10.1).
    """
    return CONFIG_SNAPSHOT_HEADER + tomli_w.dumps(_none_as_empty_string_mapping(config_info))


def _counter_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    """Return the per-key difference between two counter snapshots.

    Keys present only in *before* are reported as ``0`` rather than dropped:
    "measured, nothing seen this run" and "never measured" are different
    statements, and only the caller can decide which one matters.
    """
    keys = set(after) | set(before)
    return {key: after.get(key, 0) - before.get(key, 0) for key in keys}


def _log_target_model_reasoning_stats(
    after: dict[str, int],
    before: dict[str, int],
    anchors_generated: int,
) -> dict[str, int]:
    """Log the reasoning-vs-content accounting of the target model.

    Reasoning tokens (``delta.reasoning``) are dropped from the answer by
    design, but they consume ``max_tokens`` first.  When a response spends its
    whole budget thinking, the target answer is empty and the anchor cannot be
    built — a silent loss unless it is counted and announced (§3.2).

    A run with ``empty_content > 0`` did not produce the anchors it looks like
    it produced; the usual cause is ``enable_thinking = true`` (or an
    ``enable_thinking`` value that reaches the server as ``null``) combined with
    a ``max_tokens`` that is too small to hold reasoning **and** an answer.

    Returns:
        The measured delta, so the same numbers can be published in
        ``manifest.json`` instead of being recomputed (and drifting).
    """
    delta = _counter_delta(after, before)
    responses = delta.get("responses", 0)
    if responses == 0:
        return delta
    empty = delta.get("empty_content", 0)
    reasoning_only = delta.get("reasoning_only_responses", 0)
    truncated = delta.get("truncated_empty", 0)
    summary = (
        f"Target model reasoning stats: {responses} response(s), "
        f"{delta.get('reasoning_responses', 0)} with reasoning "
        f"({delta.get('reasoning_chars', 0)} reasoning chars), "
        f"{empty} empty content, {reasoning_only} reasoning-only, "
        f"{truncated} empty-and-truncated by max_tokens; "
        f"{anchors_generated} anchor(s) generated."
    )
    if empty == 0:
        logger.info(summary)
        return delta
    logger.warning(
        "%s At least one target answer was empty — the failure is a model-output "
        "problem, not a network problem. Check whether enable_thinking is on and "
        "whether max_tokens is large enough for reasoning plus answer.",
        summary,
    )
    return delta


#: Signature of the plan builder: one :class:`~ard.config.ARDConfig` in, the
#: run's :class:`~ard.core.types.AnchorSpec` plan out.  Named so the ``run``
#: argument and its default stay in sync.
SpecSampler = Callable[[ARDConfig], list[AnchorSpec]]

#: Images allocated per anchor.  The construction rule gives an image-modality
#: coordinate exactly one ``visual_domain``, so one image per anchor is what the
#: plan describes — it is not a configurable knob.
IMAGES_PER_ANCHOR = 1

#: How many missing coordinates the acceptance warning spells out before it
#: summarises the rest (the count is always exact).  A wholly failed run must not
#: paste 1,826 coordinates into the log.
_MISSING_COORDINATE_LIMIT = 20

SMOKE_RUN_SUFFIX = "_smoke"
"""Run-directory suffix that keeps a smoke artifact from looking like a delivery.

``--smoke`` is a *run-boundary* parameter (constitution §10.1): it produces a
deliberately incomplete artifact and must never be confusable with the full
dataset.  The marker travels with the artifact itself — in the run directory's
name and in ``manifest.json`` — not only in a log line (§2.4 操作防呆).
"""


def _resolve_run_directory(config: ARDConfig, *, timestamp: str, smoke: bool) -> Path:
    """Return the run directory, tagging it with :data:`SMOKE_RUN_SUFFIX` on smoke.

    The marker is appended to the configured ``output.directory`` as well as to
    the timestamped default, so a smoke run can never write into the directory a
    delivery is expected from.  Idempotent: a name already ending in the suffix
    is not tagged twice.
    """
    output_dir = (
        Path(config.output.directory)
        if config.output.directory
        else Path("outputs") / f"ard_dataset_{timestamp}"
    )
    if smoke and not output_dir.name.endswith(SMOKE_RUN_SUFFIX):
        output_dir = output_dir.with_name(output_dir.name + SMOKE_RUN_SUFFIX)
    return output_dir


def _recorded_plan_identity(manifest_path: Path) -> PlanIdentity | None:
    """The ``plan_identity`` a previous run recorded in its ``manifest.json``.

    Returns ``None`` when the manifest is absent, unreadable, or does not carry
    a well-formed identity of this algorithm — "no recorded identity" is a
    different fact from "identity 0", and is handled separately below (§3.2).

    The same reader serves :func:`_recorded_progress_identity`, so both sinks
    define "a recorded identity" identically.
    """
    return _identity_from_file(manifest_path)


def _recorded_progress_identity(output_dir: Path) -> PlanIdentity | None:
    """The ``plan_identity`` an unfinished run left in *output_dir*.

    That is the plan the run directory is already bound to, so a resume must
    match it exactly like it matches a finished run's manifest identity —
    otherwise an interrupted run would be the one state a caller could append a
    different plan to (§2.3).  ``None`` when no (well-formed) record exists.
    """
    return _identity_from_file(output_dir / PROGRESS_RECORD_FILENAME)


MANIFEST_STATUS_COMPLETE = "complete"
"""``status`` the authoritative ``manifest.json`` declares.

The final manifest is a *completed* declaration, while
:data:`PROGRESS_RECORD_FILENAME` carries ``status: "in_progress"``.  Both sinks
use the key ``status`` so a reader compares like with like, and a completed run
can never be presented as an unfinished one (or the other way round).
"""

PROGRESS_RECORD_FILENAME = "plan_identity.in_progress.json"
"""Name of the run directory's intermediate plan-identity record.

The authoritative plan name of a run directory is ``manifest.json``'s
``plan_identity``, and that file is written only at the very end.  A run that is
interrupted — crashed, killed, or still generating when the box goes away; on
the 1,826-anchor path that is the common case, not the exception — would
otherwise leave a bank with **no plan name at all**, so the artifact could not
be bound to the plan that produced it and could not be audited or resumed with
confidence.

This file closes that gap without ever standing in for the manifest: its own
``status`` (``"in_progress"``), its own schema key and its own filename all say
"this is not the final declaration", and it deliberately carries no run health
(no ``generation`` counters/failures) because those numbers do not exist until
the run is over.  It is refreshed by every invocation that samples a plan and
removed once the authoritative manifest has been written.
"""


def _identity_from_record(data: object) -> PlanIdentity | None:
    """The well-formed ``plan_identity`` inside *data*, or ``None``.

    One reader for both sinks (the final ``manifest.json`` and the intermediate
    progress record): the identity is the same object either way, and validating
    it in one place is what keeps "no recorded identity" a different fact from
    "identity 0" (§3.2).

    **Version tolerance.** A ``version: 1`` record — written by a v4 run — has
    only ``algorithm`` / ``version`` / ``plan_size`` / ``digest``.  Those four
    fields are the ones the resume guard compares, so a v1 record stays
    *readable*; the v2-only fields it does not carry (``sampling``,
    ``ontology_sha256``, ``seed``, ``count``, ``unit_total``) are read as
    explicit "unknown" placeholders rather than recomputed.  Recomputing them
    would mean re-deriving a v4 plan with a v5 rule and inventing a history the
    artifact never had (§1.4 单一真相源); "unknown" is the honest value, and a
    v1 directory is refused on its *ids* (see
    :func:`_refuse_foreign_records_on_resume`), not on these placeholders.
    """
    recorded = data.get("plan_identity") if isinstance(data, dict) else None
    if not isinstance(recorded, dict):
        return None
    digest = recorded.get("digest")
    plan_size = recorded.get("plan_size")
    version = recorded.get("version")
    if recorded.get("algorithm") != PLAN_IDENTITY_ALGORITHM:
        return None
    if not isinstance(digest, str) or not digest:
        return None
    if not isinstance(plan_size, int) or not isinstance(version, int):
        return None

    def _text(key: str) -> str:
        value = recorded.get(key)
        return value if isinstance(value, str) else ""

    def _integer(key: str, *, default: int) -> int:
        value = recorded.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            return default
        return value

    count = recorded.get("count")
    return PlanIdentity(
        algorithm=PLAN_IDENTITY_ALGORITHM,
        version=version,
        sampling=_text("sampling"),
        ontology_sha256=_text("ontology_sha256"),
        seed=_integer("seed", default=0),
        count=None if count is None else _integer("count", default=0),
        unit_total=_integer("unit_total", default=0),
        plan_size=plan_size,
        digest=digest,
    )


def _identity_from_file(path: Path) -> PlanIdentity | None:
    """The identity a JSON artifact at *path* records, or ``None`` when unusable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _identity_from_record(data)


def _build_progress_record(
    *,
    identity: PlanIdentity,
    started_at: str,
    existing: int,
    new: int,
    written: int,
    smoke: bool,
    output_dir: Path,
) -> dict[str, Any]:
    """Build the intermediate plan-identity record for an unfinished run (§2.4).

    The record declares its own status in the file name *and* in ``status``, so
    it can never be mistaken for ``manifest.json``: it is a *bound* plan name
    written before the first endpoint call and never refreshed afterwards, so
    its ``counters`` describe the *plan*, not progress: ``existing`` is the bank
    size when this invocation started, ``new`` the still-pending anchors it was
    asked for (``len(specs)``), and ``written`` the same number as ``new`` — the
    output planned, not records on disk, which is the manifest's ``total_anchors``.

    **The counters are not a counting truth source, and say so.**  Nothing in
    this module refreshes the file while the generator runs — the write loop
    lives in :mod:`ard.domain.text_anchor`, which does not know about the
    progress record — so its numbers can only ever describe the plan as this
    invocation saw it at the start.  A reader that wants the bank's current size
    reads ``anchor_bank.jsonl`` (``written`` records) or the finished
    ``manifest.json``'s ``total_anchors`` / ``generation``; the record's
    ``counters_are_live: false`` makes that a machine-readable statement rather
    than a convention, so a live-looking ``written`` can no longer be read as
    "this many records are on disk" (§3.2 透明退路).

    Args:
        identity: The plan's identity (``PlanIdentity.of(plan, ontology_sha256=...,
            seed=..., count=..., unit_total=...)``).
        started_at: Local start time, ``%Y-%m-%d %H:%M:%S``.
        existing: Anchors already in the bank when this invocation started.
        new: Anchors this invocation asked the generator for.
        written: The planned output count; ``pipeline.run`` passes the same
            value as *new*, so it is never a count of persisted records.
        smoke: Whether this run was ``--smoke``.
        output_dir: The run directory the record describes.

    Returns:
        The JSON-ready record.
    """
    return {
        "ard_progress_record": "plan_identity/v1",
        "status": "in_progress",
        "note": (
            "Intermediate record of a run that has not finished. It binds this "
            "directory to a plan, but its counters are incomplete and it is NOT "
            "the final declaration — read manifest.json for that. Safe to delete."
        ),
        "counters_are_live": False,
        "counters_note": (
            "counters is a snapshot of the plan as this invocation started, not "
            "of the bank on disk; the counting truth sources are "
            "anchor_bank.jsonl and the finished manifest.json."
        ),
        "started_at": started_at,
        "plan_identity": identity.as_dict(),
        "counters": {"existing": existing, "new": new, "written": written},
        "smoke": smoke,
        "output_dir": str(output_dir),
    }


def _write_progress_record(output_dir: Path, record: dict[str, Any]) -> Path:
    """Write *record* to the run directory's progress record, replacing a stale one.

    A run that samples a plan **is** the run that directory is now about, so the
    record is refreshed in place rather than appended to: one file, one current
    state (§1.4 单一真相源).  It is always written, even when it did not exist
    before — "the process is running" is exactly the state that was invisible.
    """
    path = output_dir / PROGRESS_RECORD_FILENAME
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    identity = record["plan_identity"]
    logger.info(
        "Plan identity %s recorded in %s (status: in_progress; manifest.json is "
        "written only when the run finishes)",
        identity["digest"],
        path,
    )
    return path


def _clear_progress_record(output_dir: Path) -> None:
    """Remove the intermediate record once the authoritative manifest exists.

    ``manifest.json`` with ``status: complete`` and the same ``plan_identity``
    is the final word; keeping a *stale* ``in_progress`` file next to it would be
    an ambiguity for no benefit — and it is the ambiguity a smoke-scaled audit
    script could trip over.  A failed unlink is a WARNING, never a run failure
    (§3.2: announce the degradation, do not fail a finished run over
    housekeeping).
    """
    path = output_dir / PROGRESS_RECORD_FILENAME
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning(
            "Could not remove the intermediate progress record %s: %s. It no longer "
            "matches the finished manifest; delete it by hand if the run directory "
            "must hold only current state.",
            path,
            exc,
        )
        return
    logger.info("Removed the intermediate progress record %s (run finished).", path)


def _declare_manifest_status(manifest: dict[str, Any], *, status: str) -> dict[str, Any]:
    """Declare the artifact's lifecycle ``status`` in *manifest* (§7.4)."""
    manifest["status"] = status
    return manifest


def _refresh_manifest_for_no_generation(
    manifest: dict[str, Any], path: Path, recorded: PlanIdentity | None
) -> bool:
    """Write *manifest* to ``manifest.json`` — unless a final one already names the plan.

    The checkpoint/no-op path (``run`` found nothing left to generate) used to
    write its manifest unconditionally.  That manifest is built from the bank
    alone, so it carries **no** ``generation`` section: the counters, dropped
    anchors and failure accounting of the run that actually did the generating
    were replaced by a clean-looking summary — the §3.2 failure this project
    keeps paying for.  It would also have let a later invocation restate a
    finished run as unfinished.  So when the manifest already on disk records
    *this* plan's identity, it is the authoritative statement about this run
    directory and is left byte-for-byte untouched.  A manifest that is absent
    (or that records a different plan — unreachable here, the identity guard
    refuses such a resume earlier) is written as before.

    Args:
        manifest: The manifest built from the current bank.
        path: The run directory's ``manifest.json``.
        recorded: The identity the existing ``manifest.json`` records, or
            ``None`` when there is no readable manifest.

    Returns:
        ``True`` when the manifest was written, ``False`` when it was preserved.
    """
    declared = manifest["plan_identity"]
    same_plan = (
        recorded is not None
        and recorded.digest == declared["digest"]
        and recorded.plan_size == declared["plan_size"]
        and recorded.version == declared["version"]
    )
    if same_plan:
        logger.info(
            "Nothing to generate; %s already records this plan. Left untouched "
            "(its run health stays authoritative).",
            path,
        )
        return False
    write_manifest(_declare_manifest_status(manifest, status=MANIFEST_STATUS_COMPLETE), path)
    return True


def _coordinate_matches(record: JsonObject, spec: AnchorSpec) -> bool:
    """Whether *record* holds the same coordinate as *spec* on every plan axis.

    The comparison is restricted to :data:`~ard.core.sampling.PLAN_IDENTITY_AXES`
    — the canonical coordinate axes — because a bank record's ``anchor_meta``
    carries pipeline bookkeeping on top of the coordinate (``has_image``,
    ``image_count``, added by image assignment).  Comparing the full mappings
    would call every real record a mismatch; comparing the axes compares the
    coordinate, which is what the guard is about.

    An axis absent on both sides compares equal (a text-only coordinate names no
    ``visual_domain``), and a value present on only one side is a mismatch.  Axis
    values are rendered through ``str`` so a JSON scalar and its Python
    equivalent (``1`` vs ``"1"``) do not read as two different coordinates.
    """
    meta = record.get("anchor_meta")
    if not isinstance(meta, dict):
        return False
    for axis in PLAN_IDENTITY_AXES:
        in_record = axis in meta
        in_spec = axis in spec.anchor_meta
        if in_record != in_spec:
            return False
        if in_record and str(meta[axis]) != str(spec.anchor_meta[axis]):
            return False
    return True


def _refuse_foreign_records_on_resume(
    output_dir: Path,
    identity: PlanIdentity,
    plan: list[AnchorSpec],
    existing_records: JsonObjectList,
    recorded: PlanIdentity | None,
) -> None:
    """Refuse a resume whose bank does not line up with this run's plan (§2.3).

    A resume may append to *output_dir* exactly when every record already on
    disk is at its **own plan position**: its id is an id of this plan **and**
    the coordinate that plan position carries is the coordinate the record
    holds.  A record that fails either half is refused; nothing is ever
    "repaired" or overwritten.

    Why both halves are needed.  The v5 id is a *position serial number*
    (``run_key + cycle + position``), deliberately independent of N, so raising
    ``[generation] count`` leaves every existing id byte-identical and
    "raise N and re-run the same directory" appends cleanly.  But a position id
    is **not a content fingerprint**: two different plans can mint the *same*
    ids for **different** coordinates.  The case that must not slip through is
    the smoke plan — it takes the first ``k`` units of *each modality* from
    cycle 0, while the full plan takes the first ``N`` units of the single
    cycle-0 order.  The smoke ids are a subset of the full plan's ids, yet the
    coordinate behind a shared id can differ, so an id-subset test would happily
    mix smoke records into a full-plan directory.  Comparing the coordinate at
    the id's own position catches seed changes, ontology changes *and* plan
    shape changes (smoke vs full) with one rule.

    The coordinate is compared only **at the same id**; it is never used to
    decide uniqueness.  Two samples with the same coordinate at two different
    ids are two legitimate samples (the generator is stochastic), and a bank
    holding both appends cleanly.

    Args:
        output_dir: The run directory being resumed.
        identity: This run's plan identity (used for the diagnostic message).
        plan: This run's plan, in plan order (ids and their coordinates).
        existing_records: The records already in the bank.
        recorded: The identity the directory already records, or ``None``.

    Raises:
        ConfigError: If the bank holds an id outside this plan, or an id whose
            recorded coordinate differs from the plan's coordinate there.
    """
    if not existing_records:
        return
    by_id = {spec.id: spec for spec in plan}
    existing_by_id = {
        record["id"]: record for record in existing_records if isinstance(record.get("id"), str)
    }
    if not existing_by_id:
        return
    foreign = sorted(set(existing_by_id) - set(by_id))
    mismatched = sorted(
        anchor_id
        for anchor_id, record in existing_by_id.items()
        if anchor_id in by_id and not _coordinate_matches(record, by_id[anchor_id])
    )
    if not foreign and not mismatched:
        if recorded is None:
            logger.warning(
                "%s has %d record(s) but no recorded plan_identity (checked %s and "
                "manifest.json); verified every existing anchor id and coordinate "
                "belongs to this run's plan, so this resume appends within one plan.",
                output_dir,
                len(existing_by_id),
                PROGRESS_RECORD_FILENAME,
            )
        elif recorded.digest != identity.digest:
            logger.info(
                "Resume extends the plan recorded in %s: %d existing anchor(s) all "
                "keep their plan position and coordinate (recorded %d coordinate(s), "
                "version %d -> this run %d coordinate(s), version %d). Appending the "
                "%d new anchor(s).",
                output_dir,
                len(existing_by_id),
                recorded.plan_size,
                recorded.version,
                identity.plan_size,
                identity.version,
                identity.plan_size - len(existing_by_id),
            )
        return

    reasons: list[str] = []
    if foreign:
        reasons.append(
            f"{len(foreign)} anchor id(s) that are not part of this run's plan "
            f"(first: {foreign[0]})"
        )
    if mismatched:
        reasons.append(
            f"{len(mismatched)} anchor id(s) whose recorded coordinate differs from "
            f"this run's plan at that position (first: {mismatched[0]}; the id names a "
            "plan position, not coordinate content, so an id that is present does not "
            "by itself mean the same sample)"
        )
    recorded_name = (
        f"plan identity {recorded.digest} (version {recorded.version}, "
        f"{recorded.plan_size} coordinates)"
        if recorded is not None
        else f"no recorded plan_identity (checked {PROGRESS_RECORD_FILENAME} and manifest.json)"
    )
    raise ConfigError(
        f"refusing to resume {output_dir}: its anchor_bank.jsonl holds "
        + " and ".join(reasons)
        + f". The run directory records {recorded_name}; this run's plan identity is "
        f"{identity.digest} (version {identity.version}, {identity.plan_size} "
        "coordinates). Appending would mix two datasets in one run directory while "
        "every counter stays green. A resume may only append to a bank whose ids and "
        "coordinates both line up with this plan — raising [generation] count keeps "
        "that true, changing [generation] seed, the ontology, or smoke/full shape does "
        "not. Give this batch of anchors its own output.directory (or set "
        "output.overwrite = true to replace this one deliberately)."
    )


def _declare_plan_identity(manifest: dict[str, Any], identity: PlanIdentity) -> None:
    """Record the run's plan identity in *manifest* (§7.4: self-naming field).

    One definition of the field name, both manifest sinks (the completed run and
    the no-op resume), so a reader of the artifact always finds the plan's name.
    """
    manifest["plan_identity"] = identity.as_dict()


SMOKE_PER_MODALITY: tuple[int, int] = (4, 4)
"""The smoke plan's ``(text, image)`` subset, taken from cycle 0's shuffle order.

``--smoke`` is a run-boundary parameter (constitution §10.1), not a config
field, so the *size* of the smoke artifact is defined here, next to the marker
that keeps it from looking like a delivery.  The two numbers are the pipeline's
one definition of "smoke scale"; the sampling rule itself takes them as the
``per_modality`` argument and has no smoke constant of its own (§1.4).
"""


def _declare_smoke(
    manifest: dict[str, Any],
    *,
    plan_size: int,
    unit_total: int,
    output_dir: Path,
) -> None:
    """Declare in *manifest* that this artifact is a smoke run (§3.2 透明退路).

    Writes the machine-readable claim (``smoke: true``) next to the numbers that
    prove it: what this run planned, and what one full cycle of this ontology
    holds.  A reader of the artifact alone can tell a smoke run from a delivery.
    """
    manifest["smoke"] = True
    manifest["smoke_plan"] = {
        "run_name": output_dir.name,
        "planned_anchors": plan_size,
        "text_blocks": SMOKE_PER_MODALITY[0],
        "image_blocks": SMOKE_PER_MODALITY[1],
        "full_expected_anchors": unit_total,
        "note": "SMOKE RUN — deliberately incomplete; not a deliverable.",
    }


def _distinct_coordinate_count(plan: list[AnchorSpec]) -> int:
    """How many *different coordinates* the plan holds (§4 readout).

    A coordinate is the sample's full axis mapping (``spec.anchor_meta``) — the
    ``id`` is deliberately left out, because the id is the plan *position*, and
    two positions legitimately carry the same coordinate.  Counting the ids
    would answer "how many samples", which is ``len(plan)``; counting the
    coordinates answers "how much of the sample space this plan touches".

    The key is the mapping's items sorted by axis name, so the count is
    independent of dict insertion order and of any axis the ontology adds.  The
    count spans the whole plan, not just the records written: it describes what
    the run *asked for*, and a partially failed run's coverage is still the
    plan's coverage (§3.2 — the shortfall is reported separately).
    """
    seen: set[tuple[tuple[str, str], ...]] = set()
    for spec in plan:
        seen.add(tuple(sorted((str(key), str(value)) for key, value in spec.anchor_meta.items())))
    return len(seen)


def _declare_plan_readout(
    manifest: dict[str, Any],
    *,
    plan: list[AnchorSpec],
    count: int | None,
    unit_total: int,
    rounds: tuple[int, int],
    written: int,
    seed: int,
    ontology_fingerprint: str,
    smoke: bool,
) -> None:
    """Publish the plan's shape and its coverage/density readouts (§4).

    One machine-readable section next to ``plan_identity``, so a consumer reads
    "what was planned, how far it reaches and how dense it is" without parsing
    logs or re-deriving the plan:

    * ``count`` — N exactly as requested; ``None`` is the "one full cycle"
      default, kept as ``None`` rather than silently replaced by ``U``.
    * ``unit_total`` — ``U``, one full cycle's unit count for this ontology.
    * ``full_cycles`` / ``last_cycle_size`` — the ``plan_rounds`` decomposition
      (``last_cycle_size == 0`` means the plan ends on a cycle boundary).
    * ``coverage_ratio`` — ``min(distinct coordinates, U) / U``, the same
      saturated-coverage convention the acceptance report uses
      (``ard.core.acceptance.structure_readout``).  A coordinate repeated in a
      later cycle is counted once, so this is a property of the plan's
      coordinate **set**, not of the bank; the raw count is published next to it
      as ``distinct_coordinates``.
    * ``density`` — ``count / U`` (with the unset default read as one cycle).
      Unlike coverage this is a *sample* count and repeats count again.

    ``ontology_fingerprint`` is the explicit "unknown" ``""`` only on the
    injected-plan test seam (``generate_specs``), which never loads an ontology;
    every real run carries the digest.
    """
    resolved_count = unit_total if count is None else count
    distinct = _distinct_coordinate_count(plan)
    manifest["plan"] = {
        "ontology_sha256": ontology_fingerprint,
        "seed": seed,
        "count": count,
        "unit_total": unit_total,
        "full_cycles": rounds[0],
        "last_cycle_size": rounds[1],
        "planned_anchors": len(plan),
        "written_anchors": written,
        "distinct_coordinates": distinct,
        "coverage_ratio": min(distinct, unit_total) / unit_total,
        "density": resolved_count / unit_total,
        "smoke": smoke,
    }


def _missing_image_message(
    image_root: Path,
    resolution: DomainImageResolution,
    planned: int,
    extensions: set[str],
) -> str:
    """The refusal message for a plan whose image tree holds no usable image.

    Reached only when the tree has **no usable image at all** — a domain whose
    own directory is empty is served from the tree-wide pool instead.  Names
    **every** domain left without a picture, the directory it would be filed
    under and how many anchors it affects (§2.3: a boundary rejection must be
    actionable, not a traceback from somewhere inside the copy loop), plus the
    sample count the refusal applies to — one call, one fix.
    """
    missing = resolution.missing
    affected = sum(len(anchor_ids) for anchor_ids in missing.values())
    listed = "\n".join(
        f"  - {domain}: would be read from {domain_directory(image_root, domain)} "
        f"(affects {len(missing[domain])} anchor(s))"
        for domain in sorted(missing)
    )
    allowed = ", ".join(sorted(extensions))
    return (
        f"image directory {image_root} has no usable image for the "
        f"{len(missing)} visual_domain(s) this plan requires — it holds no image at all.\n"
        f"Expected layout: {VISUAL_DOMAIN_LAYOUT} — one subdirectory per visual_domain "
        f"holding at least one image ({allowed}).\n"
        f"{listed}\n"
        f"Total affected anchors: {affected} of {planned} planned sample(s) (each is "
        f"labelled with a visual_domain and there is no picture anywhere under "
        f"{image_root} to reuse for it).\n"
        f"Provide at least one image, or set [images] skip_missing_images = true to "
        f"skip those {affected} anchor(s) — each is then logged as a WARNING and the "
        f"skipped count and domains are declared in manifest.json."
    )


def _declare_images(
    manifest: dict[str, Any],
    *,
    image_dir: str,
    config: ARDConfig,
    records: JsonObjectList,
    plan: list[AnchorSpec],
    cycle_of: dict[str, int],
    seed: int,
) -> None:
    """Declare the run directory's image addressing and any skipped anchors (§3.2/§3.3).

    A dataset with anchors missing is a different artifact from the one the
    construction rule describes, so the difference is machine-readable: the
    skipped count and the affected ``visual_domain`` values travel with the
    manifest, where a consumer reads them without parsing logs.

    **The readout is over the whole run directory, not over one invocation.**
    Every field is derived from what the directory actually holds — the
    ``anchor_bank.jsonl`` records plus the image tree — rather than from the
    pending set of the call that happened to write this manifest.  A segmented
    or resumed run (a first segment that generates the image anchors, a second
    that only appends a text-only one) used to restate the whole ``images``
    section from the *last* segment's handful of anchors, so a real 10-picture
    tree was declared as ``pool_candidate_count = 0`` and every reuse count as
    empty.  Deriving from the bank + tree makes the single-segment, the
    segmented and the resumed path agree, and makes the numbers recomputable
    offline (list the bank's ``(round, visual_domain)`` keys, then read the
    tree).

    v5 rotates a domain's images by round, so the pick is declared per
    ``(cycle, visual_domain)`` as well as aggregated:

    * ``resolved_images`` — one ``{cycle, visual_domain, image, fallback}`` row
      per ``(round, domain)`` the bank holds a record for, sorted by
      ``(cycle, domain)``.  The picture is the deterministic v5 pick for that
      ``(tree, domain, seed, round)``, so it is the file the run actually
      placed.  Two rounds with one image each are visibly different rows, so
      "the rotation happened" is checkable; ``fallback`` on the row says whether
      that round's picture is the domain's own or a reuse from the tree-wide
      pool.
    * ``domain_candidate_counts`` — ``visual_domain -> usable files in its own
      directory``, for every domain the bank holds an anchor of.  This is the
      readout that separates **real** variety (a domain with more than one
      candidate actually showed different files) from **fake** variety (a domain
      with one candidate necessarily reused it, however many rounds ran).
    * ``pool_candidate_count`` — how many usable images the tree-wide pool holds
      altogether.  ``1`` is the one-picture tree, where every domain and round
      necessarily shows that single file; ``0`` is the only state in which
      anchors are genuinely without a picture.
    * ``fallback_visual_domains`` / ``fallback_anchor_count`` — which domains,
      and how many anchors, are shown a **reused** picture because their own
      directory holds none.  Reuse must be visible: the user judges whether the
      substituted picture is good enough for their dataset, which they cannot
      do if the run quietly passes it off as a domain-matched one.
    * ``skipped_anchor_count`` / ``skipped_visual_domains`` — the planned image
      coordinates the bank does **not** hold, which happens in exactly one
      served state: the tree holds no usable image at all and
      ``[images] skip_missing_images`` is on, so the run dropped them before
      generation.  Anything the bank does not hold is dropped for no other
      reason, because with a usable picture anywhere no anchor is skipped.

    Args:
        manifest: The manifest dict to add the ``images`` section to (in place).
        image_dir: The image tree the run was pointed at (``--image-dir``).
        config: The run's config; its ``[images]`` section decides the accepted
            input extensions (the same set the copy step used) and whether
            skipping was pre-authorised.
        records: The run directory's bank records, in file order.
        plan: The run's plan, used only to name the image coordinates that a
            skipping run left out of *records*.
        cycle_of: ``spec id -> plan round`` (``index // U`` over the full plan),
            the same map the copy/assignment steps use.
        seed: The run seed, so a recomputed pick equals the placed one.
    """
    extensions = CONVERTABLE_EXTENSIONS if config.images.convert else SUPPORTED_EXTENSIONS
    image_root = Path(image_dir)
    pool = list_pool_images(image_root, extensions=extensions)

    # The ``(cycle, visual_domain)`` pairs the directory actually holds records
    # for, in canonical order.  A record whose round is not derivable (a
    # hand-written or legacy record) is read as round 0 rather than dropped: the
    # readout describes the bank, so a record must never vanish from it.
    held: dict[tuple[int, str], int] = {}
    for record in records:
        meta = record.get("anchor_meta")
        domain = meta.get("visual_domain") if isinstance(meta, dict) else None
        if not isinstance(domain, str):
            continue
        # A bank record's ``id`` is a JSON value, so it is narrowed before it
        # indexes the plan's ``str`` keys.  The count is unconditional: the
        # readout describes the bank, so a record with a valid domain is counted
        # even when its id is not a string (it is then read as round 0).
        anchor_id = record.get("id")
        cycle = cycle_of.get(anchor_id) if isinstance(anchor_id, str) else None
        key = (cycle if isinstance(cycle, int) else 0, domain)
        held[key] = held.get(key, 0) + 1

    # Scanned once per domain, not once per round: the candidates of a domain do
    # not depend on the round (only the pick does).
    candidates_by_domain: dict[str, list[Path]] = {}
    for domain in sorted({domain for _, domain in held}):
        candidates_by_domain[domain] = list_domain_images(image_root, domain, extensions=extensions)
    candidate_counts = {
        domain: len(candidates) for domain, candidates in candidates_by_domain.items()
    }
    # A domain is served by reuse when its own directory holds nothing and the
    # tree holds something; with an empty tree every domain is `missing`
    # instead, and the anchors are skipped (or the run was refused earlier).
    fallback_domains = {
        domain for domain, candidates in candidates_by_domain.items() if not candidates and pool
    }
    fallback_anchors = sum(
        count for (_, domain), count in held.items() if domain in fallback_domains
    )

    resolved_images: list[JsonObject] = []
    for cycle, domain in sorted(held):
        candidates = candidates_by_domain[domain]
        if candidates:
            chosen = select_domain_image(candidates, domain, seed, cycle=cycle)
        elif pool:
            chosen = select_domain_image(pool, domain, seed, cycle=cycle)
        else:
            # No usable image anywhere: this anchor has no picture to declare.
            continue
        resolved_images.append(
            {
                "cycle": cycle,
                "visual_domain": domain,
                "image": chosen.name,
                "fallback": domain in fallback_domains,
            }
        )

    skipped: dict[str, StringList] = {}
    if config.images.skip_missing_images and not pool:
        held_ids = {record.get("id") for record in records}
        for spec in plan:
            domain = spec.anchor_meta.get("visual_domain")
            if isinstance(domain, str) and spec.id not in held_ids:
                skipped.setdefault(domain, []).append(spec.id)

    manifest["images"] = {
        "image_dir": str(Path(image_dir).resolve()),
        "addressing": VISUAL_DOMAIN_LAYOUT,
        "skip_missing_images": config.images.skip_missing_images,
        "resolved_visual_domains": sorted({row["visual_domain"] for row in resolved_images}),
        "resolved_images": resolved_images,
        "domain_candidate_counts": {
            domain: candidate_counts[domain] for domain in sorted(candidate_counts)
        },
        "pool_candidate_count": len(pool),
        "fallback_visual_domains": sorted(fallback_domains),
        "fallback_anchor_count": fallback_anchors,
        "skipped_anchor_count": sum(len(anchor_ids) for anchor_ids in skipped.values()),
        "skipped_visual_domains": sorted(skipped),
    }


def _assign_images_by_domain(
    specs: list[AnchorSpec],
    rel_by_cycle_domain: dict[tuple[int, str], str],
    cycle_of: dict[str, int],
    output_dir: Path,
    rng: random.Random,
) -> None:
    """Give every image-modality spec the image of its own round and domain.

    v5 rotates a domain's picture by round, so the group key is
    ``(cycle, visual_domain)`` — not ``visual_domain`` alone.  Grouping by the
    domain only would hand every round the *same* file (the copy step keeps them
    apart, this step must too), which is exactly the "one run, one picture per
    domain" behaviour the rotation replaces.

    One ``(cycle, domain)`` = one group; :func:`ard.core.quota.allocate_images`
    is reused per group so the turn-filling rule (earliest eligible ``user``
    turn, at most :data:`IMAGES_PER_ANCHOR` images) keeps a single
    implementation.  Text-only specs never receive an image: their coordinate
    names no visual domain, so an image would be the same coordinate/content
    mismatch the addressing exists to prevent.

    This function only *places* pictures.  The ``has_image`` / ``image_count``
    bookkeeping is not decided here: ``run`` stamps it for every spec it will
    write, through the one implementation of that rule
    (:func:`ard.core.quota.stamp_image_bookkeeping`), so a spec that this
    function never sees is stamped exactly like the ones it does.

    ``cycle_of`` maps a spec's primary key to its plan round (``index // U`` over
    the full plan).  A spec whose round is unknown, or whose ``(cycle, domain)``
    has no resolved file, is left without a picture — its record then declares
    itself text-only rather than carrying some other round's image.
    """
    groups: dict[tuple[int, str], AnchorSpecList] = {}
    for spec in specs:
        domain = spec.anchor_meta.get("visual_domain")
        cycle = cycle_of.get(spec.id)
        key = (cycle, domain) if isinstance(cycle, int) and isinstance(domain, str) else None
        if key is not None and key in rel_by_cycle_domain:
            groups.setdefault(key, []).append(spec)
    for key, group in groups.items():
        allocate_images(group, [rel_by_cycle_domain[key]], IMAGES_PER_ANCHOR, rng)

    # Resolve image paths relative to output_dir for base64 encoding.
    for spec in specs:
        for turn in spec.turns:
            if turn.image_path:
                turn.image_path = str(output_dir / turn.image_path)


@dataclass(frozen=True, slots=True)
class _PlanContext:
    """A plan plus the facts needed to name and describe it.

    ``PlanIdentity`` v2 and the manifest's coverage/density readouts need more
    than the spec list: the ontology fingerprint, ``U``, the requested ``count``
    and the round decomposition.  They are computed once, here, next to the plan
    they describe, so the naming and the reporting cannot disagree with the
    plan (§1.4 单一真相源).

    Attributes:
        specs: The plan, in plan order.
        count: N exactly as requested; ``None`` means "one full cycle".  On the
            injected-plan seam it is the plan's length.
        unit_total: ``U`` — one full cycle's coverage-unit count.
        rounds: ``(full_cycles, last_cycle_size)``.
        ontology_sha256: The ontology fingerprint, or ``""`` on the
            injected-plan seam, which never loads an ontology.
        ontology: The validated v4 ontology the plan's reporting is about, or
            ``None`` when the injected-plan seam ran with the acceptance phase
            disabled (then nothing needs it).  The acceptance readout takes it
            from here, so the pipeline loads it once and never re-loads
            (§1.4 单一真相源).
    """

    specs: list[AnchorSpec]
    count: int | None
    unit_total: int
    rounds: tuple[int, int]
    ontology_sha256: str
    ontology: OntologyV4 | None = None


def _build_plan(config: ARDConfig, *, smoke: bool) -> _PlanContext:
    """Build the run's plan from the ontology and describe it (production path).

    The ontology is loaded once and its two derived facts (``U`` via
    :func:`~ard.core.sampling.unit_total`, the fingerprint via
    :func:`~ard.core.sampling.ontology_sha256`) are read from it; the plan
    itself comes from the one sampling entry point, so a smoke run and a full
    run differ only in the ``per_modality`` / ``count`` argument — never in a
    second construction rule (§18.1 不留负债).

    Args:
        config: Validated ARD configuration.
        smoke: ``True`` builds the reduced ``(4, 4)`` cycle-0 plan.

    Returns:
        The plan and its descriptive facts.
    """
    ontology: OntologyV4 = load_ontology_v4(config.ontology.path)
    gen_config = AnchorGenerationConfig(
        seed=config.generation.resolved_seed,
        concurrency=config.generation.concurrency,
    )
    total = unit_total(ontology)
    if smoke:
        specs = sample_anchors(ontology, gen_config, per_modality=SMOKE_PER_MODALITY)
        # A smoke plan is a subset of cycle 0, so its length is the requested
        # count and it spans one (partial) cycle.
        return _PlanContext(
            specs=specs,
            count=len(specs),
            unit_total=total,
            rounds=divmod(len(specs), total),
            ontology_sha256=ontology_sha256(ontology),
            ontology=ontology,
        )
    count = config.generation.count
    specs = sample_anchors(ontology, gen_config, count=count)
    return _PlanContext(
        specs=specs,
        count=count,
        unit_total=total,
        # ``count=None`` is one full cycle by definition (§2.2); ``plan_rounds``
        # takes the resolved number so the decomposition is always exact.
        rounds=plan_rounds(ontology, total if count is None else count),
        ontology_sha256=ontology_sha256(ontology),
        ontology=ontology,
    )


def sample_specs(config: ARDConfig) -> list[AnchorSpec]:
    """Build the run's anchor plan from the v4 ontology (the production seam).

    The ontology is loaded here — not before the checkpoint check in
    :func:`run` — so a run whose bank is already complete never pays for the
    rule's enumeration.  N comes from ``[generation] count`` and nowhere else;
    ``None`` means one full cycle (``U``), and there is no upper bound.

    Kept as a list-returning function because it is the seam existing callers
    and tests use; :func:`_build_plan` is the same construction with the plan's
    descriptive facts attached, and is what :func:`run` calls.

    Args:
        config: Validated ARD configuration.

    Returns:
        The plan's :class:`~ard.core.types.AnchorSpec` objects, in plan order.

    Raises:
        OntologySchemaError: If the ontology is not a readable v4 document.
        SamplingError: If the ontology cannot produce the rule's coordinate set.
    """
    return _build_plan(config, smoke=False).specs


def _injected_plan_context(
    specs: list[AnchorSpec],
    *,
    ontology: OntologyV4 | None = None,
) -> _PlanContext:
    """Describe a plan handed in through the ``generate_specs`` seam.

    The seam exists so a test can exercise the pipeline's resume/imaging
    arithmetic without materialising the real plan.  The sampling facts are
    therefore taken from the plan itself: the whole injected list is treated as
    **one cycle** (every spec is cycle 0), ``count`` is its length, and the
    fingerprint is the explicit "unknown" ``""`` rather than a fabricated digest
    (§2.2 显式即防呆).  An empty injected plan has no N (and N is never ``0``), so
    its ``count`` is ``None``.

    *ontology* is supplied by the caller only when the acceptance phase is
    enabled — that phase's structure readout is defined against the run's
    sampling space, so it needs the real ontology even though the plan was
    injected.  When it is present, ``unit_total`` is the ontology's real ``U``
    (the honest cycle length even for an injected prefix); with coverage
    disabled the ontology is not loaded at all, which keeps the pure
    resume/imaging tests free of the half-second enumeration.

    Args:
        specs: The injected plan, in plan order.
        ontology: The run's ontology when the acceptance phase needs it, else
            ``None``.

    Returns:
        The plan and the described facts.
    """
    ordered = list(specs)
    if ontology is not None:
        total = unit_total(ontology)
    else:
        total = len(ordered) if ordered else 1
    if ordered:
        count: int | None = len(ordered)
        rounds = divmod(len(ordered), total)
    else:
        # An empty injected plan requests nothing; ``count`` must not be ``0``
        # (no run has N = 0), so it is the "unspecified" ``None`` and the
        # acceptance readout falls back to the ontology's ``U``.  This shape is
        # reachable only through the test seam, whose point is to take the
        # pipeline's no-work branch without loading a plan.
        count = None
        rounds = (0, 0)
    return _PlanContext(
        specs=ordered,
        count=count,
        unit_total=total,
        rounds=rounds,
        ontology_sha256="",
        ontology=ontology,
    )


def _write_coverage_report(report: acceptance.AcceptanceReport, results_dir: Path) -> Path:
    """Write ``coverage.json`` + ``coverage.md`` into *results_dir*; return the JSON path."""
    coverage_json = results_dir / "coverage.json"
    coverage_json.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (results_dir / "coverage.md").write_text(acceptance.render_markdown(report), encoding="utf-8")
    return coverage_json


def _coordinate_label(meta: JsonObject) -> str:
    """A stable, human-readable rendering of one anchor coordinate."""
    return ", ".join(f"{key}={meta[key]!r}" for key in sorted(meta))


def _missing_plan_coordinates(
    plan: list[AnchorSpec],
    records: JsonObjectList,
) -> list[AnchorSpec]:
    """The planned specs whose id is absent from *records*, in plan order.

    Identity, not position: a bank missing a *middle* coordinate is not a short
    prefix of the plan, and only the id says which coordinate is missing.
    """
    present = {record.get("id") for record in records}
    return [spec for spec in plan if spec.id not in present]


def _missing_coordinates_warning(missing: list[AnchorSpec]) -> str:
    """One WARNING naming the planned coordinates the bank does not hold.

    The list is capped so a wholly failed run cannot paste 1,826 coordinates
    into the log; the count is always exact.
    """
    shown = missing[:_MISSING_COORDINATE_LIMIT]
    listed = "; ".join(f"{spec.id} ({_coordinate_label(spec.anchor_meta)})" for spec in shown)
    if len(missing) > _MISSING_COORDINATE_LIMIT:
        listed += f"; … and {len(missing) - _MISSING_COORDINATE_LIMIT} more"
    return (
        f"the bank is missing {len(missing)} planned coordinate(s), so within_rule is "
        f"reported false: {listed}. results/coverage.json and manifest.json describe the "
        "same artifact — rerun to fill them (a resume asks only for coordinates absent "
        "from the bank)."
    )


def _run_acceptance(
    config: ARDConfig,
    plan: list[AnchorSpec],
    records: JsonObjectList,
    output_path: Path,
    plan_context: _PlanContext,
) -> dict[str, Any] | None:
    """Write ``results/coverage.json`` + ``coverage.md`` and return the manifest pointer.

    Runs after generation and before the manifest is written.  The report is the
    **structure readout** and nothing else: zero model calls, zero numpy, no
    endpoint and no target set, so it is always published.

    The structure readout compares the plan against **this run's own sampling
    space** (:func:`ard.core.acceptance.structure_readout`), so it is handed the
    ontology the pipeline already loaded and the run's own ``count``:

    * the ontology comes from *plan_context* — one load per run, the single
      source of ``U`` / ``K`` / ``V`` (§1.4).  It is present whenever this
      function can be reached (coverage enabled ⇒ :func:`run` loaded it);
    * ``count`` is ``None`` only for "one full cycle" (the readout applies ``U``
      itself).  A ``per_modality`` smoke plan is *not* one full cycle, so
      :attr:`_PlanContext.count` is its length (8) and the smoke readout is
      measured against 8, not against 1,826.

    Returns:
        The manifest's ``acceptance`` section, or ``None`` when the phase is
        disabled (nothing was written).
    """
    if not config.coverage.enabled:
        return None

    output_dir = output_path.parent
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if plan_context.ontology is None:
        # Coverage is enabled, so ``run`` loaded the ontology for this plan
        # (including on the injected-plan seam); reaching here means a caller
        # bypassed that wiring.  Refusing beats inventing an expectation set.
        raise ConfigError(
            "the acceptance phase is enabled but the run's ontology is not "
            "available: the structure readout is defined against the run's own "
            "sampling space and cannot be computed without it"
        )
    structure = acceptance.structure_readout(
        [spec.anchor_meta for spec in plan],
        ontology=plan_context.ontology,
        count=plan_context.count,
    )
    warnings: list[str] = []
    # The structure readout describes the *plan*; it must not claim the artifact
    # is within the rule while the bank is short a planned coordinate.  A
    # plan can be perfectly rule-conformant and still have an anchor that never
    # reached the library (a failed generation the resume did not retry — the
    # bug this fixes — or one that failed again).  Reconcile the two by identity
    # and name what is missing instead of only flipping the flag.
    missing = _missing_plan_coordinates(plan, records)
    if missing:
        structure = structure.model_copy(update={"within_rule": False})
        warnings.append(_missing_coordinates_warning(missing))
    mismatch = acceptance.structure_mismatch(structure)
    if mismatch is not None:
        warnings.append(mismatch)

    report = acceptance.AcceptanceReport(structure=structure, warnings=warnings)
    coverage_json = _write_coverage_report(report, results_dir)
    logger.info("Acceptance readout written to %s: structure readout only.", coverage_json)
    return {
        "coverage_json": "results/coverage.json",
        "coverage_md": "results/coverage.md",
    }


def run(
    config: ARDConfig,
    *,
    image_dir: str | None = None,
    generate_specs: SpecSampler | None = None,
    smoke: bool = False,
) -> Path:
    """Run the ARD anchor generation pipeline.

    Args:
        config: Validated ARD configuration.
        image_dir: Optional path to image directory. If provided, image-modality
            anchors resolve their picture under ``<image_dir>/<visual_domain>/``
            (see :mod:`ard.domain.image_store`): the image is addressed by the
            anchor's own coordinate instead of being sampled from a flat pool.
            A domain whose own directory holds no usable image **reuses** one
            from the tree-wide pool instead of losing its anchors, so a tree
            with few pictures (even a single one) still yields a complete run;
            only a tree with no usable image at all is refused — naming the
            domains left without a picture, the directories they would be read
            from and the affected anchor count — unless
            ``[images] skip_missing_images`` is true (then those anchors are
            skipped, WARNING-logged and declared in ``manifest.json``).
            Whether the selected picture is transcoded to JPEG on the way into
            ``<output_dir>/images`` is ``[images] convert`` (§10.1) — not an
            argument of this function.
        generate_specs: Optional override for the plan builder, defaulting to
            the ontology-backed rule.  Tests inject a small deterministic plan
            here so the resume arithmetic can be exercised without materialising
            a full plan or loading the ontology.  An injected plan is described
            as **one cycle** (every spec is round 0) and its ontology fingerprint
            is recorded as ``""``; it is not used to implement ``--smoke`` — a
            smoke run goes through the real builder with ``per_modality=(4, 4)``.
        smoke: Run-boundary flag (``--smoke``, constitution §10.1).  The
            **same** construction rule materialises a reduced plan (the first
            :data:`SMOKE_PER_MODALITY` units of each modality from cycle 0's
            shuffle order — 8 of one full cycle) so a clone can see an artifact
            quickly.  A smoke run is tagged in three places: the run directory
            gets ``_smoke``, the log carries a WARNING, and ``manifest.json``
            declares ``smoke: true`` with its planned count against one full
            cycle.  ``False`` (the default) changes nothing about a full run.

    Returns:
        Path to the output directory.

    Raises:
        ConfigError: If a required LLM endpoint field (``api_base`` /
            ``model_name``) is unset, if ``--image-dir`` holds no usable image
            at all while ``[images] skip_missing_images`` is false, or if the
            bank already holds a record that does not sit at its own position in
            this run's plan — a foreign id or a coordinate that differs from the
            plan's coordinate at that id (a different plan would mix two
            datasets in one run directory).  All are checked before the output
            directory is created, so a refused run leaves no side effect behind
            (§2.3).
    """
    if smoke:
        logger.warning(
            "SMOKE RUN (--smoke): a reduced plan will be generated; the artifact is "
            "deliberately incomplete and must not be delivered as a dataset."
        )

    # ── Credential boundary check (§2.3 边界校验即防呆) ─────────────────────
    # Done *before* the output directory exists: a config without an endpoint
    # used to explode later inside ChatAPIConfig, after a half-built output dir
    # (and a config snapshot) had already been written.  Refusing here keeps the
    # failure readable and side-effect free, and gives the pipeline plain ``str``
    # values to hand to the clients instead of ``str | None``.
    #
    # The check is deliberately unconditional and sits first, ahead of **every**
    # side effect — including the output directory.  One consequence is
    # intended: a run whose bank is already complete ("nothing left to do, just
    # re-run/inspect it") is refused too when credentials are missing, even
    # though that path would never have contacted an endpoint.  That is a
    # deliberate behaviour change: fail before touching anything, with a
    # field-level actionable message, rather than create the directory first and
    # die later (the old order left a half-built output directory with a config
    # snapshot and ``logs/`` behind).  If "re-run a finished bank without
    # credentials" is ever needed, restructure the side-effect order — do not
    # move this guard down.
    input_endpoint = config.input_generator.resolved_endpoint()
    target_endpoint = config.target_model.resolved_endpoint()

    # ── Plan (v5 cycle-shuffle rule) — built before any side effect ─────────
    # The ontology parsing layer refuses a schema it cannot read, so a damaged
    # file stops the run here instead of silently sampling 0 anchors further down
    # (§2.3 边界校验即防呆).
    #
    # N comes from ``[generation] count`` and nowhere else; ``None`` means one
    # full cycle.  There is no upper bound — a large N simply runs more cycles
    # (:func:`~ard.core.sampling.plan_rounds` decomposes it for the INFO line
    # below, it never refuses).  An anchor id is the plan *position*
    # (``run_key + cycle + position``), independent of N, so the first N ids of
    # a larger plan are exactly the smaller plan's — the property resume relies
    # on.  It is **not** a content fingerprint, which is why the resume guard
    # compares coordinates per id (see :func:`_refuse_foreign_records_on_resume`).
    #
    # The plan **is** the target: the checkpoint/resume path asks for the plan
    # entries whose stable id is not in the bank yet, and a middle coordinate
    # abandoned by an earlier run is requested again instead of being shadowed by
    # a later, already-written one (see the ``pending_specs`` comment below).
    #
    # It is built *before* the output directory exists so the image-addressing
    # check below runs on it and can still refuse before any side effect.
    gen_config = AnchorGenerationConfig(
        seed=config.generation.resolved_seed,
        concurrency=config.generation.concurrency,
    )
    rng = random.Random(gen_config.seed)
    if generate_specs is not None:
        # The injected plan replaces the sampler, but the acceptance phase (when
        # enabled) still needs the run's sampling space: load the ontology once
        # here — not in ``_run_acceptance`` — so there is one ontology per run
        # (§1.4).  With coverage disabled nothing needs it, so nothing is loaded.
        injected_ontology = (
            load_ontology_v4(config.ontology.path) if config.coverage.enabled else None
        )
        plan_context = _injected_plan_context(generate_specs(config), ontology=injected_ontology)
    else:
        # Smoke and full runs go through the very same construction rule; only
        # the count / per_modality argument differs, never a second rule.
        plan_context = _build_plan(config, smoke=smoke)
    plan = plan_context.specs
    target_count = len(plan)
    plan_id = PlanIdentity.of(
        plan,
        ontology_sha256=plan_context.ontology_sha256,
        seed=gen_config.seed,
        count=plan_context.count,
        unit_total=plan_context.unit_total,
    )
    full_cycles, last_cycle_size = plan_context.rounds
    logger.info(
        "Plan: %d coordinate(s) = %d full cycle(s) + %d coordinate(s) in the last "
        "cycle; one full cycle is %d unit(s) (density %.4f). Plan identity: %s "
        "(algorithm=%s, version=%d, sampling=%s).",
        target_count,
        full_cycles,
        last_cycle_size,
        plan_context.unit_total,
        target_count / plan_context.unit_total,
        plan_id.digest,
        plan_id.algorithm,
        plan_id.version,
        plan_id.sampling,
    )
    if smoke:
        logger.warning(
            "SMOKE RUN: plan reduced to %d of %d anchors (%d text + %d image, the "
            "first of each modality in cycle 0's shuffle order) by the standard "
            "construction rule. results/coverage.json measures this artifact against "
            "its own %d-anchor count — within_rule=true is the expected smoke "
            "readout; the artifact is still deliberately incomplete. Do not deliver "
            "it; run without --smoke for the full dataset.",
            target_count,
            plan_context.unit_total,
            SMOKE_PER_MODALITY[0],
            SMOKE_PER_MODALITY[1],
            target_count,
        )

    # ── Output paths (still no side effect) + resume arithmetic ────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = _resolve_run_directory(config, timestamp=timestamp, smoke=smoke)
    output_path = output_dir / "anchor_bank.jsonl"

    # Overwrite is *decided* here but performed only after every boundary check
    # has passed: an unusable image directory must not destroy the previous
    # bank.  Opting into overwrite means "replaceable by a run that succeeds",
    # not "delete first and find out later" (§3.3 预授权退路).
    overwrite_existing = config.output.overwrite and output_path.exists()
    existing_records = [] if overwrite_existing else read_anchor_bank(output_path)
    existing_ids = {
        record["id"] for record in existing_records if isinstance(record.get("id"), str)
    }
    existing_count = len(existing_ids)
    remaining = target_count - existing_count

    # ── Resume guard (§2.3, still before any side effect) ───────────────────
    # Appending is allowed exactly when every record already on disk sits at its
    # own plan position in *this* plan: same id, same coordinate.  The recorded
    # ``plan_identity`` (from the finished run's ``manifest.json`` or, when the
    # previous run was interrupted, from the intermediate progress record) is
    # read for the diagnostic and kept as an audit trail, but the decision is
    # made by comparing ids *and* coordinates — see the guard's docstring for why
    # an id-subset test would let a smoke plan into a full-plan directory.
    recorded_plan_identity = _recorded_plan_identity(output_dir / "manifest.json")
    _refuse_foreign_records_on_resume(
        output_dir,
        plan_id,
        plan,
        existing_records,
        recorded_plan_identity or _recorded_progress_identity(output_dir),
    )

    # ── Image addressing boundary check (§2.3, before the output dir) ──────
    # Every image-modality anchor prefers a picture under
    # ``<image_dir>/<visual_domain>/`` and otherwise reuses one from the
    # tree-wide pool, so a required domain is only *really* missing when the
    # tree holds no usable image at all.  That state is refused here — naming
    # the domains left without a picture, the directories they would be read
    # from and how many anchors they affect — unless the user explicitly opted
    # into skipping those anchors ([images] skip_missing_images = true, §3.3
    # 预授权退路).
    # Only the anchors still to be generated are checked: a completed bank needs
    # no image lookup at all.
    #
    # "Still to be generated" is decided by **coordinate identity**, not by a
    # count: the bank stores each coordinate's stable id, so the shortfall is
    # ``plan - bank``.  Slicing the tail after ``N`` records assumed the bank was
    # a prefix of the plan; a run that abandoned a *middle* anchor wrote 1,825 of
    # 1,826 records, and the resume then asked for ``plan[1825:]`` — the plan's
    # last entry, already on disk — so the abandoned coordinate was never
    # retried while ``coverage.json`` still read ``within_rule=true`` against the
    # full plan — a green readout decoupled from the library.  Identity makes
    # every missing coordinate pending again, in plan order.
    pending_specs = [spec for spec in plan if spec.id not in existing_ids]
    # ``[images] convert`` decides both the accepted input set and the bytes
    # that land in <output_dir>/images, so it is read here from the config
    # rather than from a CLI flag (§10.1: one source of truth per parameter).
    image_extensions = CONVERTABLE_EXTENSIONS if config.images.convert else SUPPORTED_EXTENSIONS
    # The round every spec belongs to, taken from its **plan index** — the
    # position the spec's id names — and not from the id string, which is opaque
    # to this module.  ``U`` comes from the plan's own context (the ontology's
    # real ``U``; the injected seam without an ontology falls back to the plan's
    # length, i.e. one cycle).
    cycle_of = {spec.id: index // plan_context.unit_total for index, spec in enumerate(plan)}
    image_resolutions: dict[int, DomainImageResolution] = {}
    # ``(cycle, visual_domain) -> source file``, the resolution readout the copy
    # step and the manifest both consume.
    selected_sources: dict[tuple[int, str], Path] = {}
    skipped_domains: dict[str, StringList] = {}
    if image_dir is not None and pending_specs:
        image_root = Path(image_dir)
        image_specs = sum(
            1 for spec in pending_specs if isinstance(spec.anchor_meta.get("visual_domain"), str)
        )
        if image_specs and not image_root.is_dir():
            raise ConfigError(
                f"image directory not found: {image_root} — it is required for the "
                f"{image_specs} image-modality anchor(s) this run would generate. "
                f"Point --image-dir at an existing directory laid out as "
                f"{VISUAL_DOMAIN_LAYOUT}."
            )
        # One call per round: a single call describes exactly one ``cycle``, so
        # handing it specs from several cycles would pin every round to that
        # round's file.  Group by ``cycle_of``, then resolve each group.
        by_cycle: dict[int, list[AnchorSpec]] = {}
        for spec in pending_specs:
            if isinstance(spec.anchor_meta.get("visual_domain"), str):
                by_cycle.setdefault(cycle_of[spec.id], []).append(spec)
        image_resolutions = {
            cycle: resolve_domain_images(
                image_root,
                cycle_specs,
                seed=gen_config.seed,
                cycle=cycle,
                extensions=image_extensions,
            )
            for cycle, cycle_specs in sorted(by_cycle.items())
        }
        for cycle, resolution in image_resolutions.items():
            for domain, source in resolution.selected.items():
                selected_sources[(cycle, domain)] = source
        # The boundary readout is over the whole run, not one round.  Since the
        # fallback covers a domain whose own directory holds no image, a domain
        # lands here only when the image tree holds **no usable image at all**;
        # the merged view is what the refusal message and the skip filter
        # reason about.
        merged_missing: dict[str, StringList] = {}
        for resolution in image_resolutions.values():
            for domain, anchor_ids in resolution.missing.items():
                merged_missing.setdefault(domain, []).extend(anchor_ids)
        if merged_missing:
            merged_selected = {
                domain: source for (_, domain), source in sorted(selected_sources.items())
            }
            merged_counts: dict[str, int] = {}
            merged_fallback: set[str] = set()
            merged_fallback_anchors = 0
            merged_pool = 0
            for resolution in image_resolutions.values():
                merged_counts.update(resolution.candidate_counts)
                merged_fallback |= resolution.fallback
                merged_fallback_anchors += resolution.fallback_anchor_count
                merged_pool = max(merged_pool, resolution.pool_candidate_count)
            merged = DomainImageResolution(
                selected=merged_selected,
                missing=merged_missing,
                cycle=min(image_resolutions),
                candidate_counts=merged_counts,
                fallback=merged_fallback,
                pool_candidate_count=merged_pool,
                fallback_anchor_count=merged_fallback_anchors,
            )
            if not config.images.skip_missing_images:
                raise ConfigError(
                    _missing_image_message(
                        image_root,
                        merged,
                        len(pending_specs),
                        image_extensions,
                    )
                )
            skipped_domains = merged_missing
            available = set(merged_selected)
            pending_specs = [
                spec
                for spec in pending_specs
                if not isinstance(spec.anchor_meta.get("visual_domain"), str)
                or spec.anchor_meta["visual_domain"] in available
            ]

    # The plan this run actually owns: every planned coordinate that is either
    # already in the bank or about to be generated.  Domain-skipped coordinates
    # are removed, so both the resume arithmetic and the acceptance structure
    # readout stay consistent with what is on disk instead of claiming a
    # rule-conformant plan.  Built by identity (same reason as ``pending_specs``
    # above), so the readout describes the artifact this run owns rather than the
    # first N plan entries.
    specs = pending_specs
    owned_ids = existing_ids | {spec.id for spec in specs}
    effective_plan = [spec for spec in plan if spec.id in owned_ids]

    # ── Output directory (first side effect) ───────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── File logging (must happen after output_dir exists) ──────────────
    configure_file_logging(output_dir)

    # ── Overwrite / checkpoint-resume ──────────────────────────────────────
    if overwrite_existing:
        output_path.unlink()
        logger.info("Overwrite mode: cleared existing anchor bank at %s", output_path)

    # Archive the merged config in the output directory for reproducibility.
    # It is written in the project's own config format (§8.5) so the snapshot
    # is itself a usable ``--config`` — the output directory then describes the
    # run without the user having to find the original config back (§10.1).
    # Credentials are masked first (§2.3 — the output directory is a boundary
    # users share); the same redacted dict feeds the manifest's `config`
    # section below, so both sinks are covered by this single definition.
    config_info = _redact_secrets(config.model_dump(mode="json"))
    config_snapshot_path = output_dir / "config.toml"
    config_snapshot_path.write_text(_config_snapshot_toml(config_info), encoding="utf-8")
    logger.info("Merged config snapshot written to %s", config_snapshot_path)

    # Skipping is a §3.2 透明退化: one WARNING per dropped anchor (which domain,
    # which sample) — declared again, in aggregate, in manifest.json below.
    if image_dir is not None:
        for domain in sorted(skipped_domains):
            for anchor_id in skipped_domains[domain]:
                logger.warning(
                    "Skipping anchor %s: the image tree under %s holds no usable "
                    "picture; visual_domain %r therefore has none either "
                    "([images] skip_missing_images = true).",
                    anchor_id,
                    image_dir,
                    domain,
                )

    if not specs:
        if remaining > 0:
            logger.warning(
                "Every remaining anchor (%d) was skipped because the image tree under %s "
                "holds no usable picture; the visual_domains that need one therefore "
                "have none either. Nothing to generate.",
                remaining,
                image_dir,
            )
        else:
            logger.info(
                "Already have %d anchors (plan: %d), skipping generation.",
                existing_count,
                target_count,
            )
        all_records = read_anchor_bank(output_path)
        manifest = build_manifest_from_records(all_records, output_dir, config_info)
        _declare_plan_identity(manifest, plan_id)
        _declare_plan_readout(
            manifest,
            plan=plan,
            count=plan_context.count,
            unit_total=plan_context.unit_total,
            rounds=plan_context.rounds,
            written=len(all_records),
            seed=gen_config.seed,
            ontology_fingerprint=plan_context.ontology_sha256,
            smoke=smoke,
        )
        if smoke:
            _declare_smoke(
                manifest,
                plan_size=target_count,
                unit_total=plan_context.unit_total,
                output_dir=output_dir,
            )
        if image_dir is not None:
            _declare_images(
                manifest,
                image_dir=image_dir,
                config=config,
                records=all_records,
                plan=plan,
                cycle_of=cycle_of,
                seed=gen_config.seed,
            )
        # No generation happens on this path, so there are no run counters to
        # report.  A manifest that already names this plan is therefore left
        # byte-for-byte as it is rather than overwritten with a clean-looking
        # one (§3.2 — a missing field must not read as "healthy", and a finished
        # run must not be restated as unfinished).  The intermediate progress
        # record is dropped either way: the manifest is now the current word.
        acceptance_pointer = _run_acceptance(
            config, effective_plan, all_records, output_path, plan_context
        )
        if acceptance_pointer is not None:
            manifest["acceptance"] = acceptance_pointer
        _refresh_manifest_for_no_generation(
            manifest, output_dir / "manifest.json", recorded_plan_identity
        )
        _clear_progress_record(output_dir)
        logger.info("Done! Output: %s", output_dir)
        logger.info("  Total anchors: %d", len(all_records))
        return output_dir

    logger.info(
        "Found %d existing anchors, generating %d more (plan: %d)...",
        existing_count,
        len(specs),
        target_count,
    )

    # ── API clients ───────────────────────────────────────────────────────
    input_client = ChatAPIClient(
        ChatAPIConfig(
            api_base=input_endpoint.api_base,
            model_name=input_endpoint.model_name,
            api_key=input_endpoint.api_key,
            temperature=config.input_generator.temperature,
            max_tokens=config.input_generator.max_tokens,
            connect_timeout=config.input_generator.connect_timeout,
            first_token_timeout=config.input_generator.first_token_timeout,
            inter_token_timeout=config.input_generator.inter_token_timeout,
            max_retries=config.input_generator.max_retries,
            retry_on_timeout=config.input_generator.retry_on_timeout,
            # Explicit, not inherited from ChatAPIConfig's default (§2.2 显式即防呆).
            #
            # The key is **always** emitted, so this line is the only
            # thing standing between the input generator and the server-side
            # template default ("``enable_thinking`` undefined" is read as *ON*).
            # Omitting it would silently flip question generation back into
            # reasoning mode, and ``[input_generator]`` deliberately has no
            # ``enable_thinking`` field to fall back on — so "what the input
            # generator does" has to be readable here, at the construction site,
            # rather than inferred from a dataclass default.
            #
            # The value is fixed at ``False`` because question generation has no
            # use for reasoning: the generated user turn is stored verbatim as
            # the anchor's question, so reasoning tokens would spend
            # ``max_tokens`` first and dilute what is meant to be a clean user
            # utterance.  Only ``[target_model]`` is configurable: reasoning
            # distils into the student model, a question does not.
            enable_thinking=False,
        )
    )
    target_client = ChatAPIClient(
        ChatAPIConfig(
            api_base=target_endpoint.api_base,
            model_name=target_endpoint.model_name,
            api_key=target_endpoint.api_key,
            temperature=config.target_model.temperature,
            max_tokens=config.target_model.max_tokens,
            connect_timeout=config.target_model.connect_timeout,
            first_token_timeout=config.target_model.first_token_timeout,
            inter_token_timeout=config.target_model.inter_token_timeout,
            max_retries=config.target_model.max_retries,
            retry_on_timeout=config.target_model.retry_on_timeout,
            enable_thinking=config.target_model.enable_thinking,
        )
    )

    # ── Generate anchors (unified flow) ───────────────────────────────────
    # The plan was sampled and the images were resolved above.
    #
    # Step 2: place the selected image of every ``(round, visual_domain)`` into
    # the output tree and assign it to the anchors of that round and domain.
    # The files were already validated (existence, missing-domain refusal) before
    # the output directory existed; this step only copies bytes.  v5 rotates a
    # domain's picture by round, so several rounds may share one source file when
    # the domain holds few images: the source is copied **once** and every
    # ``(cycle, domain)`` that selected it reuses the placed path.
    if selected_sources and specs:
        rel_by_cycle_domain: dict[tuple[int, str], str] = {}
        rel_by_source: dict[Path, str] = {}
        for (cycle, domain), source in sorted(selected_sources.items()):
            placed_rel = rel_by_source.get(source)
            if placed_rel is not None:
                rel_by_cycle_domain[(cycle, domain)] = placed_rel
                continue
            if config.images.convert:
                placed = convert_and_copy_images([source], output_dir, subdir=domain)
            else:
                placed = copy_images_to_output([source], output_dir, subdir=domain)
            if not placed:
                raise ConfigError(
                    f"image {source} for visual_domain {domain!r} could not be "
                    f"converted/copied into {output_dir / 'images' / domain}. "
                    f"Replace it with a readable image that the run may reuse."
                )
            rel_by_source[source] = placed[0]
            rel_by_cycle_domain[(cycle, domain)] = placed[0]
        _assign_images_by_domain(specs, rel_by_cycle_domain, cycle_of, output_dir, rng)

    # Every record this run writes declares its own image state, whether or not
    # this invocation addressed a picture.  Stamping only inside the copy step
    # above coupled the value to the *invocation*: a finishing run whose pending
    # set held nothing but text anchors selected no source, skipped the copy
    # step, and wrote those records with both fields absent — while the README
    # contract says a text-state coordinate carries ``has_image: false`` /
    # ``image_count: 0``.  The fields belong to the coordinate, so they are
    # stamped for the whole pending set from each spec's own turns (§1.4).
    stamp_image_bookkeeping(specs)

    # ── Early plan-identity record (§2.4, before the first endpoint call) ──
    # The plan is fixed and the config snapshot exists, but the authoritative
    # ``manifest.json`` is written only after generation.  A run interrupted
    # between here and there (the normal outcome on the 1,826-anchor path, where
    # ~25% of coordinates are abandoned and the run is resumed) would leave a
    # bank with no plan name at all.  So bind the run directory to the plan
    # *now*, in a record that is explicitly not the manifest — different file
    # name, different schema key, ``status: "in_progress"`` and no run health.
    # It is refreshed by a later invocation and removed once the manifest is
    # written, so a finished directory carries exactly one declaration.
    _write_progress_record(
        output_dir,
        _build_progress_record(
            identity=plan_id,
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            existing=existing_count,
            new=len(specs),
            written=len(specs),
            smoke=smoke,
            output_dir=output_dir,
        ),
    )

    # Step 3: Generate all anchors via the unified generator
    #
    # Reasoning observability: snapshot the API client's reasoning
    # counters before and after generation.  Reasoning tokens arrive as
    # ``delta.reasoning``, are persisted as ``targets[0].output.reasoning``, and
    # also consume ``max_tokens`` first — so a run whose budget was eaten by
    # thinking produces *empty* target answers, a failure that used to be
    # visible only as scattered per-request logs.  This delta makes the rate
    # visible once per run (§3.2 透明退路: an affected result must be announced).
    reasoning_before = api_client_reasoning_stats()
    generation_stats = AnchorGenerationStats(requested=len(specs))
    new_anchors = generate_text_anchors(
        specs=specs,
        input_client=input_client,
        target_client=target_client,
        input_model_name=input_endpoint.model_name,
        target_model_name=target_endpoint.model_name,
        concurrency=gen_config.concurrency,
        output_path=output_path,
        backpressure_threshold=config.generation.backpressure_threshold,
        backpressure_cooldown=config.generation.backpressure_cooldown,
        stats=generation_stats,
    )
    reasoning_delta = _log_target_model_reasoning_stats(
        api_client_reasoning_stats(), reasoning_before, len(new_anchors)
    )

    # ── Build manifest from ALL anchors (existing + new) ──────────────────
    all_records = read_anchor_bank(output_path)
    _prune_abandoned_images(specs, output_dir, all_records)
    manifest = build_manifest_from_records(all_records, output_dir, config_info)
    _declare_plan_identity(manifest, plan_id)
    _declare_plan_readout(
        manifest,
        plan=plan,
        count=plan_context.count,
        unit_total=plan_context.unit_total,
        rounds=plan_context.rounds,
        written=len(all_records),
        seed=gen_config.seed,
        ontology_fingerprint=plan_context.ontology_sha256,
        smoke=smoke,
    )
    # Publish the run's failures next to the anchors that survived, so a short
    # bank can never be mistaken for a healthy one (§3.2).
    with_generation_report(
        manifest,
        stats=generation_stats.to_manifest_dict(),
        failures=reasoning_delta,
    )
    acceptance_pointer = _run_acceptance(
        config, effective_plan, all_records, output_path, plan_context
    )
    if acceptance_pointer is not None:
        manifest["acceptance"] = acceptance_pointer
    if smoke:
        _declare_smoke(
            manifest,
            plan_size=target_count,
            unit_total=plan_context.unit_total,
            output_dir=output_dir,
        )
    if image_dir is not None:
        _declare_images(
            manifest,
            image_dir=image_dir,
            config=config,
            records=all_records,
            plan=plan,
            cycle_of=cycle_of,
            seed=gen_config.seed,
        )
    write_manifest(
        _declare_manifest_status(manifest, status=MANIFEST_STATUS_COMPLETE),
        output_dir / "manifest.json",
    )
    # The authoritative declaration now exists (and names the same plan), so the
    # intermediate record has served its purpose; leaving its stale
    # ``in_progress`` next to a finished manifest would be needless ambiguity.
    _clear_progress_record(output_dir)

    total = len(all_records)
    logger.info("Done! Output: %s", output_dir)
    logger.info(
        "  Total anchors: %d (%d existing + %d new)",
        total,
        existing_count,
        total - existing_count,
    )
    return output_dir


def _relative_image_path(raw: str, output_dir: Path) -> str | None:
    """Address *raw* as a posix path relative to *output_dir*, or ``None``.

    Two shapes meet in :func:`_prune_abandoned_images`: a spec carries an
    absolute path stamped by :func:`_assign_images_by_domain`, while a bank
    record carries the relative ``images/<domain>/<file>`` form written by
    :func:`ard.domain.text_anchor._convert_images_to_paths`.  Both must key the
    same file for the ownership lookup to protect it, so the normalisation
    lives here instead of at each call site.

    A path that does not address a file inside *output_dir* returns ``None``:
    this run must never reason about a file it did not put there.

    Args:
        raw: The stored image reference (absolute or output-relative).
        output_dir: The run directory the reference is addressed against.

    Returns:
        The posix path relative to *output_dir*, or ``None`` when *raw* does
        not address a file inside it.
    """
    path = Path(raw)
    if path.is_relative_to(output_dir):
        return path.relative_to(output_dir).as_posix()
    if path.is_absolute():  # pragma: no cover - not addressed inside this run
        return None
    return path.as_posix()


def _record_image_paths(record: JsonObject, output_dir: Path) -> Iterator[str]:
    """Every image file one bank record references, as paths under *output_dir*.

    The persisted image part is ``{"type": "image", "image":
    "images/<domain>/<file>"}``; the inline ``image_url`` form holds a base64
    data URI, which names no file and is therefore not a reference.  The walk
    follows the parsed record rather than a fixed message schema, so a record
    shape the current writer did not produce cannot silently drop out of the
    protection the way a hand-kept list of fields would.

    Args:
        record: One parsed ``anchor_bank.jsonl`` record.
        output_dir: The run directory the references are addressed against.

    Yields:
        Each referenced file, normalised by :func:`_relative_image_path`.
    """
    stack: list[Any] = [record]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            image = value.get("image")
            if isinstance(image, str):
                rel = _relative_image_path(image, output_dir)
                if rel is not None:
                    yield rel
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)


def _prune_abandoned_images(
    specs: list[AnchorSpec],
    output_dir: Path,
    records: JsonObjectList,
) -> int:
    """Delete the pictures of anchors that were ultimately abandoned.

    The copy of a run's images happens *before* generation
    (:func:`_assign_images_by_domain`), one file per ``visual_domain`` shared by
    every anchor of that domain.  An anchor that is then abandoned — generation
    failed, retries exhausted — leaves its picture in ``<output_dir>/images/``
    with nothing in ``anchor_bank.jsonl`` pointing at it, so the artifact's
    image count would no longer match its anchor count.  Reconcile the two here.

    **When is a file deletable?** Only when both hold:

    * **this run placed or reused it.**  The candidate set is built from the
      pending specs' own ``image_path``, and every one of those was stamped
      with *output_dir* — a file the run never touched is never a candidate;
    * **no record of the final bank references it.**  The final bank is the
      pre-existing records *plus* the records this run wrote, and the ownership
      map is built from all of them, so an anchor written by an earlier run
      protects its picture exactly like one written now.

    The second half is what keeps a **resume** run safe: a pending anchor
    reuses the picture an earlier run placed (``force=False`` skips the copy)
    and is then abandoned, while the pre-existing record still points at that
    file.  Reading ownership from the pending specs alone would delete a file
    the artifact still references and leave a dangling path in the bank.

    A picture shared by a domain's other anchors is therefore never touched:
    those anchors are records too.  A failed unlink is a WARNING, never a run
    failure (§3.2 透明退路: announce the degradation, do not crash the run over
    housekeeping).

    Args:
        specs: The run's pending specs, after image assignment.
        output_dir: The run directory; the specs' image paths are relative to it.
        records: Every anchor in the bank after generation — the pre-existing
            ones read back from disk plus this run's new records.

    Returns:
        How many files were removed.
    """
    usage: dict[str, set[str]] = {}
    for spec in specs:
        for turn in spec.turns:
            if not turn.image_path:
                continue
            rel = _relative_image_path(turn.image_path, output_dir)
            if rel is None:  # pragma: no cover - not addressed inside this run
                continue
            usage.setdefault(rel, set()).add(spec.id)
    if not usage:
        return 0

    # Ownership of a candidate file: the pending spec ids that asked for it ...
    owners = {rel: set(ids) for rel, ids in usage.items()}
    # ... plus the id of every record of the final bank that references it.  A
    # pre-existing record has no spec in *specs*, so its id can only come from
    # the record itself — without this pass it could never protect the picture
    # it points at (the resume defect this function must not reintroduce).
    for record in records:
        record_id = record.get("id")
        if not isinstance(record_id, str):
            continue
        for rel in _record_image_paths(record, output_dir):
            if rel in owners:
                owners[rel].add(record_id)

    kept = {record["id"] for record in records if isinstance(record.get("id"), str)}
    removed = 0
    for rel, referencing in sorted(owners.items()):
        if referencing & kept:
            continue  # a surviving record of the final bank still uses this file
        try:
            (output_dir / rel).unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning(
                "Could not remove image %s of abandoned anchor(s) %s: %s. "
                "It is no longer referenced; delete it by hand if the run "
                "directory must match anchor_bank.jsonl.",
                rel,
                ", ".join(sorted(referencing)),
                exc,
            )
            continue
        removed += 1
        logger.info(
            "Removed image %s: every anchor that referenced it was abandoned (%s).",
            rel,
            ", ".join(sorted(referencing)),
        )
    return removed
