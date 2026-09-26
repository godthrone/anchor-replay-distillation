"""Main pipeline orchestrator for ARD anchor generation.

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
from functools import partial
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
from ard.backends.coverage_wiring import TargetSet, build_metric_readout, load_target_set
from ard.backends.ontology_loader import load_ontology_v4
from ard.config import ARDConfig, ConfigError, CoverageEmbedding
from ard.core import acceptance
from ard.core.ontology import OntologyV4
from ard.core.quota import allocate_images
from ard.core.sampling import (
    EXPECTED_TOTAL,
    PLAN_IDENTITY_ALGORITHM,
    SMOKE_IMAGE_BLOCKS,
    SMOKE_SCALE,
    SMOKE_TEXT_BLOCKS,
    PlanIdentity,
    PlanScale,
    sample_anchors,
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
    resolve_domain_images,
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
# snapshot exists to record.  The accepted residual risk (sharing an output
# directory also shares the endpoint) is documented in the R7 report.

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
credential (see the R7 report for the accepted residual risk).
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
    """Log the reasoning-vs-content accounting of the target model (WP-F3).

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
    progress record): the identity is the same four-field object either way, and
    validating it in one place is what keeps "no recorded identity" a different
    fact from "identity 0" (§3.2).
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
    return PlanIdentity(
        algorithm=PLAN_IDENTITY_ALGORITHM, version=version, plan_size=plan_size, digest=digest
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

    Args:
        identity: The plan's identity (``PlanIdentity.of(plan)``).
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


def _refuse_plan_change_on_resume(
    output_dir: Path,
    identity: PlanIdentity,
    plan_ids: set[str],
    existing_records: JsonObjectList,
    recorded: PlanIdentity | None,
) -> None:
    """Refuse a resume that would append a *different* plan to the bank (§2.3).

    The guard keys on the plan's identity, never on ``[generation] seed``: the
    seed is a process-level config value that does not name a plan (two runs can
    share one seed and still build different plans, and a run directory's
    recorded seed is rewritten by every later invocation, including ones that
    generate nothing — see ``docs/algorithm.md`` §6).  The identity is the one
    the run directory already records: the finished run's ``manifest.json``
    ``plan_identity``, or — when the previous run never got that far — the
    ``plan_identity`` in :data:`PROGRESS_RECORD_FILENAME`.  Both describe the
    plan actually sampled, and both are checked before any side effect.

    A resume whose directory records no identity (a hand-assembled or
    pre-identity bank) falls back to the structural fact that proves it safe:
    every anchor id already in the bank must be part of *this* run's plan.  A
    bank holding a foreign coordinate is refused; a bank that is a subset is
    resumed with a WARNING, so "cannot verify" never silently becomes "mixed".

    Args:
        output_dir: The run directory being resumed.
        identity: This run's plan identity.
        plan_ids: Every anchor id of this run's plan.
        existing_records: The records already in the bank.
        recorded: The identity the directory already records, or ``None``.

    Raises:
        ConfigError: If records already exist for *output_dir* and either the
            recorded identity differs from this run's, or no identity is
            recorded and the bank holds an id outside this run's plan.
    """
    if not existing_records:
        return
    existing_ids = {
        record["id"] for record in existing_records if isinstance(record.get("id"), str)
    }
    if recorded is not None:
        if (
            recorded.digest == identity.digest
            and recorded.plan_size == identity.plan_size
            and recorded.version == identity.version
        ):
            return
        raise ConfigError(
            f"refusing to resume {output_dir}: the bank already on disk holds a "
            f"different plan. Recorded plan identity {recorded.digest} "
            f"(version {recorded.version}, {recorded.plan_size} coordinates); this "
            f"run's plan identity is {identity.digest} (version {identity.version}, "
            f"{identity.plan_size} coordinates). Appending would mix two datasets in "
            "one run directory while every counter stays green. Re-run with the plan "
            "that produced this bank, or give the new plan its own output.directory "
            "(or set output.overwrite = true to replace this one deliberately)."
        )
    foreign = sorted(existing_ids - plan_ids)
    if foreign:
        raise ConfigError(
            f"refusing to resume {output_dir}: its anchor_bank.jsonl holds "
            f"{len(foreign)} anchor id(s) that are not part of this run's plan "
            f"(first: {foreign[0]}), and neither manifest.json nor "
            f"{PROGRESS_RECORD_FILENAME} records a plan_identity to verify. "
            "Appending would mix two datasets in one run directory; give "
            "the new plan its own output.directory (or set output.overwrite = true "
            "to replace this one deliberately)."
        )
    if existing_ids:
        logger.warning(
            "%s has %d record(s) but no recorded plan_identity; verified every "
            "existing anchor id belongs to this run's plan, so this resume appends "
            "within one plan.",
            output_dir,
            len(existing_ids),
        )


def _declare_plan_identity(manifest: dict[str, Any], identity: PlanIdentity) -> None:
    """Record the run's plan identity in *manifest* (§7.4: self-naming field).

    One definition of the field name, both manifest sinks (the completed run and
    the no-op resume), so a reader of the artifact always finds the plan's name.
    """
    manifest["plan_identity"] = identity.as_dict()


def _declare_smoke(manifest: dict[str, Any], *, plan_size: int, output_dir: Path) -> None:
    """Declare in *manifest* that this artifact is a smoke run (§3.2 透明退路).

    Writes the machine-readable claim (``smoke: true``) next to the counts that
    prove it: what this run planned, and what a full run plans.  A reader of the
    artifact alone can tell a smoke run from a delivery.
    """
    manifest["smoke"] = True
    manifest["smoke_plan"] = {
        "run_name": output_dir.name,
        "planned_anchors": plan_size,
        "full_expected_anchors": EXPECTED_TOTAL,
        "text_blocks": SMOKE_TEXT_BLOCKS,
        "image_blocks": SMOKE_IMAGE_BLOCKS,
        "note": "SMOKE RUN — deliberately incomplete; not a deliverable.",
    }


def _missing_image_message(
    image_root: Path,
    resolution: DomainImageResolution,
    planned: int,
    extensions: set[str],
) -> str:
    """The refusal message for a plan whose visual domains have no images.

    Names **every** missing domain, its expected directory and how many anchors
    it affects (§2.3: a boundary rejection must be actionable, not a traceback
    from somewhere inside the copy loop), plus the sample count the refusal
    applies to — one call, one fix.
    """
    missing = resolution.missing
    affected = sum(len(anchor_ids) for anchor_ids in missing.values())
    required = len(resolution.selected) + len(missing)
    listed = "\n".join(
        f"  - {domain}: expected {domain_directory(image_root, domain)} "
        f"(affects {len(missing[domain])} anchor(s))"
        for domain in sorted(missing)
    )
    allowed = ", ".join(sorted(extensions))
    return (
        f"image directory {image_root} has no usable image for {len(missing)} of the "
        f"{required} visual_domain(s) this plan requires.\n"
        f"Expected layout: {VISUAL_DOMAIN_LAYOUT} — one subdirectory per visual_domain "
        f"holding at least one image ({allowed}); files directly in {image_root} "
        f"are not used.\n"
        f"{listed}\n"
        f"Total affected anchors: {affected} of {planned} planned sample(s) (each is "
        f"labelled with the missing visual_domain and cannot be given an image from "
        f"another domain).\n"
        f"Provide the missing image(s), or set [images] skip_missing_images = true to "
        f"skip those {affected} anchor(s) — each is then logged as a WARNING and the "
        f"skipped count and domains are declared in manifest.json."
    )


def _declare_images(
    manifest: dict[str, Any],
    *,
    image_dir: str,
    config: ARDConfig,
    resolution: DomainImageResolution | None,
    skipped: dict[str, StringList],
) -> None:
    """Declare the run's image addressing and any skipped anchors (§3.2/§3.3).

    A dataset with anchors missing is a different artifact from the one the
    construction rule describes, so the difference is machine-readable: the
    skipped count and the affected ``visual_domain`` values travel with the
    manifest, where a consumer reads them without parsing logs.
    """
    manifest["images"] = {
        "image_dir": str(Path(image_dir).resolve()),
        "addressing": VISUAL_DOMAIN_LAYOUT,
        "skip_missing_images": config.images.skip_missing_images,
        "resolved_visual_domains": sorted(resolution.selected) if resolution else [],
        "skipped_anchor_count": sum(len(anchor_ids) for anchor_ids in skipped.values()),
        "skipped_visual_domains": sorted(skipped),
    }


def _assign_images_by_domain(
    specs: list[AnchorSpec],
    rel_by_domain: dict[str, str],
    output_dir: Path,
    rng: random.Random,
) -> None:
    """Give every image-modality spec the image of its own ``visual_domain``.

    One domain = one group; :func:`ard.core.quota.allocate_images` is reused per
    group so the turn-filling rule (earliest eligible ``user`` turn, at most
    :data:`IMAGES_PER_ANCHOR` images) keeps a single implementation.  Text-only
    specs are stamped ``has_image=False`` and never receive an image: their
    coordinate names no visual domain, so an image would be the same
    coordinate/content mismatch the addressing exists to prevent.
    """
    groups: dict[str, AnchorSpecList] = {}
    for spec in specs:
        domain = spec.anchor_meta.get("visual_domain")
        if isinstance(domain, str) and domain in rel_by_domain:
            groups.setdefault(domain, []).append(spec)
        else:
            spec.anchor_meta["has_image"] = False
            spec.anchor_meta["image_count"] = 0
    for domain, group in groups.items():
        allocate_images(group, [rel_by_domain[domain]], IMAGES_PER_ANCHOR, rng)

    # Resolve image paths relative to output_dir for base64 encoding.
    for spec in specs:
        for turn in spec.turns:
            if turn.image_path:
                turn.image_path = str(output_dir / turn.image_path)


def sample_specs(config: ARDConfig, *, scale: PlanScale | None = None) -> list[AnchorSpec]:
    """Build the run's anchor plan from the v4 ontology (the production seam).

    The ontology is loaded here — not before the checkpoint check in
    :func:`run` — so a run whose bank is already complete never pays for the
    rule's enumeration.  The count is the plan's length: the v4 construction
    rule derives it, so no config field hands it in.

    Args:
        config: Validated ARD configuration.
        scale: Which subset of each modality's legal blocks to plan.  ``None``
            means every legal block; ``--smoke`` passes
            :data:`~ard.core.sampling.SMOKE_SCALE` through the same rule.

    Returns:
        The plan's :class:`~ard.core.types.AnchorSpec` objects, in plan order.

    Raises:
        OntologySchemaError: If the ontology is not a readable v4 document.
        SamplingError: If the ontology cannot produce the rule's coordinate set.
    """
    ontology: OntologyV4 = load_ontology_v4(config.ontology.path)
    gen_config = AnchorGenerationConfig(
        seed=config.generation.resolved_seed,
        concurrency=config.generation.concurrency,
    )
    return sample_anchors(ontology, gen_config, scale=scale)


@dataclass(frozen=True, slots=True)
class _CoverageInputs:
    """The acceptance phase's validated inputs, prepared before any side effect.

    The target set is parsed and checked *before* the output directory exists
    (§2.3 边界校验即防呆): a missing / malformed / dimension-mismatched target
    set is a config-side mistake, and refusing it up front costs nothing while
    discovering it after 1,826 paid generations costs the whole run.
    """

    embedding: CoverageEmbedding
    target_set: TargetSet


def _prepare_coverage(config: ARDConfig) -> _CoverageInputs | None:
    """Validate the acceptance phase's inputs, or return ``None`` for no metric readout.

    Returns:
        The resolved embedder and target set when a metric readout is
        configured; ``None`` when the phase is disabled or no target set is
        provided (the structure-only path, announced in a WARNING).

    Raises:
        ConfigError: If a target set is configured but the embedder is unusable
            (also checked at config load — this is the run-side boundary).
        CoverageWiringError: If the target-set file cannot be read or violates
            its declared count / dimension / epsilon.
    """
    coverage = config.coverage
    if not coverage.enabled:
        logger.info("Acceptance phase disabled: coverage.enabled = false.")
        return None
    target_set_path = coverage.target_set_path
    if target_set_path is None:
        logger.warning(
            "Acceptance metric readout not measured: coverage.target_set_path is unset. "
            "Only the structure readout (plan counts vs the construction rule) is "
            "published under results/. Set coverage.target_set_path to measure q95."
        )
        return None
    embedding = coverage.resolved_embedding()
    if embedding is None:
        # ``resolved_embedding`` returns None only for "disabled" or "no target
        # set", and both are handled above; reaching this means the two checks
        # disagree, which must not silently skip the metric readout (§3.2).
        raise ConfigError(
            "coverage.resolved_embedding() returned None although the acceptance phase is "
            f"enabled and coverage.target_set_path is set ({target_set_path!r})"
        )
    target_set = load_target_set(target_set_path, expected_dimension=embedding.dimension)
    logger.info(
        "Acceptance phase: target set %s (%d entries) will be measured after generation.",
        target_set.path,
        len(target_set.texts),
    )
    return _CoverageInputs(embedding=embedding, target_set=target_set)


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
    """One WARNING naming the planned coordinates the bank does not hold (F1).

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
    coverage_inputs: _CoverageInputs | None,
    plan: list[AnchorSpec],
    records: JsonObjectList,
    output_path: Path,
) -> dict[str, Any] | None:
    """Write ``results/coverage.json`` + ``coverage.md`` and return the manifest pointer.

    Runs after generation and before the manifest is written.  The structure
    readout is published first and always (zero model calls, so the artifact
    exists even if the embedding step below fails); the metric readout is
    produced only when :func:`_prepare_coverage` resolved an embedding endpoint
    and a target set — otherwise the report carries an explicit WARNING instead
    of a silently missing number (§3.2 透明退化).

    Returns:
        The manifest's ``acceptance`` section, or ``None`` when the phase is
        disabled (nothing was written).
    """
    if not config.coverage.enabled:
        return None

    output_dir = output_path.parent
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    structure = acceptance.structure_readout([spec.anchor_meta for spec in plan])
    warnings: list[str] = []
    # The structure readout describes the *plan*; it must not claim the artifact
    # is within the rule while the bank is short a planned coordinate (F1).  A
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

    if coverage_inputs is None:
        report = acceptance.AcceptanceReport(
            structure=structure,
            metrics=None,
            warnings=[
                *warnings,
                "metric readout not measured: coverage.target_set_path is unset, so only "
                "the structure readout was produced",
            ],
        )
        coverage_json = _write_coverage_report(report, results_dir)
        logger.warning(
            "Acceptance readout written to %s: structure only, no metric readout "
            "(coverage.target_set_path is unset).",
            coverage_json,
        )
        return {
            "coverage_json": "results/coverage.json",
            "coverage_md": "results/coverage.md",
            "metric_readout": False,
            "q95": None,
        }

    # Publish the structure readout before the embedding call: an embedding
    # failure must abort the run loudly (the client's own exception type), but it
    # must not be able to erase the zero-cost artifact that already describes
    # what the plan covers.
    _write_coverage_report(
        acceptance.AcceptanceReport(structure=structure, metrics=None, warnings=warnings),
        results_dir,
    )
    metrics = build_metric_readout(
        embedding=coverage_inputs.embedding,
        records=records,
        anchors_source=str(output_path),
        target_set=coverage_inputs.target_set,
    )
    if not metrics.noise.available:
        warnings.append(f"noise band unavailable: {metrics.noise.reason}")
    coverage_json = _write_coverage_report(
        acceptance.AcceptanceReport(structure=structure, metrics=metrics, warnings=warnings),
        results_dir,
    )
    logger.info(
        "Acceptance readout written to %s: q95 = %.6f over |T| = %d.",
        coverage_json,
        metrics.quantiles.q95,
        metrics.space.n_target,
    )
    return {
        "coverage_json": "results/coverage.json",
        "coverage_md": "results/coverage.md",
        "metric_readout": True,
        "q95": metrics.quantiles.q95,
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
            A required ``visual_domain`` whose directory is missing or holds no
            supported image is refused — naming the missing domains, their
            expected paths and the affected anchor count — unless
            ``[images] skip_missing_images`` is true (then those anchors are
            skipped, WARNING-logged and declared in ``manifest.json``).
            Whether the selected picture is transcoded to JPEG on the way into
            ``<output_dir>/images`` is ``[images] convert`` (§10.1) — not an
            argument of this function.
        generate_specs: Optional override for the plan builder, defaulting to
            :func:`sample_specs`.  Tests inject a small deterministic plan here
            so the resume arithmetic can be exercised without materialising the
            1,826-coordinate plan or loading the ontology.  It is not used to
            implement ``--smoke``: a smoke run goes through the real builder with
            the real scale knob.
        smoke: Run-boundary flag (``--smoke``, constitution §10.1).  The
            **same** construction rule is materialised at a reduced scale
            (:data:`~ard.core.sampling.SMOKE_SCALE`, 8 of 1,826 anchors) so a
            clone can see an artifact quickly.  A smoke run is tagged in three
            places: the run directory gets ``_smoke``, the log carries a
            WARNING, and ``manifest.json`` declares ``smoke: true`` with its
            planned count against the full one.  ``False`` (the default) changes
            nothing about a full run.

    Returns:
        Path to the output directory.

    Raises:
        ConfigError: If a required LLM endpoint field (``api_base`` /
            ``model_name``) is unset, if the acceptance phase is configured
            without a usable embedder, if an image-modality anchor's
            ``visual_domain`` has no image under ``--image-dir`` while
            ``[images] skip_missing_images`` is false, or if the bank already
            holds records and this run's plan identity differs from the plan
            identity recorded in the previous run's ``manifest.json`` (a
            different plan would mix two datasets in one run directory).  All are
            checked before the output directory is created, so a refused run
            leaves no side effect behind (§2.3).
        CoverageWiringError: If ``coverage.target_set_path`` names a file that
            is missing, unreadable, unparsable, empty, or whose declared count
            / dimension contradicts the file or the config — also before any
            side effect.
    """
    if smoke:
        logger.warning(
            "SMOKE RUN (--smoke): a reduced plan will be generated; the artifact is "
            "deliberately incomplete and must not be delivered as a dataset."
        )

    if generate_specs is not None:
        spec_sampler: SpecSampler = generate_specs
    elif smoke:
        # The very same builder, one named scale knob — never a second rule.
        spec_sampler = partial(sample_specs, scale=SMOKE_SCALE)
    else:
        spec_sampler = sample_specs
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

    # ── Acceptance inputs (also before the output directory exists) ────────
    # A target set that is missing, malformed or dimension-mismatched is a
    # config-side mistake: refusing it here costs nothing, while finding it after
    # the run paid for every anchor costs the run (§2.3).
    coverage_inputs = _prepare_coverage(config)

    # ── Plan (v4 construction rule) — built before any side effect ──────────
    # v4 only: the parsing layer refuses a schema it cannot read, so a v3 file
    # (or a damaged one) stops the run here instead of silently sampling 0
    # anchors further down (§2.3 边界校验即防呆).  ``sample_specs`` loads it.
    #
    # The plan **is** the target: ``len(plan)`` is the rule-derived count, and
    # the checkpoint/resume path asks for the plan entries whose stable id is not
    # in the bank yet.  Resume is append-only by construction — the generator
    # writes only ids absent from the bank — and it is *identity*-based so a
    # middle coordinate abandoned by an earlier run is requested again instead of
    # being shadowed by a later, already-written one (see the ``pending_specs``
    # comment below).
    #
    # It is built *before* the output directory exists so the image-addressing
    # check below runs on it and can still refuse before any side effect: a run
    # that cannot put a matching image under an image-modality anchor must fail
    # without leaving an empty output directory (or a config snapshot) behind.
    # The consumers below (image allocation, the generator) read the same
    # ``AnchorGenerationConfig`` the plan builder builds; this one is for the
    # generator's concurrency and the image-allocation rng.
    gen_config = AnchorGenerationConfig(
        seed=config.generation.resolved_seed,
        concurrency=config.generation.concurrency,
    )
    rng = random.Random(gen_config.seed)
    plan = spec_sampler(config)
    target_count = len(plan)
    plan_id = PlanIdentity.of(plan)
    logger.info(
        "Plan identity: %s (algorithm=%s, version=%d, coordinates=%d)",
        plan_id.digest,
        plan_id.algorithm,
        plan_id.version,
        plan_id.plan_size,
    )
    if smoke:
        logger.warning(
            "SMOKE RUN: plan reduced to %d of %d anchors (%d text blocks + %d "
            "image blocks) by the standard construction rule at smoke scale. "
            "results/coverage.json will report within_rule=false against the full "
            "plan — expected for a smoke artifact. Do not deliver it; run without "
            "--smoke for the full dataset.",
            target_count,
            EXPECTED_TOTAL,
            SMOKE_TEXT_BLOCKS,
            SMOKE_IMAGE_BLOCKS,
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

    # ── Resume identity check (§2.3, still before any side effect) ──────────
    # The plan's identity — not ``[generation] seed``, which is a process-level
    # config value that every invocation redraws and rewrites — is what decides
    # whether appending to this bank stays within one plan.  The run directory
    # records it in one of two places: the finished run's ``manifest.json``, or —
    # when the previous run was interrupted before the manifest was written —
    # the intermediate progress record.  Both are read here, and a mismatch is
    # refused before the config snapshot is rewritten, so nothing on disk is
    # touched.
    recorded_plan_identity = _recorded_plan_identity(output_dir / "manifest.json")
    _refuse_plan_change_on_resume(
        output_dir,
        plan_id,
        {spec.id for spec in plan},
        existing_records,
        recorded_plan_identity or _recorded_progress_identity(output_dir),
    )

    # ── Image addressing boundary check (§2.3, before the output dir) ──────
    # Every image-modality anchor resolves its picture under
    # ``<image_dir>/<visual_domain>/``.  A required domain with no usable image
    # is refused here — naming the missing domains, their expected paths and how
    # many anchors they affect — unless the user explicitly opted into skipping
    # those anchors ([images] skip_missing_images = true, §3.3 预授权退路).
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
    # full plan (F1: a green readout decoupled from the library).  Identity makes
    # every missing coordinate pending again, in plan order.
    pending_specs = [spec for spec in plan if spec.id not in existing_ids]
    # ``[images] convert`` decides both the accepted input set and the bytes
    # that land in <output_dir>/images, so it is read here from the config
    # rather than from a CLI flag (§10.1: one source of truth per parameter).
    image_extensions = CONVERTABLE_EXTENSIONS if config.images.convert else SUPPORTED_EXTENSIONS
    image_resolution: DomainImageResolution | None = None
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
        image_resolution = resolve_domain_images(
            image_root,
            pending_specs,
            seed=gen_config.seed,
            extensions=image_extensions,
        )
        if image_resolution.missing and not config.images.skip_missing_images:
            raise ConfigError(
                _missing_image_message(
                    image_root,
                    image_resolution,
                    len(pending_specs),
                    image_extensions,
                )
            )
        skipped_domains = image_resolution.missing
        if skipped_domains:
            available = set(image_resolution.selected)
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
            expected = domain_directory(image_dir, domain)
            for anchor_id in skipped_domains[domain]:
                logger.warning(
                    "Skipping anchor %s: visual_domain %r has no usable image under %s "
                    "([images] skip_missing_images = true).",
                    anchor_id,
                    domain,
                    expected,
                )

    if not specs:
        if remaining > 0:
            logger.warning(
                "Every remaining anchor (%d) was skipped because its visual_domain "
                "has no image; nothing to generate.",
                remaining,
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
        if smoke:
            _declare_smoke(manifest, plan_size=target_count, output_dir=output_dir)
        if image_dir is not None:
            _declare_images(
                manifest,
                image_dir=image_dir,
                config=config,
                resolution=image_resolution,
                skipped=skipped_domains,
            )
        # No generation happens on this path, so there are no run counters to
        # report.  A manifest that already names this plan is therefore left
        # byte-for-byte as it is rather than overwritten with a clean-looking
        # one (§3.2 — a missing field must not read as "healthy", and a finished
        # run must not be restated as unfinished).  The intermediate progress
        # record is dropped either way: the manifest is now the current word.
        acceptance_pointer = _run_acceptance(
            config, coverage_inputs, effective_plan, all_records, output_path
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
            # Since B3 the key is **always** emitted, so this line is the only
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
    # Step 2: place the selected image of every required visual_domain into the
    # output tree and assign it to the anchors that carry that coordinate.  The
    # files were already validated (existence, missing-domain refusal) before
    # the output directory existed; this step only copies bytes.
    if image_dir is not None and image_resolution is not None and specs:
        rel_by_domain: dict[str, str] = {}
        for domain, source in image_resolution.selected.items():
            if config.images.convert:
                placed = convert_and_copy_images([source], output_dir, subdir=domain)
            else:
                placed = copy_images_to_output([source], output_dir, subdir=domain)
            if not placed:
                raise ConfigError(
                    f"image {source} for visual_domain {domain!r} could not be "
                    f"converted/copied into {output_dir / 'images' / domain}. "
                    f"Replace it with a readable image, or remove the domain's "
                    f"directory to be told it is missing."
                )
            rel_by_domain[domain] = placed[0]
        _assign_images_by_domain(specs, rel_by_domain, output_dir, rng)

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
    # Reasoning observability (WP-F3): snapshot the API client's reasoning
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
    # Publish the run's failures next to the anchors that survived, so a short
    # bank can never be mistaken for a healthy one (§3.2).
    with_generation_report(
        manifest,
        stats=generation_stats.to_manifest_dict(),
        failures=reasoning_delta,
    )
    acceptance_pointer = _run_acceptance(
        config, coverage_inputs, effective_plan, all_records, output_path
    )
    if acceptance_pointer is not None:
        manifest["acceptance"] = acceptance_pointer
    if smoke:
        _declare_smoke(manifest, plan_size=target_count, output_dir=output_dir)
    if image_dir is not None:
        _declare_images(
            manifest,
            image_dir=image_dir,
            config=config,
            resolution=image_resolution,
            skipped=skipped_domains,
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
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        pass
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
