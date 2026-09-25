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
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from ard.backends.api_client import (
    ChatAPIClient,
    ChatAPIConfig,
)
from ard.backends.api_client import (
    reasoning_stats as api_client_reasoning_stats,
)
from ard.backends.coverage_wiring import TargetSet, build_metric_readout, load_target_set
from ard.config import ARDConfig, ConfigError, CoverageEmbedding
from ard.core import acceptance
from ard.core.ontology import OntologyV4, load_ontology_v4
from ard.core.quota import allocate_images
from ard.core.sampling import (
    EXPECTED_TOTAL,
    SMOKE_IMAGE_BLOCKS,
    SMOKE_SCALE,
    SMOKE_TEXT_BLOCKS,
    PlanScale,
    sample_anchors,
)
from ard.core.types import AnchorGenerationConfig, AnchorSpec
from ard.domain.bank import (
    build_manifest_from_records,
    count_existing_anchors,
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
# dict that feeds **both** ``config.json`` and the manifest's ``config``
# section — one definition, both sinks (§1.4).

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
    skipped: dict[str, list[str]],
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
    groups: dict[str, list[AnchorSpec]] = {}
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


def _run_acceptance(
    config: ARDConfig,
    coverage_inputs: _CoverageInputs | None,
    plan: list[AnchorSpec],
    records: list[dict[str, Any]],
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
    no_convert: bool = False,
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
        no_convert: If ``True``, skip image format conversion — only
            :data:`ard.domain.image_store.SUPPORTED_EXTENSIONS` are
            accepted and images are copied as-is.  The default
            (``False``) enables automatic conversion of RAW / BMP /
            TIFF / GIF / WebP images to JPEG.
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
            without a usable embedder, or if an image-modality anchor's
            ``visual_domain`` has no image under ``--image-dir`` while
            ``[images] skip_missing_images`` is false.  All are checked before
            the output directory is created, so a refused run leaves no side
            effect behind (§2.3).
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
    # die later (the old order left an empty ``out/config.json`` and ``logs/``
    # behind).  If "re-run a finished bank without credentials" is ever needed,
    # restructure the side-effect order — do not move this guard down.
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
    # the checkpoint/resume path asks for the plan entries not on disk yet.
    # Slicing the *tail* keeps resume append-only — plan order is deterministic,
    # so the first N entries are exactly the N anchors a previous run wrote
    # first.
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
    existing_count = 0 if overwrite_existing else count_existing_anchors(output_path)
    remaining = target_count - existing_count

    # ── Image addressing boundary check (§2.3, before the output dir) ──────
    # Every image-modality anchor resolves its picture under
    # ``<image_dir>/<visual_domain>/``.  A required domain with no usable image
    # is refused here — naming the missing domains, their expected paths and how
    # many anchors they affect — unless the user explicitly opted into skipping
    # those anchors ([images] skip_missing_images = true, §3.3 预授权退路).
    # Only the anchors still to be generated are checked: a completed bank needs
    # no image lookup at all.
    pending_specs = plan[existing_count:]
    image_extensions = SUPPORTED_EXTENSIONS if no_convert else CONVERTABLE_EXTENSIONS
    image_resolution: DomainImageResolution | None = None
    skipped_domains: dict[str, list[str]] = {}
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

    # The plan this run actually owns: skipped anchors are removed, so both the
    # resume arithmetic and the acceptance structure readout stay consistent
    # with what is on disk instead of claiming a rule-conformant plan.
    specs = pending_specs
    effective_plan = plan[:existing_count] + specs

    # ── Output directory (first side effect) ───────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── File logging (must happen after output_dir exists) ──────────────
    configure_file_logging(output_dir)

    # ── Overwrite / checkpoint-resume ──────────────────────────────────────
    if overwrite_existing:
        output_path.unlink()
        logger.info("Overwrite mode: cleared existing anchor bank at %s", output_path)

    # Backup merged config to output directory for reproducibility.
    # Credentials are masked first (§2.3 — the output directory is a boundary
    # users share); the same redacted dict feeds the manifest's `config`
    # section below, so both sinks are covered by this single definition.
    config_info = _redact_secrets(config.model_dump(mode="json"))
    config_json_path = output_dir / "config.json"
    config_json_path.write_text(
        json.dumps(config_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Merged config snapshot written to %s", config_json_path)

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
        # No generation happened, so there are no run counters to report: the
        # previous run's manifest is left as it is rather than overwritten with
        # a clean-looking one (§3.2 — a missing field must not read as "healthy").
        acceptance_pointer = _run_acceptance(
            config, coverage_inputs, effective_plan, all_records, output_path
        )
        if acceptance_pointer is not None:
            manifest["acceptance"] = acceptance_pointer
        write_manifest(manifest, output_dir / "manifest.json")
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
            if no_convert:
                placed = copy_images_to_output([source], output_dir, subdir=domain)
            else:
                placed = convert_and_copy_images([source], output_dir, subdir=domain)
            if not placed:
                raise ConfigError(
                    f"image {source} for visual_domain {domain!r} could not be "
                    f"converted/copied into {output_dir / 'images' / domain}. "
                    f"Replace it with a readable image, or remove the domain's "
                    f"directory to be told it is missing."
                )
            rel_by_domain[domain] = placed[0]
        _assign_images_by_domain(specs, rel_by_domain, output_dir, rng)

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
    manifest = build_manifest_from_records(all_records, output_dir, config_info)
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
    write_manifest(manifest, output_dir / "manifest.json")

    total = len(all_records)
    logger.info("Done! Output: %s", output_dir)
    logger.info(
        "  Total anchors: %d (%d existing + %d new)",
        total,
        existing_count,
        total - existing_count,
    )
    return output_dir
