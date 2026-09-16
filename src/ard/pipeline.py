"""Main pipeline orchestrator for ARD anchor generation.

Phase 3 unified flow: all anchors (text + multimodal) are generated from
:class:`AnchorSpec` objects via a single code path.
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from ard.backends.api_client import (
    ChatAPIClient,
    ChatAPIConfig,
    reasoning_stats as api_client_reasoning_stats,
)
from ard.config import ARDConfig
from ard.core.ontology import load_ontology
from ard.core.quota import allocate_images
from ard.core.sampler import sample_anchors
from ard.core.types import AnchorGenerationConfig
from ard.logging import configure_file_logging
from ard.domain.bank import (
    build_manifest_from_records,
    count_existing_anchors,
    read_anchor_bank,
    with_generation_report,
    write_manifest,
)
from ard.domain.image_store import (
    CONVERTABLE_EXTENSIONS,
    convert_and_copy_images,
    copy_images_to_output,
    sample_images,
    scan_images,
)
from ard.domain.text_anchor import AnchorGenerationStats, generate_text_anchors

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

REDACTED_KEY_FLAT_SET: frozenset[str] = frozenset(
    _flatten_key(word) for word in REDACTED_KEY_WORDS
)
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


def _redact_value(value: object, found: set[str]) -> object:
    """Recursively copy *value*, masking credential values, collecting keys hit.

    Containers are rebuilt (never mutated in place) so the caller's config
    dict is left untouched.
    """
    if isinstance(value, dict):
        redacted: dict[object, object] = {}
        for key, item in value.items():
            if _is_secret_key(key) and item is not None and item != "":
                redacted[key] = REDACTED_PLACEHOLDER
                found.add(str(key))
            else:
                redacted[key] = _redact_value(item, found)
        return redacted
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
    redacted_info = _redact_value(config_info, found)
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


def run(
    config: ARDConfig,
    *,
    image_dir: str | None = None,
    no_convert: bool = False,
) -> Path:
    """Run the ARD anchor generation pipeline.

    Args:
        config: Validated ARD configuration.
        image_dir: Optional path to image directory. If provided, multimodal
            anchors are generated alongside text anchors.
        no_convert: If ``True``, skip image format conversion — only
            :data:`ard.domain.image_store.SUPPORTED_EXTENSIONS` are
            accepted and images are copied as-is.  The default
            (``False``) enables automatic conversion of RAW / BMP /
            TIFF / GIF / WebP images to JPEG.

    Returns:
        Path to the output directory.

    Raises:
        RuntimeError: If multimodal mode is requested but LLM API
            configuration is incomplete.
    """
    # ── Output directory ──────────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_name = f"ard_dataset_{timestamp}"
    output_dir = (
        Path(config.output.directory)
        if config.output.directory
        else Path("outputs") / dataset_name
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── File logging (must happen after output_dir exists) ──────────────
    configure_file_logging(output_dir)

    output_path = output_dir / "anchor_bank.jsonl"

    # ── Overwrite / checkpoint-resume ──────────────────────────────────────
    if output_path.exists() and config.output.overwrite:
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

    # ── Checkpoint / resume ───────────────────────────────────────────────
    existing_count = count_existing_anchors(output_path)
    target_count = config.generation.target_count
    remaining = target_count - existing_count

    if remaining <= 0:
        logger.info("Already have %d anchors, skipping generation.", existing_count)
        all_records = read_anchor_bank(output_path)
        manifest = build_manifest_from_records(all_records, output_dir, config_info)
        # No generation happened, so there are no run counters to report: the
        # previous run's manifest is left as it is rather than overwritten with
        # a clean-looking one (§3.2 — a missing field must not read as "healthy").
        write_manifest(manifest, output_dir / "manifest.json")
        logger.info("Done! Output: %s", output_dir)
        logger.info("  Total anchors: %d", len(all_records))
        return output_dir

    logger.info("Found %d existing anchors, generating %d more...", existing_count, remaining)

    # ── Ontology ──────────────────────────────────────────────────────────
    ontology = load_ontology(config.ontology.path)

    # ── Generation config (adjusted for remaining) ────────────────────────
    gen_config = AnchorGenerationConfig(
        target_count=remaining,
        seed=config.generation.resolved_seed,
        concurrency=config.generation.concurrency,
        languages=config.generation.languages,
        task_types=config.generation.task_types,
        max_turns=config.generation.max_turns,
        max_turns_with_image=config.generation.max_turns_with_image,
        embeddings_path=config.generation.embeddings_path,
    )

    # ── API clients ───────────────────────────────────────────────────────
    input_client = ChatAPIClient(
        ChatAPIConfig(
            api_base=config.input_generator.api_base,
            model_name=config.input_generator.model_name,
            api_key=config.input_generator.api_key,
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
            api_base=config.target_model.api_base,
            model_name=config.target_model.model_name,
            api_key=config.target_model.api_key,
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

    # ── Multimodal validation ───────────────────────────────────────────
    if image_dir:
        if config.input_generator.api_base is None:
            raise RuntimeError(
                "Multimodal mode (--image-dir) requires input_generator.api_base "
                "to be set in config."
            )
        if config.target_model.api_base is None:
            raise RuntimeError(
                "Multimodal mode (--image-dir) requires target_model.api_base "
                "to be set in config."
            )
        if config.input_generator.model_name is None:
            raise RuntimeError(
                "Multimodal mode (--image-dir) requires input_generator.model_name "
                "to be set in config."
            )
        if config.target_model.model_name is None:
            raise RuntimeError(
                "Multimodal mode (--image-dir) requires target_model.model_name "
                "to be set in config."
            )

    # ── Generate anchors (unified flow) ───────────────────────────────────
    rng = random.Random(gen_config.seed)

    # Step 1: Sample AnchorSpec objects from the ontology
    specs = sample_anchors(ontology, gen_config, rng)

    # Step 2: If image_dir is provided, scan, sample, convert/copy, and allocate images
    if image_dir:
        if no_convert:
            images = scan_images(image_dir, recursive=True)
            if images:
                sampled = sample_images(images, 100, seed=gen_config.seed)
                rel_paths = copy_images_to_output(sampled, output_dir)
                specs = allocate_images(
                    specs, rel_paths, config.generation.max_turns_with_image, rng
                )
                # Resolve image paths relative to output_dir for base64 encoding
                for spec in specs:
                    for turn in spec.turns:
                        if turn.image_path:
                            turn.image_path = str(output_dir / turn.image_path)
            else:
                logger.warning(
                    "No images found in %s. All anchors will be pure text.",
                    image_dir,
                )
        else:
            images = scan_images(image_dir, recursive=True, extensions=CONVERTABLE_EXTENSIONS)
            if images:
                sampled = sample_images(images, 100, seed=gen_config.seed)
                rel_paths = convert_and_copy_images(sampled, output_dir)
                specs = allocate_images(
                    specs, rel_paths, config.generation.max_turns_with_image, rng
                )
                # Resolve image paths relative to output_dir for base64 encoding
                for spec in specs:
                    for turn in spec.turns:
                        if turn.image_path:
                            turn.image_path = str(output_dir / turn.image_path)
            else:
                logger.warning(
                    "No images found in %s. All anchors will be pure text.",
                    image_dir,
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
        input_model_name=config.input_generator.model_name,
        target_model_name=config.target_model.model_name,
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
    write_manifest(manifest, output_dir / "manifest.json")

    total = len(all_records)
    logger.info("Done! Output: %s", output_dir)
    logger.info(
        "  Total anchors: %d (%d existing + %d new)",
        total, existing_count, total - existing_count,
    )
    return output_dir
