"""Main pipeline orchestrator for ARD anchor generation.

Phase 3 unified flow: all anchors (text + multimodal) are generated from
:class:`AnchorSpec` objects via a single code path.
"""

from __future__ import annotations

import logging
import random
import shutil
import sys
import time
from pathlib import Path

from ard.backends.api_client import ChatAPIClient, ChatAPIConfig
from ard.config import ARDConfig
from ard.core.ontology import load_ontology
from ard.core.quota import allocate_images
from ard.core.sampler import sample_anchors
from ard.core.types import AnchorGenerationConfig
from ard.domain.bank import (
    build_manifest_from_records,
    count_existing_anchors,
    read_anchor_bank,
    write_manifest,
)
from ard.domain.image_store import (
    CONVERTABLE_EXTENSIONS,
    convert_and_copy_images,
    copy_images_to_output,
    sample_images,
    scan_images,
)
from ard.domain.text_anchor import generate_text_anchors

logger = logging.getLogger(__name__)

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

    output_path = output_dir / "anchor_bank.jsonl"

    # ── Overwrite / checkpoint-resume ──────────────────────────────────────
    if output_path.exists() and config.output.overwrite:
        output_path.unlink()
        logger.info("Overwrite mode: cleared existing anchor bank at %s", output_path)

    # Backup config to output directory for reproducibility
    base_config = Path("configs/config.toml")
    if base_config.exists():
        shutil.copy2(base_config, output_dir / "config.toml")

    # ── Checkpoint / resume ───────────────────────────────────────────────
    existing_count = count_existing_anchors(output_path)
    target_count = config.generation.target_count
    remaining = target_count - existing_count

    if remaining <= 0:
        logger.info("Already have %d anchors, skipping generation.", existing_count)
        all_records = read_anchor_bank(output_path)
        manifest = build_manifest_from_records(all_records, output_dir)
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
        seed=config.generation.seed,
        concurrency=config.generation.concurrency,
        languages=config.generation.languages,
        task_types=config.generation.task_types,
        max_turns=config.generation.max_turns,
        max_turns_with_image=config.generation.max_turns_with_image,
        system_persona=config.generation.system_persona,
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
            timeout=config.input_generator.timeout,
            connect_timeout=config.input_generator.connect_timeout,
            first_token_timeout=config.input_generator.first_token_timeout,
            inter_token_timeout=config.input_generator.inter_token_timeout,
            max_retries=config.input_generator.max_retries,
            retry_on_timeout=config.input_generator.retry_on_timeout,
        )
    )
    target_client = ChatAPIClient(
        ChatAPIConfig(
            api_base=config.target_model.api_base,
            model_name=config.target_model.model_name,
            api_key=config.target_model.api_key,
            temperature=config.target_model.temperature,
            max_tokens=config.target_model.max_tokens,
            timeout=config.target_model.timeout,
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
    new_anchors = generate_text_anchors(
        specs=specs,
        input_client=input_client,
        target_client=target_client,
        input_model_name=config.input_generator.model_name,
        target_model_name=config.target_model.model_name,
        concurrency=gen_config.concurrency,
        output_path=output_path,
    )

    # ── Build manifest from ALL anchors (existing + new) ──────────────────
    all_records = read_anchor_bank(output_path)
    manifest = build_manifest_from_records(all_records, output_dir)
    write_manifest(manifest, output_dir / "manifest.json")

    total = len(all_records)
    logger.info("Done! Output: %s", output_dir)
    logger.info("  Total anchors: %d (%d existing + %d new)", total, existing_count, total - existing_count)
    return output_dir