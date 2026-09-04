"""Main pipeline orchestrator for ARD anchor generation.

Phase 3 unified flow: all anchors (text + multimodal) are generated from
:class:`AnchorSpec` objects via a single code path.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

from ard.backends.api_client import ChatAPIClient, ChatAPIConfig
from ard.config import ARDConfig
from ard.core.ontology import load_ontology
from ard.core.quota import allocate_images
from ard.core.sampler import sample_anchor_specs
from ard.core.types import AnchorGenerationConfig
from ard.domain.bank import (
    build_manifest_from_records,
    count_existing_anchors,
    read_anchor_bank,
    write_manifest,
)
from ard.domain.image_store import copy_images_to_output, sample_images, scan_images
from ard.domain.text_anchor import generate_text_anchors


def run(
    config: ARDConfig,
    *,
    image_dir: str | None = None,
) -> Path:
    """Run the ARD anchor generation pipeline.

    Args:
        config: Validated ARD configuration.
        image_dir: Optional path to image directory. If provided, multimodal
            anchors are generated alongside text anchors.

    Returns:
        Path to the output directory.

    Raises:
        FileExistsError: If the output directory exists and ``overwrite`` is
            ``False`` in the config.
    """
    # ── Output directory ──────────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_name = f"ard_dataset_{timestamp}"
    output_dir = (
        Path(config.output.directory)
        if config.output.directory
        else Path("outputs") / dataset_name
    )

    if output_dir.exists():
        if not config.output.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                f"Set output.overwrite=true in config to overwrite."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "anchor_bank.jsonl"

    # ── Checkpoint / resume ───────────────────────────────────────────────
    existing_count = count_existing_anchors(output_path)
    target_count = config.generation.target_count
    remaining = target_count - existing_count

    if remaining <= 0:
        print(f"Already have {existing_count} anchors, skipping generation.")
        all_records = read_anchor_bank(output_path)
        manifest = build_manifest_from_records(all_records, output_dir)
        write_manifest(manifest, output_dir / "manifest.json")
        print(f"\nDone! Output: {output_dir}")
        print(f"  Total anchors: {len(all_records)}")
        return output_dir

    print(f"Found {existing_count} existing anchors, generating {remaining} more...")

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
            max_retries=config.input_generator.max_retries,
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
            max_retries=config.target_model.max_retries,
        )
    )

    # ── Generate anchors (unified flow) ───────────────────────────────────
    rng = random.Random(gen_config.seed)

    # Step 1: Sample AnchorSpec objects from the ontology
    specs = sample_anchor_specs(ontology, gen_config, rng)

    # Step 2: If image_dir is provided, scan, sample, copy, and allocate images
    if image_dir:
        images = scan_images(image_dir, recursive=True)
        if images:
            sampled = sample_images(images, 100, seed=gen_config.seed)
            rel_paths = copy_images_to_output(sampled, output_dir)
            specs = allocate_images(
                specs, rel_paths, config.generation.max_turns_with_image, rng
            )
        else:
            print(
                f"Warning: No images found in {image_dir}. "
                f"All anchors will be pure text.",
                file=sys.stderr,
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
    print(f"\nDone! Output: {output_dir}")
    print(f"  Total anchors: {total} ({existing_count} existing + {total - existing_count} new)")
    return output_dir