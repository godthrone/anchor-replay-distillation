"""Main pipeline orchestrator for ARD anchor generation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from ard.config import ARDConfig
from ard.core.ontology import load_ontology
from ard.core.types import AnchorGenerationConfig
from ard.backends.api_client import ChatAPIClient, ChatAPIConfig
from ard.domain.text_anchor import generate_text_anchors
from ard.domain.multimodal_anchor import generate_multimodal_anchors
from ard.domain.bank import write_anchor_bank, build_manifest, write_manifest


def run(
    config: ARDConfig,
    *,
    image_dir: Optional[str] = None,
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
    output_dir = Path(config.output.directory) if config.output.directory is not None else Path("outputs") / dataset_name

    if output_dir.exists():
        if not config.output.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                f"Set output.overwrite=true in config to overwrite."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Ontology ──────────────────────────────────────────────────────────
    ontology = load_ontology(config.ontology.path)

    # ── Generation config ─────────────────────────────────────────────────
    gen_config = AnchorGenerationConfig(
        target_count=config.generation.target_count,
        seed=config.generation.seed,
        languages=config.generation.languages,
        task_types=config.generation.task_types,
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

    # ── Generate anchors ──────────────────────────────────────────────────
    all_anchors = []

    # Text anchors (always generated)
    print(f"Generating {gen_config.target_count} text anchors...")
    text_anchors = generate_text_anchors(
        ontology=ontology,
        config=gen_config,
        input_client=input_client,
        target_client=target_client,
        input_model_name=config.input_generator.model_name,
        target_model_name=config.target_model.model_name,
    )
    all_anchors.extend(text_anchors)
    print(f"  Generated {len(text_anchors)} text anchors")

    # Multimodal anchors (only if image_dir is provided)
    if image_dir:
        print(f"Generating multimodal anchors from {image_dir}...")
        mm_anchors = generate_multimodal_anchors(
            image_dir=image_dir,
            ontology=ontology,
            config=gen_config,
            vlm_client=input_client,
            target_client=target_client,
            model_name=config.input_generator.model_name,
            image_count=100,
            questions_per_image=1,
            output_dir=output_dir,
        )
        all_anchors.extend(mm_anchors)
        print(f"  Generated {len(mm_anchors)} multimodal anchors")

    # ── Write output ──────────────────────────────────────────────────────
    write_anchor_bank(all_anchors, output_dir / "anchor_bank.jsonl")
    manifest = build_manifest(all_anchors, output_dir)
    write_manifest(manifest, output_dir / "manifest.json")

    print(f"\nDone! Output: {output_dir}")
    print(f"  Total anchors: {len(all_anchors)}")
    return output_dir