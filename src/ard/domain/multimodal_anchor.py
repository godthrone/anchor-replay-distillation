"""Multimodal anchor generation pipeline."""

import random
from pathlib import Path
from typing import Any

from ard.core.types import Anchor, AnchorGenerationConfig
from ard.core.sampler import generate_anchor_id
from ard.backends.api_client import ChatAPIClient, ChatRequest, encode_image_to_base64
from ard.domain.image_store import scan_images, sample_images, copy_images_to_output

QUESTION_TYPES = {
    "description": "Describe what you see in this image in detail.",
    "reasoning": "What can you infer from this image? Explain your reasoning.",
    "comparison": "Compare and contrast the different elements visible in this image.",
    "counting": "How many distinct objects can you identify in this image? List them.",
    "spatial_reasoning": "Describe the spatial relationships between objects in this image.",
}


def generate_multimodal_anchors(
    image_dir: str | Path,
    ontology: dict[str, Any],
    config: AnchorGenerationConfig,
    vlm_client: ChatAPIClient,
    target_client: ChatAPIClient,
    model_name: str,
    image_count: int = 100,
    questions_per_image: int = 1,
    output_dir: str | Path | None = None,
) -> list[Anchor]:
    images = scan_images(image_dir, recursive=True)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    sampled = sample_images(images, image_count, seed=config.seed)
    rng = random.Random(config.seed)
    if output_dir:
        relative_paths = copy_images_to_output(sampled, output_dir)
        image_map = dict(zip(sampled, relative_paths))
    else:
        image_map = {p: p for p in sampled}
    anchors: list[Anchor] = []
    q_names = list(QUESTION_TYPES.keys())
    for src_path in sampled:
        data_url = None
        try:
            data_url = encode_image_to_base64(src_path)
        except Exception:
            continue
        rel_path = image_map.get(src_path, src_path)
        rel_path_str = str(rel_path).replace("\\", "/")
        q_types = rng.sample(q_names, min(questions_per_image, len(q_names)))
        for q_type in q_types:
            q_prompt = QUESTION_TYPES.get(q_type, QUESTION_TYPES["description"])

            # Step 1: Generate question via VLM
            vlm_req = ChatRequest(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": q_prompt},
                        ],
                    }
                ],
                temperature=0.7,
            )
            vlm_results = vlm_client.chat_batch([vlm_req], concurrency=1)
            if not vlm_results[0].success:
                continue
            question = vlm_results[0].content.strip()

            # Step 2: Generate answer with logprobs via target
            try:
                result = target_client.chat_with_logprobs(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data_url}},
                                {"type": "text", "text": question},
                            ],
                        }
                    ],
                    temperature=0.0,
                )
            except Exception:
                continue

            answer = result["content"].strip()
            logprobs = result["logprobs"]

            if len(answer) < 8:
                continue

            meta = {
                "visual_domain": "general",
                "question_type": q_type,
                "image_source": src_path.name,
                "language": "English",
            }
            anchor_id = generate_anchor_id(meta)
            anchor = Anchor(
                id=anchor_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": rel_path_str},
                            {"type": "text", "text": question},
                        ],
                    }
                ],
                target_answer=answer,
                target_model=model_name,
                input_generator_model=model_name,
                anchor_meta=meta,
                logprobs=logprobs,
            )
            anchors.append(anchor)
            if len(anchors) >= config.target_count:
                break
        if len(anchors) >= config.target_count:
            break
    return anchors