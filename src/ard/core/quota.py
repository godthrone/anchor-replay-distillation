"""Quota allocation for multi-turn and multimodal anchor generation.

Core layer — pure computation, no network or API dependencies.
"""

from __future__ import annotations

import itertools
import logging
import random
from typing import Any

from ard.core.types import AnchorSpec

logger = logging.getLogger(__name__)


def compute_turn_distribution(
    target_count: int, max_turns: int, rng: random.Random
) -> list[int]:
    """Distribute *target_count* anchors evenly across 1..*max_turns* turns.

    Returns a list of length *max_turns* where ``result[i]`` is the number
    of anchors that should have ``i + 1`` turns.  The distribution is as
    even as possible; any remainder is spread randomly among the buckets.

    Example:
        >>> rng = random.Random(42)
        >>> compute_turn_distribution(100, 3, rng)
        [34, 33, 33]  # 34 anchors with 1 turn, 33 with 2, 33 with 3
    """
    base = target_count // max_turns
    remainder = target_count % max_turns
    result = [base] * max_turns
    for i in range(remainder):
        result[i] += 1
    rng.shuffle(result)
    return result


def allocate_images(
    anchor_specs: list[AnchorSpec],
    image_pool: list[str],
    max_turns_with_image: int,
    rng: random.Random,
) -> list[AnchorSpec]:
    """Allocate images to anchor specs.

    Strategy:

    1. Cycle through image pool (images can be reused)
    2. Prefer single-turn specs (simpler scenarios get images first)
    3. Each spec gets at most *max_turns_with_image* image turns
    4. Images go on the earliest user turns
    5. If images are fewer than specs, remainder get text-only
       (warning logged)

    Args:
        anchor_specs: Anchor specifications to assign images to.
            Modified in place — each applicable ``TurnSpec.image_path``
            is set to a pool entry.
        image_pool: List of image paths to cycle through.
        max_turns_with_image: Maximum number of user turns per anchor
            that receive an image.  Must be ≥ 0.
        rng: Seeded :class:`random.Random` instance (unused; kept for
            API consistency).

    Returns:
        The same *anchor_specs* list, modified in place. Each spec's
        ``anchor_meta`` is updated with ``"has_image"`` (bool) and
        ``"image_count"`` (int) fields.
    """
    if not image_pool:
        return anchor_specs

    # Single-turn specs first (better visual quality)
    single_turn = [s for s in anchor_specs if len(s.turns) <= 2]
    multi_turn = [s for s in anchor_specs if len(s.turns) > 2]

    image_iter = itertools.cycle(image_pool)
    specs_with_images = 0

    for spec in single_turn + multi_turn:
        images_this_spec = 0
        for turn in spec.turns:
            if turn.role == "user" and turn.image_path is None:
                if images_this_spec >= max_turns_with_image:
                    break
                turn.image_path = next(image_iter)
                images_this_spec += 1
        spec.anchor_meta["has_image"] = images_this_spec > 0
        spec.anchor_meta["image_count"] = images_this_spec
        if images_this_spec > 0:
            specs_with_images += 1

    if specs_with_images < len(anchor_specs):
        logger.warning(
            "only %d image-allocated specs for %d total specs (image pool: %d)",
            specs_with_images,
            len(anchor_specs),
            len(image_pool),
        )

    return anchor_specs