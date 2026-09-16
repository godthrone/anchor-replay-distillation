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

    1. Cycle through *image_pool* with :func:`itertools.cycle` (images are
       reused, so the pool is infinite in effect)
    2. Prefer single-turn specs (``len(turns) <= 2``); they are visited first
       and then the rest, so simpler scenarios get their images first
    3. Walk each spec's turns in order and fill its **earliest** ``user`` turns
       — a turn that already carries an ``image_path`` is skipped and costs no
       budget — up to *max_turns_with_image* assigned images per spec
    4. Stamp every visited spec's ``anchor_meta`` with ``has_image`` and
       ``image_count``

    Allocation is therefore all-or-nothing per run, never "the first specs get
    images and the remainder go text-only": the pool is cycled, so it cannot run
    out and the number of specs never competes with the number of images.  There
    are exactly two text-only outcomes, and both are global:

    * *image_pool* is empty → the function returns immediately and the specs are
      left completely untouched (no ``has_image`` / ``image_count`` stamp);
    * *max_turns_with_image* is 0 → every spec is visited and stamped
      ``has_image=False`` / ``image_count=0``.

    Under 3, a spec can also stay text-only on its own: if none of its ``user``
    turns is eligible (it has no ``user`` turn, or all of them already carry an
    ``image_path``), it receives nothing.  The WARNING is logged in exactly that
    situation and in the ``max_turns_with_image = 0`` case — that is, whenever at
    least one visited spec ended up with no image — so a run that produced
    text-only anchors cannot do so silently (§3.2 透明退路).

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
        ``"image_count"`` (int) fields — unless *image_pool* is empty, in which
        case the list is returned unmodified.
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