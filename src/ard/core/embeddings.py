"""Embedding utilities for ARD anchor ontology.

Pure computation layer: loading pre-computed embeddings from JSON
and farthest-point sampling for anchor selection.  No network, no GPU
management, no file-system side-effects beyond the JSON load path.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np

_np = None


def _get_np():
    global _np
    if _np is None:
        import numpy as _np_module
        _np = _np_module
    return _np


# ---------------------------------------------------------------------------
# 1.  Loading pre-computed embeddings
# ---------------------------------------------------------------------------


def load_embeddings(path: str) -> dict[str, Any]:
    """Load pre-computed anchor-ontology embeddings from a JSON file.

    The file is expected to follow the ``anchor_ontology_embeddings.json``
    schema:

    * ``ontology_sha256`` — hex digest of the source ontology
    * ``embedding_model`` — HuggingFace model id
    * ``embedding_dimension`` — int, e.g. 1024
    * ``distance`` — metric name, always ``"cosine"``
    * ``items`` — list of dicts, each with ``section``, ``path``,
      ``leaf``, ``text``, and ``embedding`` (list of floats)

    Returns the full parsed dict.  No validation beyond JSON parsing is
    performed — callers are expected to validate the structure if needed.
    """
    raw = pathlib.Path(path).read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(raw)
    return data


# ---------------------------------------------------------------------------
# 2.  Farthest Point Sampling (FPS)
# ---------------------------------------------------------------------------


def farthest_point_sampling(
    embeddings: np.ndarray,
    n: int,
    *,
    seed: int | None = None,
) -> list[int]:
    """Select *n* diverse points via Farthest Point Sampling (FPS).

    Starting from a random initial point, the algorithm iteratively picks
    the point whose **minimum cosine distance** to the already-selected
    set is the largest.  This yields a subset that is well-spread in the
    embedding space.

    Args:
        embeddings: Array of shape ``(N, D)`` — *N* points, each a
            *D*-dimensional embedding vector.
        n: Number of points to select.  Must satisfy ``1 <= n <= N``.
        seed: Optional random seed for the initial point selection.
            When ``None`` (default) the first point is chosen via a
            fixed-seed RNG for reproducibility.

    Returns:
        List of *n* indices into ``embeddings``.

    Raises:
        ValueError: If ``n`` is out of range or ``embeddings`` is empty.
    """
    np = _get_np()
    N = embeddings.shape[0]
    if N == 0:
        raise ValueError("embeddings array is empty")
    if n < 1 or n > N:
        raise ValueError(
            f"n must be in [1, {N}], got {n}"
        )

    if n == 1:
        rng = np.random.default_rng(seed if seed is not None else 42)
        return [int(rng.integers(0, N))]

    if n == N:
        return list(range(N))

    # -- normalize for cosine-distance computation --------------------------
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Guard against zero vectors — replace with tiny value so division
    # succeeds and the zero vector ends up at the origin.
    norms = np.where(norms == 0.0, 1e-12, norms)
    unit = embeddings / norms

    # -- initialise ---------------------------------------------------------
    rng = np.random.default_rng(seed if seed is not None else 42)
    selected: list[int] = [int(rng.integers(0, N))]

    # min_dist[i] = min cosine distance from point i to any selected point
    # Cosine distance = 1 - cosine_similarity
    first_vec = unit[selected[0]]  # (D,)
    sim = unit @ first_vec          # (N,)  — dot products with first point
    min_dist = 1.0 - sim            # (N,)  — cosine distances

    # -- iterate ------------------------------------------------------------
    for _ in range(1, n):
        # Pick the point with the largest minimum distance.
        mask = np.ones(N, dtype=bool)
        mask[selected] = False
        masked_dist = np.where(mask, min_dist, -np.inf)
        best = int(np.argmax(masked_dist))
        selected.append(best)

        # Update min_dist: for every point, the distance to the new point
        # might be smaller than the current min_dist.
        new_sim = unit @ unit[best]          # (N,)
        new_dist = 1.0 - new_sim             # (N,)
        min_dist = np.minimum(min_dist, new_dist)

    return selected
