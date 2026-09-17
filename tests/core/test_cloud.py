"""Tests for :mod:`ard.core.cloud` — space-identified clouds and space-safe FPS.

The negative cases are the point of this module: a selection produced on one
cloud must fail loudly when applied to another (the historical defect), instead
of quietly computing a number from unrelated rows.
"""

from __future__ import annotations

import numpy as np
import pytest

from ard.core.cloud import (
    CloudIndex,
    CloudMismatchError,
    CloudVectors,
    SpaceMismatchError,
    fps,
)

SPACE = "test-embedder:8"
OTHER_SPACE = "other-embedder:8"


def _matrix(n: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, (n, dim))


def _cloud(
    n: int = 10,
    dim: int = 8,
    *,
    space_id: str = SPACE,
    cloud_id: str = "cloud_a",
    seed: int = 0,
) -> CloudVectors:
    return CloudVectors(
        space_id=space_id,
        cloud_id=cloud_id,
        vectors=_matrix(n, dim, seed),
        item_ids=tuple(f"item-{i}" for i in range(n)),
    )


# ---------------------------------------------------------------------------
# Carrier invariants
# ---------------------------------------------------------------------------


def test_cloud_keeps_identity_and_values():
    vectors = _matrix(6)
    cloud = CloudVectors(space_id=SPACE, cloud_id="cloud_a", vectors=vectors)
    assert cloud.space_id == SPACE
    assert cloud.cloud_id == "cloud_a"
    assert cloud.n_items == 6
    assert cloud.dimension == 8
    # Values are carried through as float64 without reordering.
    assert np.array_equal(cloud.vectors, vectors)


def test_cloud_converts_float32_to_float64():
    vectors = _matrix(4).astype(np.float32)
    cloud = CloudVectors(space_id=SPACE, cloud_id="cloud_a", vectors=vectors)
    assert cloud.vectors.dtype == np.float64


@pytest.mark.parametrize("bad_vectors", [np.zeros(8), np.empty((0, 8)), np.empty((4, 0))])
def test_cloud_rejects_invalid_shapes(bad_vectors):
    with pytest.raises(ValueError):
        CloudVectors(space_id=SPACE, cloud_id="cloud_a", vectors=bad_vectors)


def test_cloud_rejects_non_finite_values():
    vectors = _matrix(4)
    vectors[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        CloudVectors(space_id=SPACE, cloud_id="cloud_a", vectors=vectors)


@pytest.mark.parametrize(
    ("space_id", "cloud_id"),
    [("", "cloud_a"), (SPACE, "")],
)
def test_cloud_rejects_empty_identity(space_id, cloud_id):
    with pytest.raises(ValueError, match="non-empty"):
        CloudVectors(space_id=space_id, cloud_id=cloud_id, vectors=_matrix(3))


def test_cloud_rejects_item_ids_length_mismatch():
    with pytest.raises(ValueError, match="item_ids"):
        CloudVectors(
            space_id=SPACE, cloud_id="cloud_a", vectors=_matrix(3), item_ids=("a", "b")
        )


def test_require_item_ids_raises_when_absent():
    cloud = CloudVectors(space_id=SPACE, cloud_id="cloud_a", vectors=_matrix(3))
    with pytest.raises(ValueError, match="item_ids"):
        cloud.require_item_ids()


# ---------------------------------------------------------------------------
# CloudIndex invariants
# ---------------------------------------------------------------------------


def test_cloud_index_rejects_empty_positions():
    with pytest.raises(ValueError, match="must not be empty"):
        CloudIndex(space_id=SPACE, cloud_id="cloud_a", positions=())


def test_cloud_index_rejects_duplicates():
    with pytest.raises(ValueError, match="repeat"):
        CloudIndex(space_id=SPACE, cloud_id="cloud_a", positions=(0, 1, 0))


def test_cloud_index_rejects_negative_positions():
    with pytest.raises(ValueError, match="non-negative"):
        CloudIndex(space_id=SPACE, cloud_id="cloud_a", positions=(0, -1))


def test_index_of_rejects_out_of_range_positions():
    cloud = _cloud(n=5)
    with pytest.raises(ValueError, match="out of range"):
        cloud.index_of([0, 5])


# ---------------------------------------------------------------------------
# select(): the cross-cloud guard
# ---------------------------------------------------------------------------


def test_select_returns_same_identity_and_matching_rows():
    cloud = _cloud(n=6)
    index = cloud.index_of([3, 1, 4])
    selection = cloud.select(index)
    assert selection.space_id == cloud.space_id
    assert selection.cloud_id == cloud.cloud_id
    assert np.array_equal(selection.vectors, cloud.vectors[[3, 1, 4]])
    assert selection.item_ids == ("item-3", "item-1", "item-4")


def test_select_raises_when_index_comes_from_another_cloud_in_same_space():
    """The historical defect: A-cloud positions applied to B-cloud rows."""
    labels = _cloud(n=6, cloud_id="label_names", seed=1)
    texts = _cloud(n=9, cloud_id="real_texts", seed=2)
    label_index = labels.index_of([0, 1, 2])
    with pytest.raises(CloudMismatchError, match="label_names"):
        texts.select(label_index)


def test_select_raises_when_index_comes_from_another_space():
    here = _cloud(n=6, cloud_id="label_names", space_id=SPACE, seed=1)
    elsewhere = _cloud(n=6, cloud_id="label_names", space_id=OTHER_SPACE, seed=1)
    foreign_index = elsewhere.index_of([0, 1])
    with pytest.raises(SpaceMismatchError, match="row positions"):
        here.select(foreign_index)


def test_require_same_space_rejects_inconsistent_dimension():
    wide = _cloud(n=4, dim=8, cloud_id="a")
    narrow = _cloud(n=4, dim=4, cloud_id="b")
    with pytest.raises(SpaceMismatchError, match="dimensions"):
        wide.require_same_space(narrow, "coverage_to")


def test_require_same_cloud_rejects_other_cloud_in_same_space():
    a = _cloud(n=4, cloud_id="a")
    b = _cloud(n=4, cloud_id="b")
    with pytest.raises(CloudMismatchError, match="coverage_to"):
        a.require_same_cloud(b, "self_coverage")


# ---------------------------------------------------------------------------
# fps(): contract of the selection
# ---------------------------------------------------------------------------


def test_fps_returns_tagged_index_and_selected_vectors():
    cloud = _cloud(n=12)
    index, selection = fps(cloud, n=4, seed=42)
    assert isinstance(index, CloudIndex)
    assert index.space_id == cloud.space_id
    assert index.cloud_id == cloud.cloud_id
    assert len(index.positions) == 4
    assert selection.space_id == cloud.space_id
    assert selection.cloud_id == cloud.cloud_id
    assert selection.n_items == 4
    # Acceptance: the returned vectors are bit-identical to the source rows the
    # index names.
    assert np.array_equal(selection.vectors, cloud.vectors[list(index.positions)])


def test_fps_selection_carries_item_ids_in_order():
    cloud = _cloud(n=12)
    index, selection = fps(cloud, n=4, seed=7)
    assert selection.item_ids == tuple(cloud.item_ids[p] for p in index.positions)  # type: ignore[index]


def test_fps_index_cannot_be_applied_to_another_cloud():
    """The selected positions are labelled, so they cannot silently index elsewhere."""
    a = _cloud(n=12, cloud_id="cloud_a", seed=3)
    b = _cloud(n=12, cloud_id="cloud_b", seed=4)
    index, _selection = fps(a, n=4, seed=42)
    with pytest.raises(CloudMismatchError):
        b.select(index)


def test_fps_is_deterministic_for_a_seed():
    cloud = _cloud(n=20)
    first, _ = fps(cloud, n=5, seed=123)
    second, _ = fps(cloud, n=5, seed=123)
    assert first.positions == second.positions


def test_fps_selects_distinct_positions_and_full_selection_is_a_permutation():
    cloud = _cloud(n=15)
    index, selection = fps(cloud, n=15, seed=42)
    assert sorted(index.positions) == list(range(15))
    assert selection.n_items == 15


def test_fps_n_one_returns_single_position():
    cloud = _cloud(n=10)
    index, selection = fps(cloud, n=1, seed=42)
    assert len(index.positions) == 1
    assert selection.n_items == 1


@pytest.mark.parametrize("n", [0, -1, 11])
def test_fps_n_out_of_range_raises(n):
    cloud = _cloud(n=10)
    with pytest.raises(ValueError, match="n must be"):
        fps(cloud, n=n)


def test_fps_handles_zero_vectors():
    vectors = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cloud = CloudVectors(space_id=SPACE, cloud_id="cloud_a", vectors=vectors)
    index, selection = fps(cloud, n=2, seed=42)
    assert len(index.positions) == 2
    assert selection.n_items == 2
