"""Tests for :mod:`ard.core.coverage` — space-safe coverage measurement.

The regression guard is a frozen copy of the historical analysis helper
(``coverage(U_all, sel)`` from the 0916 / S2 evidence scripts).  It exists only
here, as the reference the new interface must keep reproducing bit for bit: the
space-safe API may not change any number, only refuse the meaningless ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from ard.core.cloud import CloudMismatchError, CloudVectors, SpaceMismatchError, fps
from ard.core.coverage import CoverageStats, coverage_to, self_coverage
from ard.core.embeddings import embedding_space_id

SPACE = "test-embedder:16"
OTHER_SPACE = "other-embedder:16"


# ---------------------------------------------------------------------------
# Frozen reference — verbatim from the historical evidence scripts
# (.local/.../task-language-space-gap/evidence/analyze_embeddings.py:145-154 and
# .local/.../task-s2-probe/evidence/carrier_probe.py:115-124).  Do not "improve"
# this: it is the baseline the new numbers are compared against.
# ---------------------------------------------------------------------------
def legacy_coverage(U_all: np.ndarray, sel: list[int]) -> dict:  # noqa: N803 — frozen copy
    d = 1.0 - U_all @ U_all[sel].T
    nearest = d.min(axis=1)
    return {
        "max": float(nearest.max()),
        "mean": float(nearest.mean()),
        "median": float(np.median(nearest)),
        "p90": float(np.percentile(nearest, 90)),
        "p95": float(np.percentile(nearest, 95)),
    }


def _unit_cloud(
    n: int,
    dim: int = 16,
    *,
    space_id: str = SPACE,
    cloud_id: str = "cloud_a",
    seed: int = 0,
) -> CloudVectors:
    vectors = np.random.default_rng(seed).normal(0.0, 1.0, (n, dim))
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return CloudVectors(
        space_id=space_id,
        cloud_id=cloud_id,
        vectors=vectors,
        item_ids=tuple(f"item-{i}" for i in range(n)),
    )


# ---------------------------------------------------------------------------
# Regression: identical numbers to the historical implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("n_selected", [1, 3, 10])
def test_self_coverage_is_bit_identical_to_legacy(seed, n_selected):
    cloud = _unit_cloud(30)
    _index, selection = fps(cloud, n=n_selected, seed=seed)
    new = self_coverage(cloud, selection).as_dict()
    # The legacy call takes raw positions; here they are the ones fps reported,
    # so both run the identical arithmetic on the identical rows.
    legacy_positions = index_of_items(cloud, selection)
    old = legacy_coverage(cloud.vectors, legacy_positions)
    for key, old_value in old.items():
        assert new[key] == old_value, f"{key}: {new[key]!r} != {old_value!r}"


def index_of_items(cloud: CloudVectors, selection: CloudVectors) -> list[int]:
    """Recover the row positions of *selection* inside *cloud* (test helper).

    The public API deliberately makes this unnecessary; the regression test needs
    the raw positions only to feed the frozen legacy function.
    """
    assert cloud.item_ids is not None
    assert selection.item_ids is not None
    lookup = {item_id: position for position, item_id in enumerate(cloud.item_ids)}
    return [lookup[item_id] for item_id in selection.item_ids]


@pytest.mark.parametrize("seed", [0, 42])
def test_self_coverage_matches_legacy_on_ontology_vectors(seed, embeddings_data):
    """Same check on the shipped 1024-dim embeddings (real data, not synthetic)."""
    items = embeddings_data["items"]
    names = sorted(items["knowledge_domains"])
    vectors = np.array([items["knowledge_domains"][name] for name in names], dtype=np.float64)
    cloud = CloudVectors(
        space_id=embedding_space_id(embeddings_data),
        cloud_id="knowledge_domains",
        vectors=vectors,
        item_ids=tuple(names),
    )
    _index, selection = fps(cloud, n=6, seed=seed)
    new = self_coverage(cloud, selection).as_dict()
    old = legacy_coverage(vectors, index_of_items(cloud, selection))
    assert new == old


# ---------------------------------------------------------------------------
# The cross-cloud guard
# ---------------------------------------------------------------------------


def test_self_coverage_rejects_selection_from_another_cloud():
    """coverage(U_real, fps(U_labels, k)) — the historical defect — must raise."""
    labels = _unit_cloud(12, cloud_id="label_names", seed=1)
    texts = _unit_cloud(20, cloud_id="real_texts", seed=2)
    _index, label_selection = fps(labels, n=4, seed=42)
    with pytest.raises(CloudMismatchError, match="label_names"):
        self_coverage(texts, label_selection)


def test_self_coverage_rejects_selection_from_another_space():
    here = _unit_cloud(12, cloud_id="real_texts", space_id=SPACE)
    there = _unit_cloud(12, cloud_id="real_texts", space_id=OTHER_SPACE)
    _index, foreign_selection = fps(there, n=3, seed=42)
    with pytest.raises(SpaceMismatchError, match="different embedding spaces"):
        self_coverage(here, foreign_selection)


# ---------------------------------------------------------------------------
# Cross-cloud measurement (the intended, explicit case)
# ---------------------------------------------------------------------------


def test_coverage_to_allows_different_clouds_in_one_space():
    labels = _unit_cloud(12, cloud_id="label_names", seed=1)
    texts = _unit_cloud(20, cloud_id="real_texts", seed=2)
    _index, label_selection = fps(labels, n=4, seed=42)
    stats = coverage_to(texts, label_selection)
    # Independently recompute the vector-level definition.
    nearest = (1.0 - texts.vectors @ label_selection.vectors.T).min(axis=1)
    assert stats.n_target == 20
    assert stats.n_selected == 4
    assert stats.max == float(nearest.max())
    assert stats.mean == float(nearest.mean())
    assert stats.p90 == float(np.percentile(nearest, 90))
    assert stats.same_cloud is False
    assert stats.target_cloud_id == "real_texts"
    assert stats.selected_cloud_id == "label_names"


def test_coverage_to_rejects_different_spaces():
    a = _unit_cloud(10, cloud_id="real_texts", space_id=SPACE)
    b = _unit_cloud(10, cloud_id="label_names", space_id=OTHER_SPACE)
    with pytest.raises(SpaceMismatchError, match="not comparable"):
        coverage_to(a, b)


def test_coverage_to_rejects_self_contradictory_space_identity():
    wide = _unit_cloud(10, dim=16, cloud_id="real_texts")
    narrow = _unit_cloud(10, dim=8, cloud_id="label_names")
    with pytest.raises(SpaceMismatchError, match="dimensions"):
        coverage_to(wide, narrow)


# ---------------------------------------------------------------------------
# Invariants (review report §⑥, suggestion 4)
# ---------------------------------------------------------------------------


def test_same_cloud_selection_gives_identical_self_and_cross_coverage():
    cloud = _unit_cloud(25)
    index, selection = fps(cloud, n=5, seed=42)
    same = self_coverage(cloud, selection)
    cross = coverage_to(cloud, cloud.select(index))
    assert same == cross


def test_cross_cloud_coverage_differs_from_self_coverage():
    labels = _unit_cloud(12, cloud_id="label_names", seed=1)
    texts = _unit_cloud(20, cloud_id="real_texts", seed=2)
    index, label_selection = fps(labels, n=4, seed=42)
    cross = coverage_to(texts, label_selection)
    # The historical bug computed self-coverage of the *first rows of the target*
    # instead: a different measurement, and a different number.
    wrong = coverage_to(texts, texts.select(texts.index_of(list(index.positions))))
    assert cross.max != wrong.max


def test_stats_record_their_provenance():
    cloud = _unit_cloud(9, cloud_id="cloud_a")
    _index, selection = fps(cloud, n=3, seed=0)
    stats = self_coverage(cloud, selection)
    assert isinstance(stats, CoverageStats)
    assert stats.target_space_id == SPACE
    assert stats.selected_space_id == SPACE
    assert stats.target_cloud_id == "cloud_a"
    assert stats.selected_cloud_id == "cloud_a"
    assert stats.same_cloud is True
    assert set(stats.as_dict()) == {"max", "mean", "median", "p90", "p95"}
