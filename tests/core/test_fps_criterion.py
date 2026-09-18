"""Tests for the selectable greedy criterion of :func:`ard.core.cloud.fps`.

Covers the ``"sum"`` (total-blankness / greedy facility-location) rule added on
top of the historical ``"max"`` (farthest-point) rule:

* equivalence with an independent from-scratch reference implementation;
* the historical default, byte for byte;
* a synthetic uniform-sphere ground-truth check that ``"sum"`` generalises —
  it beats uniform-random selection on a holdout it was never shown;
* degenerate inputs (``k=1``, ``k=n``, duplicate and zero rows);
* the space/cloud identity contract still holds on the new path;
* contract enforcement: an unknown criterion and an over-budget cloud are
  refused, never silently degraded.
"""

from __future__ import annotations

import numpy as np
import pytest

from ard.core.cloud import (
    CRITERION_MAX,
    CRITERION_SUM,
    FPS_SUM_MATRIX_MAX_BYTES,
    CloudMismatchError,
    CloudVectors,
    SpaceMismatchError,
    fps,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1e-12, norms)
    return vectors / norms


def _cloud(vectors: np.ndarray, *, tag: str = "a", space: str = "s:test") -> CloudVectors:
    return CloudVectors(
        space_id=space,
        cloud_id=f"cloud.{tag}",
        vectors=vectors,
        item_ids=tuple(str(i) for i in range(vectors.shape[0])),
    )


def _mean_radius_holdout(
    pool: np.ndarray, holdout: np.ndarray, positions: list[int]
) -> float:
    """``r_mean``: mean over holdout rows of the cosine distance to the nearest
    selected row (smaller = better)."""
    nearest = np.min(
        1.0 - _unit_rows(holdout) @ _unit_rows(pool)[np.asarray(positions)].T, axis=1
    )
    return float(nearest.mean())


# ---------------------------------------------------------------------------
# 1. correctness against an independent reference
# ---------------------------------------------------------------------------


def _reference_sum_scores(
    unit: np.ndarray, selected: list[int]
) -> np.ndarray:
    """Scores of the ``sum`` rule against *selected*, from the formula only.

    Recomputes the nearest-selected distance from scratch and evaluates
    ``Σ_j min(min_dist[j], d(i, j))`` with a matrix-vector product per candidate
    — deliberately different arithmetic from the production path (which keeps a
    running minimum and uses one matrix-matrix product), so a real objective bug
    cannot cancel out.
    """
    n = unit.shape[0]
    if selected:
        nearest = np.min(1.0 - unit @ unit[np.asarray(selected)].T, axis=1)
    else:
        nearest = np.full(n, np.inf)
    scores = np.full(n, np.inf)
    for candidate in range(n):
        if candidate in selected:
            continue
        scores[candidate] = float(np.minimum(nearest, 1.0 - unit @ unit[candidate]).sum())
    return scores


def test_sum_criterion_is_greedy_against_an_independent_reference():
    """Every row the production rule picks must minimise the reference score.

    Exact sequence equality is *not* asserted: two candidates can score
    identically (an exact tie, e.g. symmetric or duplicated rows) and then which
    one ``argmin`` returns is decided by the last bits of two different BLAS
    kernels. The greedy property — the chosen row is a minimiser — is the
    statement that matters.
    """
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(60, 12))
    unit = _unit_rows(vectors)
    checked_steps = 0
    for k in (1, 3, 7, 20):
        for seed in (42, 7):
            index, _ = fps(_cloud(vectors), k, seed=seed, criterion=CRITERION_SUM)
            positions = list(index.positions)
            assert len(set(positions)) == k
            selected: list[int] = []
            for chosen in positions:
                if not selected:
                    selected.append(chosen)
                    continue
                scores = _reference_sum_scores(unit, selected)
                best = float(np.nanmin(scores))
                assert scores[chosen] <= best + 1e-8, (
                    f"k={k} seed={seed}: row {chosen} scored {scores[chosen]!r} "
                    f"but {best!r} was reachable"
                )
                selected.append(chosen)
                checked_steps += 1
    assert checked_steps >= 50


def test_default_is_the_historical_max_rule():
    rng = np.random.default_rng(12)
    vectors = rng.normal(size=(80, 16))
    for k in (1, 5, 40):
        implicit, _ = fps(_cloud(vectors), k, seed=3)
        explicit, _ = fps(_cloud(vectors), k, seed=3, criterion=CRITERION_MAX)
        assert list(implicit.positions) == list(explicit.positions)


def test_unknown_criterion_is_refused_not_defaulted():
    vectors = np.random.default_rng(13).normal(size=(10, 4))
    with pytest.raises(ValueError, match="criterion must be one of"):
        fps(_cloud(vectors), 3, criterion="Mean")
    with pytest.raises(ValueError, match="criterion must be one of"):
        fps(_cloud(vectors), 3, criterion="")


def test_sum_cloud_above_memory_budget_is_refused(monkeypatch):
    # 8 rows, but pretend every cloud is over budget: the refusal must happen
    # before the distance matrix is built, and must not fall back to "max".
    monkeypatch.setattr("ard.core.cloud.FPS_SUM_MATRIX_MAX_BYTES", 8 * 36)
    vectors = np.random.default_rng(14).normal(size=(40, 3))
    with pytest.raises(ValueError, match="exceeds FPS_SUM_MATRIX_MAX_BYTES"):
        fps(_cloud(vectors), 5, criterion=CRITERION_SUM)
    # The historical criterion is unaffected by the ``sum`` budget.
    index, _ = fps(_cloud(vectors), 5)
    assert len(index.positions) == 5
    assert FPS_SUM_MATRIX_MAX_BYTES > 0


# ---------------------------------------------------------------------------
# 2. synthetic uniform-sphere ground truth: "sum" generalises to a holdout
# ---------------------------------------------------------------------------


def test_sum_criterion_beats_random_on_a_uniform_sphere_holdout():
    """Uniform points on a sphere have no cluster structure, so this is a
    *generative* check: the greedy sum rule must still cover an unseen holdout
    better on average than uniform random selection does."""
    rng = np.random.default_rng(20260917)
    pool = rng.normal(size=(200, 8))
    holdout = rng.normal(size=(4000, 8))
    k = 12
    restarts = 20

    sum_scores = []
    random_scores = []
    for restart in range(restarts):
        sum_index, _ = fps(_cloud(pool), k, seed=restart, criterion=CRITERION_SUM)
        sum_scores.append(_mean_radius_holdout(pool, holdout, list(sum_index.positions)))
        picked = rng.choice(pool.shape[0], size=k, replace=False)
        random_scores.append(_mean_radius_holdout(pool, holdout, [int(i) for i in picked]))

    sum_mean = float(np.mean(sum_scores))
    random_mean = float(np.mean(random_scores))
    assert sum_mean < random_mean, (sum_mean, random_mean)
    # Direction is the claim; a wide margin guards against a lucky seed.
    assert sum_mean <= 0.97 * random_mean, (sum_mean, random_mean)


# ---------------------------------------------------------------------------
# 3. degenerate inputs
# ---------------------------------------------------------------------------


def test_sum_criterion_k_equals_one_matches_max():
    vectors = np.random.default_rng(21).normal(size=(30, 6))
    for seed in (0, 1, 99):
        sum_index, _ = fps(_cloud(vectors), 1, seed=seed, criterion=CRITERION_SUM)
        max_index, _ = fps(_cloud(vectors), 1, seed=seed, criterion=CRITERION_MAX)
        assert list(sum_index.positions) == list(max_index.positions)
        assert len(sum_index.positions) == 1


def test_sum_criterion_k_equals_n_is_a_permutation():
    vectors = np.random.default_rng(22).normal(size=(25, 5))
    for criterion in (CRITERION_MAX, CRITERION_SUM):
        index, selection = fps(_cloud(vectors), 25, seed=5, criterion=criterion)
        assert sorted(index.positions) == list(range(25))
        assert np.array_equal(selection.vectors, vectors[list(index.positions)])


def test_sum_criterion_with_duplicate_and_zero_rows():
    base = np.random.default_rng(23).normal(size=(8, 5))
    # rows 0/1 identical, row 2 all zeros, rows 3/4 identical.
    vectors = np.vstack([base[0], base[0], np.zeros(5), base[3], base[3], base[5:]])
    assert np.any(np.all(vectors == 0.0, axis=1))
    first, _ = fps(_cloud(vectors), 4, seed=8, criterion=CRITERION_SUM)
    second, _ = fps(_cloud(vectors), 4, seed=8, criterion=CRITERION_SUM)
    assert len(first.positions) == 4
    assert len(set(first.positions)) == 4
    assert list(first.positions) == list(second.positions), "must be deterministic"


def test_sum_criterion_scores_are_finite_on_a_cloud_with_identical_rows():
    vectors = np.tile(np.random.default_rng(24).normal(size=(1, 5)), (12, 1))
    index, _ = fps(_cloud(vectors), 6, seed=1, criterion=CRITERION_SUM)
    assert len(set(index.positions)) == 6


# ---------------------------------------------------------------------------
# 4. the space/cloud identity contract still holds on the new path
# ---------------------------------------------------------------------------


def test_sum_criterion_index_cannot_address_another_cloud():
    rng = np.random.default_rng(31)
    first = _cloud(rng.normal(size=(20, 4)), tag="a", space="s:shared")
    second = _cloud(rng.normal(size=(20, 4)), tag="b", space="s:shared")
    index, selection = fps(first, 5, seed=2, criterion=CRITERION_SUM)

    assert index.cloud_id == first.cloud_id
    assert index.space_id == first.space_id
    assert np.array_equal(selection.vectors, first.vectors[list(index.positions)])
    with pytest.raises(CloudMismatchError):
        second.select(index)


def test_sum_criterion_index_cannot_cross_embedding_spaces():
    rng = np.random.default_rng(32)
    here = _cloud(rng.normal(size=(20, 4)), tag="a", space="s:here")
    elsewhere = _cloud(rng.normal(size=(20, 4)), tag="a", space="s:elsewhere")
    index, _ = fps(here, 5, seed=2, criterion=CRITERION_SUM)
    with pytest.raises(SpaceMismatchError):
        elsewhere.select(index)
