"""Unit tests for FPS correctness and edge cases (space-safe API).

Tests :func:`ard.core.cloud.fps` for basic correctness, boundary conditions,
determinism, and zero-vector handling.  The algorithm is the historical greedy
farthest-point rule; only its contract changed — it now returns a tagged
:class:`~ard.core.cloud.CloudIndex` plus the selected vectors.
"""

from __future__ import annotations

import numpy as np
import pytest
from ard.core.cloud import CloudVectors, fps


def _fps_positions(embeddings: np.ndarray, n: int, seed: int | None = None) -> list[int]:
    """Run space-safe FPS and return the row positions (test-local helper)."""
    cloud = CloudVectors(
        space_id=f"test-embeddings:{embeddings.shape[1]}",
        cloud_id="test_embeddings",
        vectors=embeddings,
    )
    index, _selection = fps(cloud, n=n, seed=seed)
    return list(index.positions)



# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_fps_selects_diverse_points():
    """FPS 选出 3 个点应分别来自 3 个 cluster。"""
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],  # cluster A
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],  # cluster B
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 0.9],  # cluster C
        ]
    )
    indices = _fps_positions(embeddings, n=3, seed=42)
    clusters = [0, 0, 1, 1, 2, 2]
    selected = {clusters[i] for i in indices}
    assert selected == {0, 1, 2}


def test_fps_n_equals_1():
    """n=1 应返回单个有效索引。"""
    embeddings = np.random.default_rng(42).normal(0, 1, (100, 64))
    indices = _fps_positions(embeddings, n=1, seed=42)
    assert len(indices) == 1
    assert 0 <= indices[0] < 100


def test_fps_n_equals_N():
    """n=N 时应返回所有索引（全选）。"""
    embeddings = np.random.default_rng(42).normal(0, 1, (100, 64))
    indices = _fps_positions(embeddings, n=100, seed=42)
    assert len(indices) == 100
    assert sorted(indices) == list(range(100))


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_fps_empty_raises():
    """空向量矩阵在载体边界上就被拒绝（不再有"空数组"这一非法输入）。"""
    with pytest.raises(ValueError, match="at least one row"):
        _fps_positions(np.empty((0, 64)), n=1)


def test_fps_n_out_of_range_raises():
    """n=0 或 n > N 应抛出 ValueError。"""
    embeddings = np.ones((10, 5))
    with pytest.raises(ValueError):
        _fps_positions(embeddings, n=0)
    with pytest.raises(ValueError):
        _fps_positions(embeddings, n=11)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_fps_deterministic():
    """相同 seed 应产生相同结果。"""
    embeddings = np.random.default_rng(42).normal(0, 1, (100, 64))
    a = _fps_positions(embeddings, n=10, seed=42)
    b = _fps_positions(embeddings, n=10, seed=42)
    assert a == b


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_fps_zero_vector_handling():
    """包含零向量时不应崩溃，应正常返回结果。"""
    embeddings = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    indices = _fps_positions(embeddings, n=2, seed=42)
    assert len(indices) == 2


def test_fps_no_seed_is_deterministic():
    """不传 seed 时使用固定默认 seed，结果应可复现。"""
    embeddings = np.random.default_rng(42).normal(0, 1, (100, 64))
    a = _fps_positions(embeddings, n=10)
    b = _fps_positions(embeddings, n=10)
    assert a == b


def test_fps_all_identical_vectors():
    """所有向量相同时，FPS 应选出 n 个不同索引（任意选，不崩溃）。"""
    embeddings = np.ones((50, 16))
    indices = _fps_positions(embeddings, n=5, seed=42)
    assert len(indices) == 5
    assert len(set(indices)) == 5  # 所有索引不同


def test_fps_small_n():
    """n=2 时检查返回的索引有效。"""
    embeddings = np.random.default_rng(99).normal(0, 1, (10, 8))
    indices = _fps_positions(embeddings, n=2, seed=42)
    assert len(indices) == 2
    assert all(0 <= i < 10 for i in indices)
    assert indices[0] != indices[1]