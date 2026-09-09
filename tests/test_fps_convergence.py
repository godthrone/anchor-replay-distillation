"""Synthetic-data convergence tests for FPS coverage.

Validates the convergence properties defined in the architecture doc §4.3:
monotonicity of domain coverage as *n* increases, full-coverage convergence
at large *n*, and minimum coverage guarantees at small *n*.
"""

from __future__ import annotations

import numpy as np
from ard.core.embeddings import farthest_point_sampling


# ---------------------------------------------------------------------------
# Synthetic data builder
# ---------------------------------------------------------------------------


def _build_synthetic_ontology_embeddings(
    n_domains: int = 10,
    n_capabilities: int = 5,
    n_languages: int = 3,
    n_conv_types: int = 3,
    dim: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """构造合成嵌入：每个 domain 形成独立 cluster，capability 形成子 cluster。

    Returns:
        (embeddings, metadata) — embeddings shape ``(total, dim)``,
        metadata is a list of dicts with keys ``domain``, ``capability``,
        ``language``, ``conv_type``.
    """
    rng = np.random.default_rng(seed)
    total = n_domains * n_capabilities * n_languages * n_conv_types
    domain_centers = rng.normal(0, 3, (n_domains, dim))
    cap_offsets = rng.normal(0, 1.0, (n_domains, n_capabilities, dim))
    lang_offsets = rng.normal(0, 0.3, (n_languages, dim))
    conv_offsets = rng.normal(0, 0.2, (n_conv_types, dim))

    embeddings = np.zeros((total, dim))
    metadata: list[dict[str, str]] = []
    idx = 0
    for di in range(n_domains):
        for ci in range(n_capabilities):
            for li in range(n_languages):
                for ti in range(n_conv_types):
                    vec = (
                        domain_centers[di]
                        + cap_offsets[di, ci]
                        + lang_offsets[li]
                        + conv_offsets[ti]
                    )
                    embeddings[idx] = vec
                    metadata.append(
                        {
                            "domain": f"d_{di}",
                            "capability": f"c_{ci}",
                            "language": f"l_{li}",
                            "conv_type": f"t_{ti}",
                        }
                    )
                    idx += 1
    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1e-12, norms)
    return embeddings, metadata


def _coverage(
    indices: list[int], metadata: list[dict[str, str]], key: str
) -> float:
    """计算选定索引在给定维度上的覆盖率（0.0–1.0）。"""
    covered = {metadata[i][key] for i in indices}
    all_vals = {m[key] for m in metadata}
    return len(covered) / len(all_vals)


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------


def test_fps_convergence_domain_monotonic():
    """知识域覆盖率随 n 增大单调递增。"""
    embeddings, metadata = _build_synthetic_ontology_embeddings()
    prev = 0.0
    for n in [50, 100, 200, 300, 400, 450]:
        indices = farthest_point_sampling(embeddings, n=n, seed=42)
        cov = _coverage(indices, metadata, "domain")
        assert cov >= prev, f"n={n}: {cov:.2%} < prev {prev:.2%}"
        prev = cov


def test_fps_convergence_full_coverage():
    """n=500 时各维度覆盖率 ≥ 95%。"""
    embeddings, metadata = _build_synthetic_ontology_embeddings(
        n_domains=20, n_capabilities=8, n_languages=4, n_conv_types=5
    )  # 总组合: 3200
    indices = farthest_point_sampling(embeddings, n=500, seed=42)
    for dim in ["domain", "capability", "language", "conv_type"]:
        cov = _coverage(indices, metadata, dim)
        assert cov >= 0.95, f"n=500, {dim}={cov:.2%}"


def test_fps_n50_min_coverage():
    """n=50 时知识域覆盖率 ≥ 50%，能力覆盖率 ≥ 60%。"""
    embeddings, metadata = _build_synthetic_ontology_embeddings()
    indices = farthest_point_sampling(embeddings, n=50, seed=42)
    assert _coverage(indices, metadata, "domain") >= 0.5
    assert _coverage(indices, metadata, "capability") >= 0.6


def test_fps_convergence_capability_monotonic():
    """能力覆盖率随 n 增大单调递增。"""
    embeddings, metadata = _build_synthetic_ontology_embeddings()
    prev = 0.0
    for n in [50, 100, 200, 300, 400, 450]:
        indices = farthest_point_sampling(embeddings, n=n, seed=42)
        cov = _coverage(indices, metadata, "capability")
        assert cov >= prev, f"n={n}: capability {cov:.2%} < prev {prev:.2%}"
        prev = cov


def test_fps_convergence_language_monotonic():
    """语言覆盖率随 n 增大单调递增。"""
    embeddings, metadata = _build_synthetic_ontology_embeddings()
    prev = 0.0
    for n in [50, 100, 200, 300, 400, 450]:
        indices = farthest_point_sampling(embeddings, n=n, seed=42)
        cov = _coverage(indices, metadata, "language")
        assert cov >= prev, f"n={n}: language {cov:.2%} < prev {prev:.2%}"
        prev = cov


def test_fps_n_equals_total_full_coverage():
    """n 等于总组合数时，所有维度覆盖率 = 100%。"""
    embeddings, metadata = _build_synthetic_ontology_embeddings(
        n_domains=5, n_capabilities=3, n_languages=2, n_conv_types=2
    )  # 总组合: 60
    total = len(metadata)
    indices = farthest_point_sampling(embeddings, n=total, seed=42)
    for dim in ["domain", "capability", "language", "conv_type"]:
        cov = _coverage(indices, metadata, dim)
        assert cov == 1.0, f"n={total}, {dim}={cov:.2%}"