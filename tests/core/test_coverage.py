"""Tests for :mod:`ard.core.coverage` — the ARD v4 acceptance ruler.

The assertions are pinned to hand-computable constructions rather than to
implementation output: four equidistant unit points on the circle, two opposite
points in one dimension, coincident points, and a single anchor / single target.
Every statistic is checked against its closed-form value, the paired bootstrap
against a manually re-run resampling loop that shares one index draw, and every
boundary (empty set, dimension mismatch, unnormalised vector, NaN, bad ε / B)
against its explicit error.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from ard.core.coverage import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    EPSILON_SCALE_FACTORS,
    CoverageReadout,
    DimensionMismatchError,
    DistanceSample,
    EmptyDistanceSampleError,
    EmptyVectorSetError,
    EpsilonSensitivity,
    InvalidDistancesError,
    InvalidParameterError,
    NoiseBand,
    NonFiniteValuesError,
    PairedBootstrap,
    PairingMismatchError,
    UnnormalizedVectorsError,
    VectorSet,
    VectorShapeError,
    acceptance_readout,
    distance_quantiles,
    epsilon_sensitivity,
    extent,
    nearest_anchor_distances,
    noise_band,
    paired_bootstrap_q95_ci,
    pairwise_distances,
    validate_vector_set,
    within_noise_band,
)

# Four unit points at 0° / 90° / 180° / 270°: every distinct pair is at cosine
# distance exactly 1, so a single anchor at (1, 0) gives d = [0, 1, 2, 1].
CIRCLE = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
UNIT_X = np.array([[1.0, 0.0]])


def _type7(values: np.ndarray, quantile: float) -> float:
    """Hand-rolled Hyndman–Fan type-7 quantile for the expected values."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    position = (ordered.size - 1) * quantile / 100.0
    lower = int(np.floor(position))
    upper = min(lower + 1, ordered.size - 1)
    return float(ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower]))


def _circle_targets() -> VectorSet:
    return VectorSet(name="targets", vectors=CIRCLE)


def _single_anchor() -> VectorSet:
    return VectorSet(name="anchors", vectors=UNIT_X)


# ---------------------------------------------------------------------------
# Readouts on known constructions
# ---------------------------------------------------------------------------


def test_nearest_anchor_distances_on_equidistant_circle() -> None:
    sample = nearest_anchor_distances(_circle_targets(), _single_anchor())
    assert sample.values.tolist() == pytest.approx([0.0, 1.0, 2.0, 1.0])
    assert sample.label == "targets"
    assert sample.unit_id == "targets"


def test_distance_quantiles_are_type7() -> None:
    sample = nearest_anchor_distances(_circle_targets(), _single_anchor())
    stats = distance_quantiles(sample)
    distances = np.array([0.0, 1.0, 2.0, 1.0])
    assert stats.n == 4  # |T| travels with the readouts
    assert stats.q50 == pytest.approx(_type7(distances, 50.0))
    assert stats.q50 == pytest.approx(1.0)
    assert stats.q90 == pytest.approx(_type7(distances, 90.0))
    assert stats.q90 == pytest.approx(1.7)
    assert stats.q95 == pytest.approx(_type7(distances, 95.0))
    assert stats.q95 == pytest.approx(1.85)
    assert stats.r_max == pytest.approx(2.0)


def test_single_anchor_and_single_target_are_defined() -> None:
    """M=1 and |T|=1: the boundary of the definition, not an error."""
    targets = VectorSet(name="targets", vectors=np.array([[1.0, 0.0]]))
    anchors = VectorSet(name="anchors", vectors=np.array([[-1.0, 0.0]]))
    sample = nearest_anchor_distances(targets, anchors)
    stats = distance_quantiles(sample)
    assert sample.values.tolist() == pytest.approx([2.0])
    assert (stats.n, stats.q50, stats.q90, stats.q95, stats.r_max) == pytest.approx(
        (1, 2.0, 2.0, 2.0, 2.0)
    )
    assert extent(sample, 1.999) == 0.0


def test_one_dimensional_opposite_points() -> None:
    """Dimension 1: d = 2 * q / 100 for the two-point sample [0, 2]."""
    targets = VectorSet(name="targets", vectors=np.array([[1.0], [-1.0]]))
    anchors = VectorSet(name="anchors", vectors=np.array([[1.0]]))
    stats = distance_quantiles(nearest_anchor_distances(targets, anchors))
    assert stats.q50 == pytest.approx(1.0)
    assert stats.q90 == pytest.approx(1.8)
    assert stats.q95 == pytest.approx(1.9)
    assert stats.r_max == pytest.approx(2.0)


def test_coincident_points_are_fully_covered() -> None:
    targets = VectorSet(name="targets", vectors=np.tile([[0.6, 0.8]], (5, 1)))
    anchors = VectorSet(name="anchors", vectors=np.array([[0.6, 0.8]]))
    readout = acceptance_readout(targets, anchors, epsilon=0.0)
    assert readout.quantiles.q95 == 0.0
    assert readout.quantiles.r_max == 0.0
    assert readout.extent == 1.0


@pytest.mark.parametrize(
    ("epsilon", "expected"), [(0.0, 0.25), (1.0, 0.75), (1.999, 0.75), (2.0, 1.0)]
)
def test_extent_is_the_mean_indicator(epsilon, expected) -> None:
    sample = nearest_anchor_distances(_circle_targets(), _single_anchor())
    assert extent(sample, epsilon) == pytest.approx(expected)


def test_epsilon_sensitivity_band_scales_the_nominal_epsilon() -> None:
    sample = nearest_anchor_distances(_circle_targets(), _single_anchor())
    band = epsilon_sensitivity(sample, 1.0)
    assert isinstance(band, EpsilonSensitivity)
    assert EPSILON_SCALE_FACTORS == (0.95, 1.00, 1.05)
    assert band.extent_at_095 == pytest.approx(0.25)  # d <= 0.95
    assert band.extent_at_100 == pytest.approx(0.75)  # d <= 1.00
    assert band.extent_at_105 == pytest.approx(0.75)  # d <= 1.05
    assert band.extent_at_100 == extent(sample, 1.0)


def test_acceptance_readout_composes_every_readout() -> None:
    readout = acceptance_readout(_circle_targets(), _single_anchor(), epsilon=1.0)
    assert isinstance(readout, CoverageReadout)
    assert readout.label == "targets"
    assert readout.n_anchor == 1
    assert readout.n_target == 4
    assert readout.quantiles.q95 == pytest.approx(1.85)
    assert readout.epsilon_band.epsilon == 1.0


# ---------------------------------------------------------------------------
# Paired target-point bootstrap
# ---------------------------------------------------------------------------


def test_paired_bootstrap_matches_a_manual_paired_resampling_run() -> None:
    a = np.linspace(0.0, 1.0, 12) ** 1.3
    b = np.linspace(0.1, 1.4, 12) ** 1.1
    result = paired_bootstrap_q95_ci(
        DistanceSample(label="arm-a", values=a, unit_id="targets"),
        DistanceSample(label="arm-b", values=b, unit_id="targets"),
        n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
        confidence_level=0.95,
        seed=7,
    )
    generator = np.random.default_rng(7)
    positions = generator.integers(0, a.size, size=(DEFAULT_BOOTSTRAP_RESAMPLES, a.size))
    differences = np.percentile(a[positions], 95.0, axis=1, method="linear") - np.percentile(
        b[positions], 95.0, axis=1, method="linear"
    )
    assert DEFAULT_BOOTSTRAP_RESAMPLES == 2000
    assert isinstance(result, PairedBootstrap)
    assert result.n_resamples == 2000
    assert result.point_estimate == pytest.approx(_type7(a, 95.0) - _type7(b, 95.0))
    assert result.ci_low == _type7(differences, 2.5)
    assert result.ci_high == _type7(differences, 97.5)
    assert result.ci_low <= result.ci_high


def test_paired_bootstrap_of_a_constant_offset_collapses_to_that_offset() -> None:
    """Pairing is structural: with a constant offset every resample gives it back."""
    base = np.linspace(0.0, 0.9, 10)
    result = paired_bootstrap_q95_ci(
        DistanceSample(label="shifted", values=base + 0.5, unit_id="targets"),
        DistanceSample(label="base", values=base, unit_id="targets"),
        n_resamples=500,
        seed=3,
    )
    assert result.point_estimate == pytest.approx(0.5, abs=1e-12)
    assert result.ci_low == pytest.approx(0.5, abs=1e-12)
    assert result.ci_high == pytest.approx(0.5, abs=1e-12)


def test_paired_bootstrap_is_reproducible_per_seed_and_varies_across_seeds() -> None:
    a = np.sort((np.arange(40) / 39.0) ** 1.7)
    b = np.sort((np.arange(40) / 39.0) ** 0.6) * 0.3
    sample_a = DistanceSample(label="a", values=a, unit_id="t")
    sample_b = DistanceSample(label="b", values=b, unit_id="t")
    first = paired_bootstrap_q95_ci(sample_a, sample_b, n_resamples=300, seed=11)
    same = paired_bootstrap_q95_ci(sample_a, sample_b, n_resamples=300, seed=11)
    other = paired_bootstrap_q95_ci(sample_a, sample_b, n_resamples=300, seed=12)
    assert first == same
    assert (first.ci_low, first.ci_high) != (other.ci_low, other.ci_high)
    assert first.ci_low <= first.ci_high


def test_paired_bootstrap_rejects_samples_of_different_length() -> None:
    with pytest.raises(PairingMismatchError, match="one entry per shared target point"):
        paired_bootstrap_q95_ci(
            DistanceSample(label="a", values=np.array([0.1, 0.2]), unit_id="t"),
            DistanceSample(label="b", values=np.array([0.1]), unit_id="t"),
        )


def test_paired_bootstrap_rejects_samples_from_different_target_sets() -> None:
    with pytest.raises(PairingMismatchError, match="same target set"):
        paired_bootstrap_q95_ci(
            DistanceSample(label="a", values=np.array([0.1, 0.2]), unit_id="targets-A"),
            DistanceSample(label="b", values=np.array([0.1, 0.2]), unit_id="targets-B"),
        )


@pytest.mark.parametrize("n_resamples", [0, -1, 1.5, True])
def test_paired_bootstrap_rejects_bad_resample_count(n_resamples) -> None:
    sample = DistanceSample(label="a", values=np.array([0.1, 0.2]), unit_id="t")
    with pytest.raises(InvalidParameterError, match="n_resamples"):
        paired_bootstrap_q95_ci(sample, sample, n_resamples=n_resamples)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.5, float("nan")])
def test_paired_bootstrap_rejects_bad_confidence_level(level) -> None:
    sample = DistanceSample(label="a", values=np.array([0.1, 0.2]), unit_id="t")
    with pytest.raises(InvalidParameterError, match="confidence_level"):
        paired_bootstrap_q95_ci(sample, sample, confidence_level=level)


# ---------------------------------------------------------------------------
# Noise band
# ---------------------------------------------------------------------------


def test_pairwise_distances_on_a_known_triangle() -> None:
    """Three unit points at 0° / 90° / 180°: pairs are 1, 2, 1."""
    repeats = VectorSet(name="cell-7", vectors=np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
    sample = pairwise_distances(repeats)
    assert sample.values.tolist() == pytest.approx([1.0, 2.0, 1.0])
    assert sample.label == "cell-7:pairs"
    assert sample.unit_id == "cell-7"


def test_pairwise_distances_need_at_least_two_repeats() -> None:
    with pytest.raises(EmptyVectorSetError, match="at least two rows"):
        pairwise_distances(VectorSet(name="cell-7", vectors=np.array([[1.0, 0.0]])))


def test_noise_band_edges_are_q50_and_max() -> None:
    band = noise_band(DistanceSample(label="cell-7", values=np.array([0.1, 0.2, 0.3, 0.4])))
    assert isinstance(band, NoiseBand)
    assert band.n_pairs == 4
    assert band.lower == pytest.approx(0.25)  # type-7 q50
    assert band.upper == pytest.approx(0.4)


def test_within_noise_band_is_inclusive_at_both_edges() -> None:
    band = noise_band(DistanceSample(label="cell", values=np.array([0.1, 0.2, 0.3, 0.4])))
    assert within_noise_band(0.25, band) is True
    assert within_noise_band(0.4, band) is True
    assert within_noise_band(0.2, band) is False
    assert within_noise_band(0.9, band) is False


def test_within_noise_band_rejects_non_finite_value() -> None:
    band = noise_band(DistanceSample(label="cell", values=np.array([0.1, 0.2])))
    with pytest.raises(NonFiniteValuesError, match="finite"):
        within_noise_band(float("nan"), band)


# ---------------------------------------------------------------------------
# Boundaries (§2.3): every invalid input raises, none passes silently
# ---------------------------------------------------------------------------


def test_empty_vector_set_is_rejected() -> None:
    with pytest.raises(EmptyVectorSetError, match="no rows"):
        VectorSet(name="targets", vectors=np.zeros((0, 2)))


def test_one_dimensional_input_is_rejected() -> None:
    # Deliberately the wrong runtime shape: that is exactly the boundary under test.
    with pytest.raises(VectorShapeError, match="2-D|matrix"):
        VectorSet(name="targets", vectors=[1.0, 0.0])  # type: ignore[arg-type]


def test_zero_column_input_is_rejected() -> None:
    with pytest.raises(VectorShapeError, match="got shape"):
        VectorSet(name="targets", vectors=np.zeros((2, 0)))


def test_non_numeric_input_is_rejected() -> None:
    with pytest.raises(VectorShapeError, match="numeric"):
        VectorSet(name="targets", vectors=[["a", "b"]])  # type: ignore[arg-type]


@pytest.mark.parametrize("norm", [2.0, 0.5, 0.0])
def test_unnormalized_vectors_are_rejected(norm) -> None:
    with pytest.raises(UnnormalizedVectorsError, match="L2 norm"):
        VectorSet(name="targets", vectors=np.array([[norm, 0.0]]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_vectors_are_rejected(bad) -> None:
    with pytest.raises(NonFiniteValuesError, match="NaN or infinity"):
        VectorSet(name="targets", vectors=np.array([[1.0, 0.0], [bad, 0.0]]))


def test_dimension_mismatch_is_rejected() -> None:
    targets = VectorSet(name="targets", vectors=np.array([[1.0, 0.0]]))
    anchors = VectorSet(name="anchors", vectors=np.array([[1.0, 0.0, 0.0]]))
    with pytest.raises(DimensionMismatchError, match="dimension"):
        nearest_anchor_distances(targets, anchors)


def test_validate_vector_set_catches_an_unchecked_construction() -> None:
    forged = VectorSet.model_construct(name="targets", vectors=np.zeros((2, 2)))
    with pytest.raises(UnnormalizedVectorsError):
        validate_vector_set(forged)


def test_empty_distance_sample_is_rejected() -> None:
    with pytest.raises(EmptyDistanceSampleError, match="empty"):
        DistanceSample(label="d", values=[])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ([0.1, float("nan")], NonFiniteValuesError),
        ([0.1, -0.5], InvalidDistancesError),
        ([0.1, 2.5], InvalidDistancesError),
    ],
)
def test_invalid_distance_samples_are_rejected(values, error) -> None:
    with pytest.raises(error):
        DistanceSample(label="d", values=values)  # type: ignore[arg-type]


def test_distance_sample_rejects_non_one_dimensional_values() -> None:
    with pytest.raises(VectorShapeError, match="1-D"):
        DistanceSample(label="d", values=[[0.1, 0.2]])  # type: ignore[arg-type]


def test_distance_sample_coerces_numeric_sequences_to_float64() -> None:
    # The list is the subject of the test: the model must accept array-like input.
    sample = DistanceSample(label="d", values=[0, 1, 2])  # type: ignore[arg-type]
    assert sample.values.dtype == np.float64
    assert sample.unit_id is None


@pytest.mark.parametrize("epsilon", [-0.5, float("nan"), float("inf")])
def test_extent_rejects_bad_epsilon(epsilon) -> None:
    sample = nearest_anchor_distances(_circle_targets(), _single_anchor())
    with pytest.raises(InvalidParameterError, match="epsilon"):
        extent(sample, epsilon)
    with pytest.raises(InvalidParameterError, match="epsilon"):
        epsilon_sensitivity(sample, epsilon)


def test_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        VectorSet(name="targets", vectors=UNIT_X, space_id="extra")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        DistanceSample(label="d", values=np.array([0.1]), confidence=0.9)  # type: ignore[call-arg]
