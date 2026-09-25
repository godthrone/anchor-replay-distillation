"""ARD acceptance ruler: q95 / Extent(ε) / ε band / paired CI / noise band.

Responsibility: turn one anchor set A and one target set T (both embedded in the
same space as L2-normalised vectors, distance = ``1 - cos``) into the ARD v4
acceptance readouts listed below.  Pure computation: numpy + pydantic only — no
file, network, model, endpoint or other ``ard`` module is touched.

The 口径 is fixed by the project and is not re-derived here.  With
``d(x) = min_{a in A} 1 - cos(x, a)`` over every target point ``x in T``:

* ``q95``  = ``percentile(d, 95)``, Hyndman–Fan **type-7** (numpy ``linear``).
  This is the **primary acceptance ruler**.  ``q50`` / ``q90`` / ``r_max`` (with
  ``|T|``) come from the same sample.
* ``Extent(ε) = mean(d <= ε)`` — **reference only**, never the ruler.
* **ε sensitivity band**: ``Extent`` at ``ε * {0.95, 1.00, 1.05}``.
* **paired bootstrap**: resampling unit = **target point**, both arms resampled
  with the *same* index draw (paired), ``B = 2000``; reports the point-estimate
  difference of ``q95`` and its 95% CI.
* **noise band**: the distance distribution between repeated generations of the
  same cell; its lower edge is ``q50`` and its upper edge is ``max``.  It answers
  "does this arm's ``q95`` stand out from repeat-generation noise".

Every public function validates its boundary and raises a
:class:`CoverageError` subclass rather than returning a meaningless number.
"""

from typing import Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

#: Hyndman–Fan type-7 quantile, the project-wide 分位 definition.
PERCENTILE_METHOD: Final[Literal["linear"]] = "linear"
#: Quantile levels reported for the nearest-anchor distance sample.
QUANTILE_LEVELS: tuple[float, float, float] = (50.0, 90.0, 95.0)
#: ε sensitivity band scale factors applied to the nominal ε.
EPSILON_SCALE_FACTORS: tuple[float, float, float] = (0.95, 1.00, 1.05)
#: Default number of paired bootstrap resamples (口径: B = 2000).
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
#: Default bootstrap confidence level.
DEFAULT_CONFIDENCE_LEVEL = 0.95
#: Absolute tolerance on the L2 norm of every input vector.
UNIT_NORM_TOLERANCE = 1e-6
#: Absolute tolerance used when a computed distance lands just outside [0, 2].
DISTANCE_TOLERANCE = 1e-9
#: Largest possible cosine distance (``1 - cos`` with ``cos >= -1``).
DISTANCE_UPPER_BOUND = 2.0


class CoverageError(Exception):
    """Base class for acceptance-ruler domain errors.

    It deliberately does **not** derive from :class:`ValueError`: pydantic
    converts ``ValueError`` raised inside a validator into a generic
    ``ValidationError``, which would erase the specific error type a caller
    needs to tell an empty set from an unnormalised one.
    """


class VectorShapeError(CoverageError):
    """Vectors are not a 2-D numeric float matrix with at least one column."""


class EmptyVectorSetError(CoverageError):
    """A vector set has zero rows (a distance to "nothing" is not measurable)."""


class NonFiniteValuesError(CoverageError):
    """NaN or infinity found in vectors or distances."""


class UnnormalizedVectorsError(CoverageError):
    """A vector's L2 norm is not 1, so ``dot`` is not ``cos``."""


class DimensionMismatchError(CoverageError):
    """Target and anchor vectors have different embedding dimensions."""


class EmptyDistanceSampleError(CoverageError):
    """A distance sample has zero entries."""


class InvalidDistancesError(CoverageError):
    """A distance falls outside ``[0, 2]`` beyond floating tolerance."""


class PairingMismatchError(CoverageError):
    """Two samples cannot be paired point-by-point for the bootstrap."""


class InvalidParameterError(CoverageError):
    """A statistic parameter (ε, B, confidence level, ...) is out of range."""


class VectorSet(BaseModel):
    """A named set of L2-normalised vectors, all of one embedding dimension.

    Constructing the model validates the vectors: 2-D numeric array, at least
    one row and one column, every entry finite, every row of L2 norm 1.  A
    violation raises the matching :class:`CoverageError` subclass — never a
    silent pass.

    Attributes:
        name: identity of the row set (e.g. ``"targets"`` or ``"anchors"``).
        vectors: ``(n, dim)`` float64 matrix, each row L2-normalised.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    name: str
    vectors: np.ndarray

    @field_validator("vectors", mode="before")
    @classmethod
    def _coerce_vectors(cls, value: object) -> np.ndarray:
        """Accept any array-like of numbers and freeze it as a float64 matrix."""
        try:
            return np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise VectorShapeError(f"vectors must be a numeric array: {exc}") from exc

    @model_validator(mode="after")
    def _validate_vectors(self) -> "VectorSet":
        validate_vector_set(self)
        return self


class DistanceSample(BaseModel):
    """A labelled 1-D sample of finite cosine distances inside ``[0, 2]``.

    Used for both the per-target nearest-anchor distances and the same-cell
    repeat-generation pair distances.  ``unit_id`` records which unit axis the
    entries are indexed by (the target set for nearest-anchor distances, the
    cell for repeat-generation pairs); the paired bootstrap refuses to pair two
    samples whose ``unit_id`` disagree.

    Attributes:
        label: human-readable identity of the sample.
        values: ``(n,)`` float64 distances.
        unit_id: identity of the axis the entries index, or ``None``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    label: str
    values: np.ndarray
    unit_id: str | None = None

    @field_validator("values", mode="before")
    @classmethod
    def _coerce_values(cls, value: object) -> np.ndarray:
        try:
            return np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise VectorShapeError(f"distances must be a numeric array: {exc}") from exc

    @model_validator(mode="after")
    def _validate_values(self) -> "DistanceSample":
        values = self.values
        if values.ndim != 1:
            raise VectorShapeError(f"distance sample must be 1-D, got shape {values.shape}")
        if values.shape[0] == 0:
            raise EmptyDistanceSampleError(f"distance sample {self.label!r} is empty")
        if not np.all(np.isfinite(values)):
            raise NonFiniteValuesError(f"distance sample {self.label!r} contains NaN or infinity")
        if np.any(values < -DISTANCE_TOLERANCE) or np.any(
            values > DISTANCE_UPPER_BOUND + DISTANCE_TOLERANCE
        ):
            raise InvalidDistancesError(
                f"distance sample {self.label!r} leaves [0, 2]: "
                f"min={float(values.min()):.6f}, max={float(values.max()):.6f}"
            )
        return self


class DistanceQuantiles(BaseModel):
    """Quantiles of one distance sample — the primary acceptance readouts.

    ``n`` is ``|T|`` (the number of target points the sample was measured on).

    Attributes:
        q50: type-7 median distance.
        q90: type-7 90th percentile.
        q95: type-7 95th percentile — the primary acceptance ruler.
        r_max: maximum distance (the worst-covered target point).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    q50: float
    q90: float
    q95: float
    r_max: float


class EpsilonSensitivity(BaseModel):
    """Reference-only ``Extent`` at ``epsilon * {0.95, 1.00, 1.05}``.

    Attributes:
        epsilon: the nominal coverage radius.
        extent_at_095: ``Extent(0.95 * epsilon)``.
        extent_at_100: ``Extent(1.00 * epsilon)`` — the reported coverage.
        extent_at_105: ``Extent(1.05 * epsilon)``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    epsilon: float
    extent_at_095: float
    extent_at_100: float
    extent_at_105: float


class CoverageReadout(BaseModel):
    """Composite acceptance readout of one anchor set against one target set.

    Attributes:
        label: identity of the target/anchor pair being measured.
        n_anchor: ``|A|``, the number of anchor points doing the covering.
        quantiles: ``q50`` / ``q90`` / ``q95`` / ``r_max`` with ``n = |T|``.
        extent: reference-only ``Extent(epsilon)``.
        epsilon_band: the ε sensitivity band around ``epsilon``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    n_anchor: int
    quantiles: DistanceQuantiles
    extent: float
    epsilon_band: EpsilonSensitivity

    @property
    def n_target(self) -> int:
        """``|T|``, the number of target points the readout was measured on."""
        return self.quantiles.n


class PairedBootstrap(BaseModel):
    """Paired target-point bootstrap of the ``q95`` difference between two arms.

    ``point_estimate = q95(arm_a) - q95(arm_b)`` on the full paired sample.  The
    CI is the type-7 percentile interval of the bootstrap differences at
    ``[100 * (1 - level) / 2, 100 * (1 + level) / 2]``.  Both arms are resampled
    with the *same* target-point draw, so the paired structure is preserved.

    Attributes:
        arm_a: label of the minuend arm.
        arm_b: label of the subtrahend arm.
        n_target: number of paired target points (``|T|``).
        n_resamples: number of bootstrap resamples ``B``.
        confidence_level: nominal coverage of the interval.
        seed: seed of the resampling generator.
        point_estimate: ``q95(a) - q95(b)`` on the observed sample.
        ci_low: lower CI edge.
        ci_high: upper CI edge.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm_a: str
    arm_b: str
    n_target: int
    n_resamples: int
    confidence_level: float
    seed: int
    point_estimate: float
    ci_low: float
    ci_high: float


class NoiseBand(BaseModel):
    """Same-cell repeat-generation distance band: ``[q50, max]``.

    Attributes:
        label: identity of the cell (or cell pool) the repeats came from.
        n_pairs: number of repeat-generation pairs measured.
        lower: ``q50`` of the pair distances — the lower noise edge.
        upper: ``max`` of the pair distances — the upper noise edge.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    n_pairs: int
    lower: float
    upper: float


def validate_vector_set(vector_set: VectorSet) -> None:
    """Validate a :class:`VectorSet` boundary; raise instead of returning.

    Construction already runs this check, so calling it again is only needed
    after an unchecked construction path such as ``model_construct``.

    Raises:
        VectorShapeError: not a 2-D matrix, or zero columns.
        EmptyVectorSetError: zero rows.
        NonFiniteValuesError: NaN or infinity present.
        UnnormalizedVectorsError: some row's L2 norm differs from 1 by more
            than :data:`UNIT_NORM_TOLERANCE`.
    """
    vectors = vector_set.vectors
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise VectorShapeError(
            f"vector set {vector_set.name!r} must be a (n, dim) matrix, got shape {vectors.shape}"
        )
    if vectors.shape[0] == 0:
        raise EmptyVectorSetError(
            f"vector set {vector_set.name!r} has no rows; a nearest-anchor distance "
            "to an empty anchor set is undefined"
        )
    if not np.all(np.isfinite(vectors)):
        raise NonFiniteValuesError(f"vector set {vector_set.name!r} contains NaN or infinity")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=UNIT_NORM_TOLERANCE):
        worst = int(np.argmax(np.abs(norms - 1.0)))
        raise UnnormalizedVectorsError(
            f"vector set {vector_set.name!r}: row {worst} has L2 norm {float(norms[worst]):.6f}; "
            f"rows must be L2-normalised (tolerance {UNIT_NORM_TOLERANCE:g}) so that "
            "dot == cos and distance == 1 - cos"
        )


def _type7(values: np.ndarray, quantile: float) -> float:
    """Hyndman–Fan type-7 quantile of *values* (numpy ``method="linear"``)."""
    return float(np.percentile(values, quantile, method=PERCENTILE_METHOD))


def _require_epsilon(epsilon: float) -> None:
    """Reject a non-finite or negative coverage radius."""
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise InvalidParameterError(f"epsilon must be finite and >= 0, got {epsilon!r}")


def nearest_anchor_distances(
    targets: VectorSet,
    anchors: VectorSet,
    *,
    label: str | None = None,
) -> DistanceSample:
    """Return ``d(x) = min_{a in A} (1 - cos(x, a))`` for every ``x in T``.

    Both sets must be L2-normalised, so the dot product *is* the cosine and the
    distance is ``1 - dot``.  The result is clamped into ``[0, 2]``: a dot of
    ``1 + 1e-16`` would otherwise yield ``-2e-16``, which is floating noise, not
    a negative distance.

    Args:
        targets: the target set ``T``; one output distance per row.
        anchors: the anchor set ``A`` covering ``T``.
        label: sample label; defaults to ``targets.name``.

    Returns:
        A :class:`DistanceSample` of length ``|T|`` whose ``unit_id`` is
        ``targets.name``.

    Raises:
        EmptyVectorSetError: either set has zero rows.
        VectorShapeError: either set is not a 2-D matrix.
        NonFiniteValuesError: either set contains NaN or infinity.
        UnnormalizedVectorsError: either set has a row of L2 norm != 1.
        DimensionMismatchError: the two sets have different embedding dimensions.
    """
    validate_vector_set(targets)
    validate_vector_set(anchors)
    if targets.vectors.shape[1] != anchors.vectors.shape[1]:
        raise DimensionMismatchError(
            f"targets {targets.name!r} have dimension {targets.vectors.shape[1]} but anchors "
            f"{anchors.name!r} have dimension {anchors.vectors.shape[1]}; "
            "a cross-dimension cosine distance has no meaning"
        )
    distances = 1.0 - targets.vectors @ anchors.vectors.T
    nearest = np.clip(distances.min(axis=1), 0.0, DISTANCE_UPPER_BOUND)
    return DistanceSample(label=label or targets.name, values=nearest, unit_id=targets.name)


def distance_quantiles(sample: DistanceSample) -> DistanceQuantiles:
    """Summarise a distance sample as ``q50`` / ``q90`` / ``q95`` / ``r_max`` / ``n``.

    Args:
        sample: per-target nearest-anchor distances.

    Returns:
        Type-7 quantiles plus the maximum; ``n`` is ``|T|``.
    """
    values = sample.values
    return DistanceQuantiles(
        n=int(values.shape[0]),
        q50=_type7(values, QUANTILE_LEVELS[0]),
        q90=_type7(values, QUANTILE_LEVELS[1]),
        q95=_type7(values, QUANTILE_LEVELS[2]),
        r_max=float(values.max()),
    )


def extent(sample: DistanceSample, epsilon: float) -> float:
    """Return ``Extent(ε) = mean(d <= ε)`` — a reference-only coverage number.

    Args:
        sample: per-target nearest-anchor distances.
        epsilon: coverage radius, finite and ``>= 0``.

    Returns:
        The fraction of target points within cosine distance ``epsilon``.

    Raises:
        InvalidParameterError: ``epsilon`` is negative or not finite.
    """
    _require_epsilon(epsilon)
    return float(np.mean(sample.values <= epsilon))


def epsilon_sensitivity(sample: DistanceSample, epsilon: float) -> EpsilonSensitivity:
    """Return ``Extent`` at ``epsilon * {0.95, 1.00, 1.05}`` — the ε sensitivity band.

    Args:
        sample: per-target nearest-anchor distances.
        epsilon: nominal coverage radius, finite and ``>= 0``.

    Returns:
        The three extents, with ``extent_at_100`` equal to :func:`extent`.

    Raises:
        InvalidParameterError: ``epsilon`` is negative or not finite.
    """
    _require_epsilon(epsilon)
    return EpsilonSensitivity(
        epsilon=epsilon,
        extent_at_095=extent(sample, epsilon * EPSILON_SCALE_FACTORS[0]),
        extent_at_100=extent(sample, epsilon * EPSILON_SCALE_FACTORS[1]),
        extent_at_105=extent(sample, epsilon * EPSILON_SCALE_FACTORS[2]),
    )


def acceptance_readout(
    targets: VectorSet,
    anchors: VectorSet,
    *,
    epsilon: float,
    label: str | None = None,
) -> CoverageReadout:
    """Measure one anchor set against one target set — the full readout.

    Args:
        targets: the target set ``T``.
        anchors: the anchor set ``A``.
        epsilon: nominal coverage radius for the reference-only ``Extent``.
        label: readout label; defaults to ``targets.name``.

    Returns:
        ``q50`` / ``q90`` / ``q95`` / ``r_max`` with ``|T|``, ``Extent(ε)`` and
        the ε sensitivity band.

    Raises:
        CoverageError: for any invalid vector boundary, dimension mismatch or
            invalid ``epsilon``.
    """
    sample = nearest_anchor_distances(targets, anchors, label=label)
    return CoverageReadout(
        label=sample.label,
        n_anchor=int(anchors.vectors.shape[0]),
        quantiles=distance_quantiles(sample),
        extent=extent(sample, epsilon),
        epsilon_band=epsilon_sensitivity(sample, epsilon),
    )


def paired_bootstrap_q95_ci(
    arm_a: DistanceSample,
    arm_b: DistanceSample,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = 0,
) -> PairedBootstrap:
    """Paired target-point bootstrap CI for ``q95(arm_a) - q95(arm_b)``.

    The resampling unit is the **target point**: entry ``i`` of both samples must
    describe the same target point, so one index draw is applied to both arms
    inside every resample (that is what makes the bootstrap *paired*).
    ``point_estimate`` is the difference of type-7 ``q95`` on the observed
    sample; the CI is the type-7 percentile interval of the ``n_resamples``
    paired bootstrap differences.

    Args:
        arm_a: per-target nearest-anchor distances of the minuend arm.
        arm_b: per-target nearest-anchor distances of the subtrahend arm.
        n_resamples: bootstrap resamples ``B`` (口径 default 2000), ``>= 1``.
        confidence_level: interval coverage in ``(0, 1)``.
        seed: seed of ``numpy.random.default_rng``; same seed and same numpy
            version reproduce the same CI.

    Returns:
        Point-estimate difference with its CI and provenance.

    Raises:
        PairingMismatchError: sample lengths differ, or two non-``None``
            ``unit_id`` values disagree.
        InvalidParameterError: ``n_resamples < 1`` or ``confidence_level``
            outside ``(0, 1)``.
    """
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples < 1:
        raise InvalidParameterError(f"n_resamples must be an integer >= 1, got {n_resamples!r}")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise InvalidParameterError(
            f"confidence_level must lie strictly inside (0, 1), got {confidence_level!r}"
        )
    n = int(arm_a.values.shape[0])
    if int(arm_b.values.shape[0]) != n:
        raise PairingMismatchError(
            f"paired bootstrap needs one entry per shared target point: {arm_a.label!r} has "
            f"{n} and {arm_b.label!r} has {int(arm_b.values.shape[0])}"
        )
    if arm_a.unit_id is not None and arm_b.unit_id is not None and arm_a.unit_id != arm_b.unit_id:
        raise PairingMismatchError(
            f"paired bootstrap needs both arms measured on the same target set: "
            f"{arm_a.unit_id!r} != {arm_b.unit_id!r}"
        )

    point_estimate = _type7(arm_a.values, QUANTILE_LEVELS[2]) - _type7(
        arm_b.values, QUANTILE_LEVELS[2]
    )
    rng = np.random.default_rng(seed)
    positions = rng.integers(0, n, size=(n_resamples, n))
    boot_a = np.percentile(
        arm_a.values[positions], QUANTILE_LEVELS[2], axis=1, method=PERCENTILE_METHOD
    )
    boot_b = np.percentile(
        arm_b.values[positions], QUANTILE_LEVELS[2], axis=1, method=PERCENTILE_METHOD
    )
    differences = boot_a - boot_b
    tail = (1.0 - confidence_level) / 2.0
    return PairedBootstrap(
        arm_a=arm_a.label,
        arm_b=arm_b.label,
        n_target=n,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        point_estimate=float(point_estimate),
        ci_low=_type7(differences, 100.0 * tail),
        ci_high=_type7(differences, 100.0 * (1.0 - tail)),
    )


def pairwise_distances(vectors: VectorSet) -> DistanceSample:
    """Pairwise cosine distances between the **distinct** vectors of one set.

    This is the raw material of the noise band: feed it the repeated generations
    of one cell and every unordered pair of distinct repeats becomes one entry.

    Args:
        vectors: repeated generations to compare with each other (``n >= 2``).

    Returns:
        A :class:`DistanceSample` of ``n * (n - 1) / 2`` pair distances, with
        ``unit_id = vectors.name``.

    Raises:
        EmptyVectorSetError: fewer than two vectors, so no pair exists.
        CoverageError: for any invalid vector boundary.
    """
    validate_vector_set(vectors)
    n = int(vectors.vectors.shape[0])
    if n < 2:
        raise EmptyVectorSetError(
            f"vector set {vectors.name!r} needs at least two rows to form a repeat pair, got {n}"
        )
    distances = 1.0 - vectors.vectors @ vectors.vectors.T
    upper = np.triu_indices(n, k=1)
    pairs = np.clip(distances[upper], 0.0, DISTANCE_UPPER_BOUND)
    return DistanceSample(label=f"{vectors.name}:pairs", values=pairs, unit_id=vectors.name)


def noise_band(pair_distances: DistanceSample) -> NoiseBand:
    """Same-cell repeat-generation noise band ``[q50, max]``.

    Args:
        pair_distances: pair distances from :func:`pairwise_distances` (or any
            sample of equal-cell repeat distances).

    Returns:
        ``lower = q50`` and ``upper = max`` of the pair distances, to be used
        with :func:`within_noise_band`.
    """
    return NoiseBand(
        label=pair_distances.label,
        n_pairs=int(pair_distances.values.shape[0]),
        lower=_type7(pair_distances.values, QUANTILE_LEVELS[0]),
        upper=float(pair_distances.values.max()),
    )


def within_noise_band(value: float, band: NoiseBand) -> bool:
    """True when *value* lies inside the noise band, i.e. within repeat noise.

    A readout inside ``[band.lower, band.upper]`` is not distinguishable from
    same-cell repeat-generation noise; one that falls outside it is.

    Raises:
        NonFiniteValuesError: ``value`` is NaN or infinite.
    """
    if not np.isfinite(value):
        raise NonFiniteValuesError(f"value must be finite, got {value!r}")
    return band.lower <= value <= band.upper
