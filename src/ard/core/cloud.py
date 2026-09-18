"""Space-identified vector clouds and space-safe farthest-point selection.

Every embedding comes from some embedder, and every matrix row means something
only inside the matrix it came from.  This module makes that explicit with two
identities carried by :class:`CloudVectors`:

* ``space_id`` — the fingerprint of the embedding space (which embedder
  produced the vectors).  Vector arithmetic — distances, coverage — is only
  meaningful between clouds that share it.
* ``cloud_id`` — the identity of the row set.  Row *positions* are meaningful
  only inside the cloud that produced them.  Two clouds embedded by the same
  model (say, a label-name cloud and a generated-text cloud) share a
  ``space_id`` and still index different rows.

That second identity is the one a project conclusion once silently violated: a
selection computed on cloud A was used as row positions into cloud B, yielding
a number that looked fine and meant nothing.  The guard is structural: row
positions never leave this module unlabelled.  :func:`fps` returns a
:class:`CloudIndex` (positions tagged with the cloud that produced them) plus
the selected vectors, and :meth:`CloudVectors.select` raises
:class:`CloudMismatchError` / :class:`SpaceMismatchError` when a selection is
applied to any other cloud.

Responsibility: own the space/cloud identity contract and the FPS selection
built on it.  No I/O, no configuration, no network.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


class VectorSpaceError(ValueError):
    """Base class for identity violations between vector clouds (§2.1 契约即防呆)."""


class SpaceMismatchError(VectorSpaceError):
    """Operands belong to different embedding spaces (different ``space_id``)."""


class CloudMismatchError(VectorSpaceError):
    """A selection of one cloud was applied to another cloud."""


@dataclass(frozen=True)
class CloudIndex:
    """Row positions of one cloud, tagged with the identity that produced them.

    An index is a *labelled* selection, not a bare ``list[int]``: it remembers
    which cloud it belongs to, so it cannot travel to a different cloud by
    accident (see :meth:`CloudVectors.select`).

    Attributes:
        space_id: embedding space the positions were produced in.
        cloud_id: cloud the positions were produced from.
        positions: distinct, non-negative, non-empty row numbers, in selection
            order (FPS order is meaningful and is preserved).
    """

    space_id: str
    cloud_id: str
    positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.space_id:
            raise ValueError("space_id must be a non-empty string")
        if not self.cloud_id:
            raise ValueError("cloud_id must be a non-empty string")
        try:
            positions = tuple(operator.index(position) for position in self.positions)
        except TypeError as exc:
            raise ValueError(
                f"positions must be integers (row numbers), got {self.positions!r}"
            ) from exc
        if not positions:
            raise ValueError(
                "positions must not be empty: an empty selection cannot be measured"
            )
        if len(set(positions)) != len(positions):
            raise ValueError(f"positions must not repeat, got {positions}")
        if any(position < 0 for position in positions):
            raise ValueError(f"positions must be non-negative, got {positions}")
        object.__setattr__(self, "positions", positions)

    def __len__(self) -> int:
        return len(self.positions)


@dataclass(frozen=True)
class CloudVectors:
    """A vector matrix together with the identity of its space and row set.

    Attributes:
        space_id: fingerprint of the embedding space, e.g. ``"embedding:1024"``
            (see :func:`ard.core.embeddings.embedding_space_id`).
        cloud_id: identity of this row set inside ``space_id``
            (e.g. ``"knowledge_domains"``, ``"ontology_combinations"``).
        vectors: ``(n_items, dimension)`` float64 matrix.  It is **not copied**:
            the production composed matrix is gigabytes, so no defensive copy is
            affordable.  A converted copy is made only when the input is not
            already float64 / C-contiguous — the caller must not mutate the
            matrix after construction.
        item_ids: optional stable identifier per row, parallel to ``vectors``.
            This — never a row position — is what crosses a cloud boundary.
            Duplicates are allowed: ontology labels legitimately repeat.

    Raises:
        ValueError: if an identity string is empty, the matrix is not 2-D, has
            no rows or no dimension, or contains NaN/inf.
    """

    space_id: str
    cloud_id: str
    vectors: np.ndarray
    item_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.space_id:
            raise ValueError("space_id must be a non-empty string")
        if not self.cloud_id:
            raise ValueError("cloud_id must be a non-empty string")
        vectors = np.asarray(self.vectors, dtype=np.float64)
        if vectors.ndim != 2:
            raise ValueError(
                f"vectors must be 2-D (n_items, dimension), got shape {vectors.shape}"
            )
        if vectors.shape[0] == 0:
            raise ValueError("vectors must contain at least one row")
        if vectors.shape[1] == 0:
            raise ValueError("vectors must contain at least one dimension")
        if not bool(np.isfinite(vectors).all()):
            raise ValueError("vectors must be finite: NaN/inf cannot be measured")
        object.__setattr__(self, "vectors", np.ascontiguousarray(vectors))
        if self.item_ids is not None:
            item_ids = tuple(self.item_ids)
            if len(item_ids) != vectors.shape[0]:
                raise ValueError(
                    f"item_ids has {len(item_ids)} entries but vectors has "
                    f"{vectors.shape[0]} rows"
                )
            object.__setattr__(self, "item_ids", item_ids)

    @property
    def n_items(self) -> int:
        """Number of rows (points) in this cloud."""
        return int(self.vectors.shape[0])

    @property
    def dimension(self) -> int:
        """Vector length (number of columns) in this cloud."""
        return int(self.vectors.shape[1])

    def require_item_ids(self) -> tuple[str, ...]:
        """Return this cloud's per-row identifiers, or raise if it has none.

        Used where a selection must be mapped back to its source rows: the
        mapping goes through item identity, never through a parallel array of
        positions.
        """
        if self.item_ids is None:
            raise ValueError(
                f"cloud {self.cloud_id!r} (space {self.space_id!r}) carries no item_ids; "
                "construct it with item_ids to map a selection back to its source rows"
            )
        return self.item_ids

    def index_of(self, positions: Sequence[int]) -> CloudIndex:
        """Label *positions* — row numbers of **this** cloud — as a CloudIndex.

        The sanctioned way to express a selection that does not come from
        :func:`fps` (e.g. a random baseline): the caller names the cloud the
        positions belong to, and the result stays bound to it.

        Raises:
            ValueError: if a position is out of range or not an integer.
        """
        try:
            tagged = tuple(operator.index(position) for position in positions)
        except TypeError as exc:
            raise ValueError(
                f"positions must be integers (row numbers), got {positions!r}"
            ) from exc
        for position in tagged:
            if position < 0 or position >= self.n_items:
                raise ValueError(
                    f"position {position} is out of range for cloud {self.cloud_id!r} "
                    f"({self.n_items} rows)"
                )
        return CloudIndex(space_id=self.space_id, cloud_id=self.cloud_id, positions=tagged)

    def select(self, index: CloudIndex) -> CloudVectors:
        """Return the rows named by *index* as a new cloud with the same identity.

        Raises:
            SpaceMismatchError: if *index* was produced in another embedding space.
            CloudMismatchError: if *index* was produced by another cloud in the
                same space — the exact misuse that otherwise computes silently.
        """
        self.require_owns(index, "select")
        item_ids = (
            None
            if self.item_ids is None
            else tuple(self.item_ids[position] for position in index.positions)
        )
        return CloudVectors(
            space_id=self.space_id,
            cloud_id=self.cloud_id,
            vectors=self.vectors[list(index.positions)],
            item_ids=item_ids,
        )

    def require_owns(self, index: CloudIndex, operation: str) -> None:
        """Raise unless *index* was produced by this cloud.

        Args:
            index: the labelled selection to check.
            operation: name of the operation, used in the error message.

        Raises:
            SpaceMismatchError: different ``space_id``.
            CloudMismatchError: different ``cloud_id``.
        """
        if index.space_id != self.space_id:
            raise SpaceMismatchError(
                f"{operation}: row positions of space {index.space_id!r} cannot address a "
                f"cloud in space {self.space_id!r}. Positions are only valid inside the "
                "embedding space that produced them."
            )
        if index.cloud_id != self.cloud_id:
            raise CloudMismatchError(
                f"{operation}: a selection of cloud {index.cloud_id!r} was applied to cloud "
                f"{self.cloud_id!r}. Both live in space {self.space_id!r}, but row positions "
                "are only valid for the cloud that produced them. Match rows by item_id, or "
                "use coverage.coverage_to() when a cross-cloud measurement is intended."
            )

    def require_same_space(self, other: CloudVectors, operation: str) -> None:
        """Raise unless *other* lives in the same embedding space as ``self``.

        Also rejects two clouds that claim the same ``space_id`` but have
        different dimensions — a self-contradictory identity, not a measurement.

        Raises:
            SpaceMismatchError: on either violation.
        """
        if self.space_id != other.space_id:
            raise SpaceMismatchError(
                f"{operation}: cloud {self.cloud_id!r} is in space {self.space_id!r} but "
                f"cloud {other.cloud_id!r} is in space {other.space_id!r}. Vectors from "
                "different embedding spaces are not comparable; the distance would be a "
                "number without meaning."
            )
        if self.dimension != other.dimension:
            raise SpaceMismatchError(
                f"{operation}: cloud {self.cloud_id!r} and cloud {other.cloud_id!r} both "
                f"declare space {self.space_id!r} but have different dimensions "
                f"({self.dimension} vs {other.dimension}); the space identity is inconsistent."
            )

    def require_same_cloud(self, other: CloudVectors, operation: str) -> None:
        """Raise unless *other* is a selection of the very same cloud as ``self``.

        Raises:
            SpaceMismatchError: different embedding space.
            CloudMismatchError: different cloud in the same space (use
                :func:`ard.core.coverage.coverage_to` for that case).
        """
        self.require_same_space(other, operation)
        if self.cloud_id != other.cloud_id:
            raise CloudMismatchError(
                f"{operation}: cloud {other.cloud_id!r} is not cloud {self.cloud_id!r} "
                f"(both in space {self.space_id!r}). Use coverage.coverage_to() for a "
                "cross-cloud measurement; self-coverage of one cloud by a selection of "
                "another is exactly the silent error this API exists to stop."
            )


#: Greedy criterion «the largest of the blanks» — every further point is the row
#: whose *minimum* cosine distance to the selected set is the largest (the
#: classic farthest-point rule).  This is the historical behaviour.
CRITERION_MAX = "max"

#: Greedy criterion «the total blankness» — every further point is the row whose
#: **sum** of distances to the nearest already-selected row (counting itself,
#: distance 0) is the smallest.  Also known as the greedy facility-location /
#: max-coverage step.  On clustered clouds — clusters plus isolated points, which
#: is what the production anchor geometry looks like — this optimises the
#: quantity the holdout radius ``r_mean`` actually measures, whereas
#: :data:`CRITERION_MAX` optimises a statistic dominated by a single isolated
#: point.
CRITERION_SUM = "sum"

FPS_CRITERIA: tuple[str, ...] = (CRITERION_MAX, CRITERION_SUM)

# Peak memory the ``sum`` criterion may spend on the ``(n_items, n_items)``
# distance matrix it needs.  ``max`` never materialises that matrix; ``sum``
# cannot be evaluated without it, so the cost is bounded here rather than left
# to whatever the caller's pool size happens to be (a 50k-row pool would ask for
# 20 GB).  Exceeding the budget is a *boundary check*: refused loudly, never
# silently degraded to another criterion (§2.3).
FPS_SUM_MATRIX_MAX_BYTES = 1 << 30  # 1 GiB


def _check_criterion(criterion: str) -> str:
    """Return *criterion* if it is a known greedy rule, else raise.

    A typo must never fall back to the default: silently running the wrong
    objective is exactly the failure this API exists to make impossible
    (§2.1 契约即防呆).
    """
    if criterion not in FPS_CRITERIA:
        raise ValueError(
            f"criterion must be one of {list(FPS_CRITERIA)}, got {criterion!r}"
        )
    return criterion


def _sum_matrix_limit() -> int:
    """Largest ``n_items`` whose ``(n, n)`` float64 distance matrix fits the budget."""
    return int(np.sqrt(FPS_SUM_MATRIX_MAX_BYTES // 8))


def fps(
    cloud: CloudVectors,
    n: int,
    *,
    seed: int | None = None,
    criterion: str = CRITERION_MAX,
) -> tuple[CloudIndex, CloudVectors]:
    """Select *n* diverse rows of *cloud* by greedy farthest-point sampling.

    Rows are unit-normalised and the first point is drawn with
    ``numpy.random.default_rng(seed if seed is not None else 42)``; the contract
    is that the caller receives a :class:`CloudIndex` carrying ``cloud``'s
    identity plus the selected vectors — never a bare ``list[int]`` that could be
    used to index a different cloud.

    Every further point is chosen by *criterion*:

    * :data:`CRITERION_MAX` (default, historical) — ``argmax_i min_j d(i, j)``:
      the row sitting furthest from the selected set.  This is what this
      function has always done; the code path is untouched.
    * :data:`CRITERION_SUM` — ``argmin_i Σ_j min_j d(i, j)``: the row that
      leaves the least *total* blankness, i.e. the greedy facility-location
      step.  Distances are cosine on L2-normalised rows, so "distance to the
      nearest selected row" includes the candidate itself at distance 0.

    Args:
        cloud: cloud to select from.
        n: number of rows to select; must satisfy ``1 <= n <= cloud.n_items``.
        seed: random seed for the initial point (``None`` → ``42``, as before).
        criterion: greedy rule, one of :data:`FPS_CRITERIA`.  The default
            reproduces the historical output **bit for bit**.

    Returns:
        ``(index, selection)`` — the labelled positions and the selected
        vectors, both carrying ``cloud``'s ``space_id`` and ``cloud_id``.

    Raises:
        ValueError: if ``n`` is outside ``[1, cloud.n_items]``, if *criterion*
            is not a known rule, or if ``CRITERION_SUM`` is asked of a cloud so
            large that its distance matrix exceeds
            :data:`FPS_SUM_MATRIX_MAX_BYTES`.
    """
    criterion = _check_criterion(criterion)
    vectors = cloud.vectors
    n_items = cloud.n_items
    if n < 1 or n > n_items:
        raise ValueError(f"n must be in [1, {n_items}], got {n}")

    rng = np.random.default_rng(seed if seed is not None else 42)
    selected: list[int] = [int(rng.integers(0, n_items))]
    if n > 1:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Guard against zero vectors — replace with a tiny value so division
        # succeeds and the zero vector ends up at the origin.
        norms = np.where(norms == 0.0, 1e-12, norms)
        unit = vectors / norms

        # min_dist[i] = min cosine distance from row i to any selected row.
        min_dist = 1.0 - unit @ unit[selected[0]]

        # The ``sum`` rule needs every pairwise distance, while the ``max`` rule
        # only needs the running minima.  Materialising the matrix once is what
        # makes the k greedy steps affordable; it is done only when asked for,
        # so the default path allocates nothing new.
        pair_distances: np.ndarray | None = None
        if criterion == CRITERION_SUM:
            limit = _sum_matrix_limit()
            if n_items > limit:
                raise ValueError(
                    f"criterion={CRITERION_SUM!r} needs the full ({n_items}, {n_items}) "
                    f"distance matrix ({n_items * n_items * 8 / 2**30:.1f} GiB), which "
                    f"exceeds FPS_SUM_MATRIX_MAX_BYTES={FPS_SUM_MATRIX_MAX_BYTES} "
                    f"(limits the cloud to {limit} rows). Refusing rather than silently "
                    "falling back to a different criterion."
                )
            # ``np.clip`` because ``1 - <u, u>`` is 0 only up to rounding: a
            # cosine *distance* is never negative, and the rest of the project
            # measures coverage with the same clamp (``coverage_to``).  This also
            # makes ``d(i, i)`` exactly 0, so a candidate's own row contributes
            # exactly the "already covered" term the objective means.
            pair_distances = np.clip(1.0 - unit @ unit.T, 0.0, None)

        for _ in range(1, n):
            mask = np.ones(n_items, dtype=bool)
            mask[selected] = False
            if criterion == CRITERION_MAX:
                masked_dist = np.where(mask, min_dist, -np.inf)
                best = int(np.argmax(masked_dist))
            else:
                assert pair_distances is not None  # set above for CRITERION_SUM
                # scores[i] = Σ_j min(min_dist[j], d(i, j)) — the distance from
                # every row to the nearest member of ``selected ∪ {i}``, summed.
                # Selected rows contribute their own 0, so they need no masking.
                # Row-blocked so the transient is a fixed 64 MiB instead of a
                # second ``(n, n)`` matrix.
                scores = np.empty(n_items, dtype=np.float64)
                nearest = np.clip(min_dist, 0.0, None)
                block = max(1, (64 << 20) // (n_items * 8))
                for start in range(0, n_items, block):
                    stop = min(start + block, n_items)
                    scores[start:stop] = np.minimum(
                        pair_distances[start:stop], nearest[None, :]
                    ).sum(axis=1)
                masked_scores = np.where(mask, scores, np.inf)
                best = int(np.argmin(masked_scores))
            selected.append(best)
            min_dist = np.minimum(min_dist, 1.0 - unit @ unit[best])

    index = cloud.index_of(selected)
    return index, cloud.select(index)
