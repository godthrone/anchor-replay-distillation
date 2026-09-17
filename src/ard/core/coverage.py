"""Space-safe coverage measurement over :class:`~ard.core.cloud.CloudVectors`.

Coverage asks "how far is the most remote row of a target cloud from the
nearest selected row": the statistic is the minimum, over the selected rows, of
``1 - dot`` (cosine distance when the rows are unit-normalised), summarised by
max / mean / median / p90 / p95.  Those five numbers are unchanged from the
historical analysis helper; what is new is that both operands carry their
identity, so a measurement can no longer silently mix the row positions of one
cloud with the matrix of another.

Two functions, named for their scope:

* :func:`coverage_to` — measure a *target* cloud against a, possibly different,
  *selected* cloud (e.g. label vectors covering a generated-text cloud).  Only
  the vectors are used, so differing row sets are exactly the intended case;
  both clouds must still share one embedding space.
* :func:`self_coverage` — one cloud covering itself; a selection that came from
  a different cloud is rejected.

Responsibility: compute coverage statistics from clouds.  No selection, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ard.core.cloud import CloudVectors


@dataclass(frozen=True)
class CoverageStats:
    """Coverage statistics together with the identities they were computed from.

    The five distances are the historical definitions, unchanged:
    ``max`` / ``mean`` / ``median`` / ``p90`` / ``p95`` over the per-target-row
    minimum distance to the selected vectors.  The remaining fields make the
    measurement auditable (§1.4 单一真相源): one look answers "which cloud
    covered which cloud, in which space".

    Attributes:
        same_cloud: True when the selection came from the target cloud itself.
    """

    max: float
    mean: float
    median: float
    p90: float
    p95: float
    n_target: int
    n_selected: int
    target_space_id: str
    target_cloud_id: str
    selected_space_id: str
    selected_cloud_id: str

    @property
    def same_cloud(self) -> bool:
        """True when the selection was taken from the target cloud itself."""
        return self.target_cloud_id == self.selected_cloud_id

    def as_dict(self) -> dict[str, float]:
        """The five statistics under the historical ``coverage()`` keys."""
        return {
            "max": self.max,
            "mean": self.mean,
            "median": self.median,
            "p90": self.p90,
            "p95": self.p95,
        }


def _stats(distances: np.ndarray, target: CloudVectors, selected: CloudVectors) -> CoverageStats:
    """Summarise a ``(n_target, n_selected)`` distance matrix.

    Args:
        distances: pairwise ``1 - dot`` distances.
        target: the cloud being covered.
        selected: the cloud doing the covering.

    Returns:
        The five statistics plus both identities.
    """
    nearest = distances.min(axis=1)
    return CoverageStats(
        max=float(nearest.max()),
        mean=float(nearest.mean()),
        median=float(np.median(nearest)),
        p90=float(np.percentile(nearest, 90)),
        p95=float(np.percentile(nearest, 95)),
        n_target=target.n_items,
        n_selected=selected.n_items,
        target_space_id=target.space_id,
        target_cloud_id=target.cloud_id,
        selected_space_id=selected.space_id,
        selected_cloud_id=selected.cloud_id,
    )


def coverage_to(target: CloudVectors, selected: CloudVectors) -> CoverageStats:
    """Measure how well *selected* covers *target* — both clouds explicit.

    Only the *vectors* of ``selected`` are used against ``target``; the
    selection's row positions are never applied to the target.  That is why the
    two clouds may be different row sets (the intended cross-cloud measurement),
    and why this function cannot reproduce the historical index-space error.

    Args:
        target: cloud whose rows are being covered.
        selected: cloud whose rows do the covering.

    Returns:
        Coverage statistics with both identities recorded.

    Raises:
        SpaceMismatchError: if the clouds are not in one embedding space — a
            distance between vectors of different spaces is a number without
            meaning, and producing it silently is the failure mode this module
            exists to stop.
    """
    target.require_same_space(selected, "coverage_to")
    distances = 1.0 - target.vectors @ selected.vectors.T
    return _stats(distances, target, selected)


def self_coverage(cloud: CloudVectors, selection: CloudVectors) -> CoverageStats:
    """Measure *cloud* against a *selection* taken from that same cloud.

    Equivalent to the historical ``coverage(U_all, sel)``: ``selection`` is a
    subset of ``cloud`` (typically the output of :func:`ard.core.cloud.fps`), so
    each target row is measured against the selected rows of its own cloud.

    Args:
        cloud: the cloud being covered.
        selection: a selection of ``cloud`` itself, carrying the same identity.

    Returns:
        Coverage statistics with both identities recorded (``same_cloud`` True).

    Raises:
        CloudMismatchError: if *selection* came from another cloud, even in the
            same embedding space.  Use :func:`coverage_to` when a cross-cloud
            measurement is intended — this function means *self*-coverage.
        SpaceMismatchError: if the identities disagree on the embedding space.
    """
    cloud.require_same_cloud(selection, "self_coverage")
    return coverage_to(cloud, selection)
