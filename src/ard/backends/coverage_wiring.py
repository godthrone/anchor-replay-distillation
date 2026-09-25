"""Coverage acceptance wiring: bank records + target set → embedded readout.

Responsibility: own the **facility** half of the acceptance phase — read the
target-set file, validate it against the configured embedding boundary, call
:class:`ard.backends.embedding_client.EmbeddingClient` to embed the anchor and
target texts, and hand the resulting vector sets to the pure assembly in
:mod:`ard.core.acceptance`.  Nothing here computes a statistic and nothing here
reads a config file: the caller passes the resolved
:class:`~ard.config.CoverageEmbedding` value, so all configuration stays one
layer up (§1.1 计算与设施分离).

Every boundary is checked loudly and nothing is swallowed:

* a missing / unreadable / unparsable / empty target-set file is refused with
  the path, and a declared ``expected_count`` / ``dimension`` that disagrees with
  the file or the config names both the expected and the actual value;
* an embedding failure propagates with the exception type the client already
  defines (:class:`~ard.backends.embedding_client.EmbeddingRequestError`,
  ``EmbeddingDimensionMismatchError``, …) — no retry policy is re-implemented
  and no error is converted into a missing number;
* the API key never reaches a message, a report or a log (§15).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from ard.backends.embedding_client import EmbeddingClient
from ard.config import CoverageEmbedding
from ard.core import acceptance
from ard.core import coverage as ruler
from ard.core.types import JsonObject, JsonObjectSequence


class CoverageWiringError(Exception):
    """The acceptance metric readout cannot be wired to its inputs.

    Raised for a target set that is missing, unreadable, unparsable, empty,
    whose entries carry no usable text, or whose declared count / dimension /
    epsilon contradicts the file or the configuration.  Deliberately not a
    ``ValueError`` subclass: the caller distinguishes "the user's target set is
    unusable" from "a measured value failed validation" (``CoverageError``).
    """


@dataclass(frozen=True, slots=True)
class TargetSet:
    """A validated target set: the texts to embed plus their declared metadata.

    Attributes:
        path: the file the entries were read from (echoed in every report).
        texts: the embedded text of every entry, in file order.
        declared_count: the header's ``expected_count``, or ``None``.
        declared_dimension: the header's ``dimension``, or ``None``.
        declared_epsilon: the header's ``epsilon``, or ``None`` — when absent,
            the target set's own intrinsic scale calibrates epsilon instead.
    """

    path: str
    texts: tuple[str, ...]
    declared_count: int | None
    declared_dimension: int | None
    declared_epsilon: float | None


def load_target_set(path: str | Path, *, expected_dimension: int | None) -> TargetSet:
    """Read and validate a target set file.

    Accepted shapes — JSON array, JSON object with a ``"targets"`` array, or
    JSONL (one document per line).  Every entry is either a non-empty string or
    an object with a non-empty ``"text"`` field.  An object document may carry a
    header with ``expected_count`` (must equal the number of entries),
    ``dimension`` (must equal the configured embedding dimension) and
    ``epsilon`` (a finite, non-negative radius).

    Args:
        path: Target-set file path.
        expected_dimension: The configured embedding dimension, or ``None``
            when the config does not pin one.

    Returns:
        The validated target set.

    Raises:
        CoverageWiringError: For every boundary listed in the module docstring.
            The message always names the file and, where relevant, the expected
            and the actual value.
    """
    target_path = Path(path)
    if not target_path.is_file():
        raise CoverageWiringError(
            f"target set file not found: {target_path} "
            "(coverage.target_set_path; the file is read before any run side effect)"
        )
    try:
        raw = target_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CoverageWiringError(f"target set file {target_path} is unreadable: {exc}") from exc

    entries, header = _parse_document(raw, target_path)
    texts = [_entry_text(entry, index, target_path) for index, entry in enumerate(entries)]

    declared_count = _declared_int(header, "expected_count", target_path)
    if declared_count is not None and declared_count != len(texts):
        raise CoverageWiringError(
            f"target set {target_path} declares expected_count={declared_count} but carries "
            f"{len(texts)} entr{'y' if len(texts) == 1 else 'ies'}"
        )
    declared_dimension = _declared_int(header, "dimension", target_path)
    if (
        declared_dimension is not None
        and expected_dimension is not None
        and declared_dimension != expected_dimension
    ):
        raise CoverageWiringError(
            f"target set {target_path} declares dimension={declared_dimension} but "
            f"coverage.embedding.dimension is {expected_dimension}"
        )
    declared_epsilon = _declared_epsilon(header, target_path)
    return TargetSet(
        path=str(target_path),
        texts=tuple(texts),
        declared_count=declared_count,
        declared_dimension=declared_dimension,
        declared_epsilon=declared_epsilon,
    )


def build_metric_readout(
    *,
    embedding: CoverageEmbedding,
    records: JsonObjectSequence,
    anchors_source: str,
    target_set: TargetSet,
) -> acceptance.MetricReadout:
    """Embed the run's anchors and the target set and assemble the metric readout.

    Args:
        embedding: The resolved ``[coverage.embedding]`` settings.
        records: The run's bank records, in file order.
        anchors_source: Where the anchor texts came from (the bank path).
        target_set: A target set already validated by :func:`load_target_set`.

    Returns:
        The metric readout, with its space declared (fields, ``|A|``, ``|T|``,
        embedder model + dimension) and its noise section stating whether a
        repeat-generation band was measurable.

    Raises:
        acceptance.AcceptanceError: If a record cannot yield an anchor text or
            carries no coordinate mapping.
        CoverageWiringError: If the client returns unnormalised rows.
        EmbeddingError: Propagated unchanged from the embedding client.
    """
    anchor_texts = acceptance.anchor_texts(records)
    client = EmbeddingClient(
        api_base=embedding.api_base,
        api_key=embedding.api_key,
        model=embedding.model,
        dimension=embedding.dimension,
        batch_size=embedding.batch_size,
        connect_timeout=embedding.connect_timeout,
        read_timeout=embedding.read_timeout,
        max_retries=embedding.max_retries,
        normalize=embedding.normalize,
    )
    anchors = ruler.VectorSet(
        name="anchors",
        vectors=_embedded_matrix(client, anchor_texts),
    )
    targets = ruler.VectorSet(
        name="targets",
        vectors=_embedded_matrix(client, list(target_set.texts)),
    )

    if target_set.declared_epsilon is not None:
        epsilon = target_set.declared_epsilon
        epsilon_source = f"target-set header 'epsilon' in {target_set.path}"
    else:
        epsilon = acceptance.intrinsic_epsilon(targets)
        epsilon_source = (
            "median nearest-neighbour distance within the target set "
            f"(no 'epsilon' in the header of {target_set.path})"
        )

    noise = acceptance.noise_section(
        anchors,
        acceptance.repeat_groups([record.get("anchor_meta", {}) for record in records]),
    )
    return acceptance.metric_readout(
        anchors,
        targets,
        epsilon=epsilon,
        epsilon_source=epsilon_source,
        anchors_source=anchors_source,
        targets_source=target_set.path,
        embedder=acceptance.EmbedderIdentity(
            model=embedding.model,
            dimension=embedding.dimension,
            normalize=embedding.normalize,
        ),
        noise=noise,
    )


#: The entries of one target-set source, in file order.
RawJsonEntries: TypeAlias = list[Any]

#: One parsed target-set source: its entries plus the header keys that are not
#: the entries themselves (empty for the bare-array form).
ParsedTargetDocument: TypeAlias = tuple[RawJsonEntries, JsonObject]


def _parse_document(raw: str, path: Path) -> ParsedTargetDocument:
    """Parse *raw* as one JSON document or as JSONL; return entries and header."""
    if not raw.strip():
        raise CoverageWiringError(f"target set file {path} is empty")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        return _parse_jsonl(raw, path)
    if isinstance(document, list):
        return _require_non_empty(document, path), {}
    if isinstance(document, dict):
        entries = document.get("targets")
        if not isinstance(entries, list):
            raise CoverageWiringError(
                f"target set {path}: a JSON object document must carry a 'targets' array "
                f"(got {type(entries).__name__})"
            )
        header = {key: value for key, value in document.items() if key != "targets"}
        return _require_non_empty(entries, path), header
    raise CoverageWiringError(
        f"target set {path}: expected a JSON array, a JSON object with 'targets', or "
        f"JSONL, got {type(document).__name__}"
    )


def _parse_jsonl(raw: str, path: Path) -> ParsedTargetDocument:
    """Parse *raw* line by line; a line that is not JSON is a named error."""
    entries: list[Any] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entries.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise CoverageWiringError(
                f"target set {path} line {number} is not valid JSON ({exc}); the file is "
                "neither a single JSON document nor valid JSONL"
            ) from exc
    return _require_non_empty(entries, path), {}


def _require_non_empty(entries: list[Any], path: Path) -> list[Any]:
    """Refuse an empty entry list — an embedding of zero targets is undefined."""
    if not entries:
        raise CoverageWiringError(
            f"target set {path} carries no entries: a nearest-anchor distance over an empty "
            "target set is undefined"
        )
    return entries


def _entry_text(entry: Any, index: int, path: Path) -> str:
    """Return the embedded text of one entry, or refuse with its position."""
    if isinstance(entry, str):
        text: Any = entry
    elif isinstance(entry, dict):
        text = entry.get("text")
    else:
        raise CoverageWiringError(
            f"target set {path} entry {index} must be a string or an object with a "
            f"'text' field, got {type(entry).__name__}"
        )
    if not isinstance(text, str) or not text.strip():
        raise CoverageWiringError(
            f"target set {path} entry {index} has no non-empty 'text' string (got {text!r})"
        )
    return text


def _declared_int(header: Mapping[str, Any], key: str, path: Path) -> int | None:
    """Return a validated integer header field, or ``None`` when absent."""
    if key not in header or header[key] is None:
        return None
    value = header[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise CoverageWiringError(
            f"target set {path} header field {key!r} must be an integer, got {value!r}"
        )
    return value


def _declared_epsilon(header: Mapping[str, Any], path: Path) -> float | None:
    """Return a validated ``epsilon`` header field, or ``None`` when absent."""
    if "epsilon" not in header or header["epsilon"] is None:
        return None
    value = header["epsilon"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CoverageWiringError(
            f"target set {path} header field 'epsilon' must be a number, got {value!r}"
        )
    epsilon = float(value)
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise CoverageWiringError(
            f"target set {path} header field 'epsilon' must be finite and >= 0, got {epsilon!r}"
        )
    return epsilon


def _embedded_matrix(client: EmbeddingClient, texts: list[str]) -> np.ndarray:
    """Embed *texts* and return the ``(n, dimension)`` float64 matrix.

    Raises:
        CoverageWiringError: If the client reports rows it did not normalise —
            :class:`~ard.core.coverage.VectorSet` requires unit-norm rows, and
            "the caller asked for normalisation" is asserted, not assumed.
        EmbeddingError: Propagated unchanged from the client.
    """
    batch = client.embed_texts(texts)
    if not batch.normalized:
        raise CoverageWiringError(
            "the embedding client returned unnormalised vectors: the acceptance ruler "
            "requires L2-normalised rows (set coverage.embedding.normalize = true)"
        )
    return np.asarray(batch.vectors, dtype=np.float64)
