"""Embedding utilities for ARD anchor ontology.

Pure computation layer: loading pre-computed embeddings from JSON and naming the
embedding space they define.  Farthest-point selection does **not** live here —
it lives in :mod:`ard.core.cloud`, because a selection must carry the identity of
the cloud it came from (a bare position list is exactly what once allowed one
cloud's row numbers to index another cloud).  No network, no GPU management, no
file-system side-effects beyond the JSON load path.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

# ---------------------------------------------------------------------------
# 1.  Loading pre-computed embeddings
# ---------------------------------------------------------------------------


def load_embeddings(path: str) -> dict[str, Any]:
    """Load pre-computed anchor-ontology embeddings from a JSON file.

    The file follows the ``anchor_ontology_embeddings.json`` schema written by
    ``scripts/generate_ontology_embeddings.py``.  Its top-level keys are:

    * ``model`` — name of the embedding model the vectors came from, as passed
      to the embedding API (the shipped file records ``"embedding"``).
    * ``embedding_dimension`` — int, the length of every vector (1024 in the
      shipped file).
    * ``ontology_sha256`` — hex digest of the ``anchor_ontology.json`` the
      vectors were built from.  Provenance only: the runtime never compares it
      against the ontology actually in use (an ontology edit that leaves the
      embedded texts untouched does not invalidate the vectors).
    * ``generated_at`` — ISO-8601 UTC timestamp of the generation run.
    * ``items`` — a **dict of dicts**, not a list.  It maps a section name to
      that section's ``{item name: vector}`` table::

          "items": {
              "knowledge_domains":  {domain_name:     [float, ...]},
              "capabilities":       {capability_name: [float, ...]},
              "languages":          {language_name:   [float, ...]},
              "conversation_types": {conv_type_name:  [float, ...]},
              "system_prompt":      {presence_or_style_name: [float, ...]},
          }

      Every vector is a flat list of ``embedding_dimension`` floats.  The
      ``system_prompt`` section is *derived* from the capability vectors rather
      than embedded through the API; see
      ``scripts/generate_ontology_embeddings.py`` for the derivation.

    There is no ``distance`` key — the metric is not data.  It is fixed in the
    code that consumes the vectors: :func:`ard.core.cloud.fps` normalises each
    vector to unit length and measures **cosine** distance
    (``1 - cosine similarity``).

    An ``items`` **list** — the previous format, whose per-item dicts carried
    ``section`` / ``path`` / ``leaf`` / ``text`` / ``embedding`` next to
    top-level ``embedding_model`` and ``distance`` keys — is not this schema;
    :func:`ard.core._fps.validate_embeddings_for_ontology` rejects it (check
    V8).

    Returns the full parsed dict.  No validation beyond JSON parsing is
    performed here — callers are expected to validate the structure if needed,
    and the sampling path does so via
    :func:`ard.core._fps.validate_embeddings_for_ontology` (section presence,
    agreement with the ontology's keys, consistent vector length).
    """
    raw = pathlib.Path(path).read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(raw)
    return data


# ---------------------------------------------------------------------------
# 2.  Embedding-space identity
# ---------------------------------------------------------------------------


def embedding_space_id(data: dict[str, Any]) -> str:
    """Fingerprint of the embedding space a loaded embeddings file describes.

    The fingerprint is ``"{model}:{embedding_dimension}"`` — the embedder name
    recorded by the generator plus the vector length.  It becomes the
    ``space_id`` of every :class:`ard.core.cloud.CloudVectors` built from this
    file, so that two clouds may only be compared arithmetically when their
    fingerprints match.

    Args:
        data: Parsed ``anchor_ontology_embeddings.json`` (see
            :func:`load_embeddings`).

    Returns:
        Non-empty fingerprint string, e.g. ``"embedding:1024"``.

    Raises:
        ValueError: If the file carries no ``model`` or ``embedding_dimension``.
            A space without an identity cannot be guarded (§2.3 边界校验即防呆),
            so this fails at the boundary instead of defaulting to a wildcard.
    """
    model = data.get("model")
    dimension = data.get("embedding_dimension")
    if not model or not dimension:
        raise ValueError(
            "embeddings file has no usable 'model'/'embedding_dimension': the embedding "
            "space cannot be identified, so space-safe operations cannot be used"
        )
    return f"{model}:{dimension}"
