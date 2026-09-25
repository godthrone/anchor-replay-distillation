"""Ontology file loader — the facility half of :mod:`ard.core.ontology`.

Responsibility: read an ontology v4 document off disk, decode the JSON, and hand
the payload to the pure schema gate
:func:`ard.core.ontology.parse_ontology_v4`.  Reading a file is facility work
(§1.3 计算与设施分离), so it lives here beside the other facilities — never in
``core/``, which stays free of filesystem, network and subprocess access.

Failure is loud and names the path (§2.3 边界校验):

* a missing file raises :class:`FileNotFoundError`;
* malformed JSON raises :class:`~ard.core.ontology.OntologySchemaError` carrying
  the ``line:column`` and the JSON parser's message;
* any schema mismatch is reported field by field by the core gate.

The loader never degrades into an empty ontology and never falls back to a
built-in default: the shipped file and the runtime model are one source of truth
(§1.4).
"""

from __future__ import annotations

import json
from pathlib import Path

from ard.core.ontology import OntologySchemaError, OntologyV4, parse_ontology_v4

__all__ = ["load_ontology_v4"]


def load_ontology_v4(path: str | Path) -> OntologyV4:
    """Read, decode and validate an ontology v4 file.

    Args:
        path: Path to ``anchor_ontology.v4.json``.

    Returns:
        The validated :class:`~ard.core.ontology.OntologyV4` model.

    Raises:
        FileNotFoundError: If the file does not exist.
        OntologySchemaError: If the file is not JSON, is not a JSON object, or
            does not match the v4 schema (missing / extra / wrongly typed field).
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"ontology file not found: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OntologySchemaError(
            source,
            [f"<json>:{exc.lineno}:{exc.colno}: expected valid JSON (received: {exc.msg})"],
        ) from exc
    return parse_ontology_v4(raw, source)
