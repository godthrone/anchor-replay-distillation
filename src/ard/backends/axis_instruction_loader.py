"""Axis-wording loader — the facility half of :mod:`ard.core.axis_instruction`.

Responsibility: locate and read one ``<axis>.json`` wording file off disk, then
hand the parsed document to the pure contract in
:mod:`ard.core.axis_instruction`.  Reading a file is facility work (§1.3 计算与
设施分离), so it lives here beside the other facilities — ``core/`` stays free of
filesystem access.

Everything is checked loudly and nothing is swallowed (§2.3): a missing
directory, a missing file, an unreadable file, invalid JSON, a document that
declares another axis and an empty instruction are all
:class:`~ard.core.axis_instruction.AxisInstructionError` naming the path.  An
anchor whose axis value has no instruction is also a hard error naming the axis
and the value — there is no built-in fallback wording, which would restore a
second source of truth (§1.4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ard.core.axis_instruction import (
    INSTRUCTION_AXES,
    AxisInstructionError,
    axis_instruction_path,
    render_axis_instructions,
    validate_axis_instructions,
)

__all__ = ["build_axis_requirements", "load_axis_instructions"]


def load_axis_instructions(axis: str, directory: str | Path | None = None) -> dict[str, str]:
    """Read one instruction axis's value → wording mapping from disk.

    Args:
        axis: An axis name from
            :data:`~ard.core.axis_instruction.INSTRUCTION_AXES`.
        directory: Wording directory override; defaults to the directory the
            runtime declares
            (:data:`~ard.core.axis_instruction.AXIS_INSTRUCTION_DIR`).

    Returns:
        ``{axis value: instruction}``.

    Raises:
        AxisInstructionError: If the file is missing, unreadable, not valid
            JSON, or fails the pure document contract.
    """
    path = axis_instruction_path(axis, directory)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AxisInstructionError(
            f"axis wording file not found: {path} (expected one JSON file per "
            f"instruction axis, holding every value of the ontology axis)"
        ) from exc
    except OSError as exc:
        raise AxisInstructionError(f"cannot read axis wording file {path}: {exc.strerror}") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AxisInstructionError(f"invalid JSON in axis wording file {path}: {exc}") from exc
    return validate_axis_instructions(axis, document, path)


def build_axis_requirements(
    anchor_meta: dict[str, Any], directory: str | Path | None = None
) -> str:
    """Build the requirement clause for an anchor's instruction axes.

    Reads each instruction axis's wording file and fills the clause with the
    sentence belonging to the anchor's own value, in
    :data:`~ard.core.axis_instruction.INSTRUCTION_AXES` order.

    An axis the coordinate does not carry (``None``) adds nothing: the clause
    describes the coordinate that was sampled, and a coordinate without the
    axis has no value to express.  A value that *is* carried but has no
    instruction is a hard error — never a silent skip, which is what made these
    axes inert in the first place.

    Args:
        anchor_meta: The anchor's coordinate mapping.
        directory: Wording directory override (tests only).

    Returns:
        The requirement clause, or ``""`` when no instruction axis is present.

    Raises:
        AxisInstructionError: If a wording file is missing or malformed, or an
            axis value the coordinate carries has no instruction.
    """
    instructions: list[str] = []
    for axis in INSTRUCTION_AXES:
        value = anchor_meta.get(axis)
        if value is None:
            continue
        allowed = load_axis_instructions(axis, directory)
        if value not in allowed:
            path = axis_instruction_path(axis, directory)
            raise AxisInstructionError(
                f"no instruction for {axis}={value!r}: {path} defines {sorted(allowed)}"
            )
        instructions.append(allowed[value])
    return render_axis_instructions(instructions)
