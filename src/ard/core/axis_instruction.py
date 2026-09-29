"""Axis instruction wording — the pure half of the axis-wording contract.

The v4 ontology holds coordinates only (``wording_policy.ontology_holds`` =
``coordinates_only``), so the text that turns an axis *value* into an
instruction for the input generator is data: one JSON file per axis under
:data:`AXIS_INSTRUCTION_DIR`.

Six axes describe what the generated conversation must look like — the six the
ontology labels ``layer = "instruction"`` (vocabulary defined in
``docs/architecture.md`` §7.2):

``response_style`` / ``output_format`` / ``difficulty`` / ``context_length`` /
``input_condition`` / ``answer_mode``.

Their values reached the sampler's coordinates but never the prompt, so two
specs differing only on one of them asked the input generator for exactly the
same message (WP-S14 audit: 935 blocks projected onto the prompt-effective
restricted axes gave only 112, and 4282 block pairs differed only on these
axes).

This module owns the **pure** half of the contract: where the files are, what a
file may contain, and how the per-axis sentences are joined into one
requirement clause.  Reading a file is facility work (§1.3 计算与设施分离) and
lives in :mod:`ard.backends.axis_instruction_loader`.  There is no built-in
wording and no silent fallback: an axis value without an instruction is a hard
error naming the axis, the value and the file (§1.4, §2.3).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ard.core.axis_instruction_error import AxisInstructionError

__all__ = [
    "AXIS_INSTRUCTION_DIR",
    "AXIS_INSTRUCTION_SUFFIX",
    "AXIS_REQUIREMENT_LEAD_IN",
    "INSTRUCTION_AXES",
    "AxisInstructionError",
    "axis_instruction_path",
    "render_axis_instructions",
    "validate_axis_instructions",
]

#: The axes whose values must reach the prompt as an instruction.  The set is
#: the ontology's own ``layer = "instruction"`` partition, stated once here so
#: a consumer cannot silently drop one (§1.4) — the contract test
#: ``test_instruction_axes_are_the_ontology_instruction_layer`` holds the two
#: together.
INSTRUCTION_AXES: tuple[str, ...] = (
    "response_style",
    "output_format",
    "difficulty",
    "context_length",
    "input_condition",
    "answer_mode",
)

#: The wording directory, stated **once** in the runtime, beside the
#: system-prompt wording directory (``configs/prompts/system_prompt``).  It is
#: a sibling family of the same "wording is data" contract
#: (``configs/prompts/README.md``).
AXIS_INSTRUCTION_DIR = Path("configs/prompts/axis_instruction")

#: Wording files are JSON: one object per axis, keyed by the axis value.
AXIS_INSTRUCTION_SUFFIX = ".json"

#: The clause that introduces the per-axis sentences in the generator prompt.
#: Structural glue only — the sentences themselves are the data.
AXIS_REQUIREMENT_LEAD_IN = "The user message you write must also satisfy this:"


def axis_instruction_path(axis: str, directory: str | Path | None = None) -> Path:
    """Return the wording file an instruction axis must have.

    Pure path arithmetic: the path is returned even when the file does not
    exist, so an error message can name what was expected and nothing is opened.

    Args:
        axis: An axis name from :data:`INSTRUCTION_AXES`.
        directory: Wording directory override; defaults to
            :data:`AXIS_INSTRUCTION_DIR`.

    Returns:
        ``<directory>/<axis>.json``.

    Raises:
        AxisInstructionError: If *axis* is not one of :data:`INSTRUCTION_AXES`.
    """
    if axis not in INSTRUCTION_AXES:
        raise AxisInstructionError(
            f"{axis!r} is not an instruction axis: expected one of {list(INSTRUCTION_AXES)}"
        )
    base = Path(directory) if directory is not None else AXIS_INSTRUCTION_DIR
    return base / f"{axis}{AXIS_INSTRUCTION_SUFFIX}"


def validate_axis_instructions(axis: str, document: object, path: str | Path) -> dict[str, str]:
    """Validate one already-parsed wording document and return its mapping.

    Pure: the caller has read and parsed the JSON; *path* is used for error
    messages only.

    Args:
        axis: The instruction axis the document must declare.
        document: The parsed JSON value of ``<axis>.json``.
        path: The file it came from, named in every error message.

    Returns:
        ``{axis value: instruction}`` with each instruction stripped.

    Raises:
        AxisInstructionError: If *axis* is not an instruction axis, the
            document is not an object, it declares another axis, its ``values``
            is not a non-empty object of strings, or any instruction is empty.
    """
    where = Path(path)
    if axis not in INSTRUCTION_AXES:
        raise AxisInstructionError(
            f"{axis!r} is not an instruction axis: expected one of {list(INSTRUCTION_AXES)}"
        )
    if not isinstance(document, dict):
        raise AxisInstructionError(f"axis wording file must be a JSON object: {where}")
    declared = document.get("axis")
    if declared != axis:
        raise AxisInstructionError(
            f"axis wording file {where} declares axis {declared!r}, expected {axis!r}"
        )
    values = document.get("values")
    if not isinstance(values, dict) or not values:
        raise AxisInstructionError(
            f"axis wording file {where} has no non-empty 'values' object "
            f"(expected one instruction per {axis} value)"
        )
    instructions: dict[str, str] = {}
    for value, text in values.items():
        if not isinstance(value, str) or not isinstance(text, str):
            raise AxisInstructionError(
                f"axis wording file {where}: every value key and instruction must be a string"
            )
        stripped = text.strip()
        if not stripped:
            raise AxisInstructionError(
                f"axis wording for {axis}={value!r} is empty in {where}: an axis "
                f"value must carry an instruction, and there is no fallback"
            )
        instructions[value] = stripped
    return instructions


def render_axis_instructions(instructions: Sequence[str]) -> str:
    """Join per-axis sentences into the generator-facing requirement clause.

    Pure and order-preserving: the caller passes the sentences in
    :data:`INSTRUCTION_AXES` order, so the same metadata always renders the
    same bytes (§6 复现).

    Args:
        instructions: One sentence per present instruction axis, in axis order.

    Returns:
        The requirement clause, or ``""`` when no instruction axis is present
        (a coordinate that carries none of the six axes adds no text).
    """
    texts = [text.strip() for text in instructions if text.strip()]
    if not texts:
        return ""
    return f"{AXIS_REQUIREMENT_LEAD_IN} {' '.join(texts)}"
