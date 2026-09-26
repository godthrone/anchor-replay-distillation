"""Contract tests for the axis-wording loader (facility half).

Two things must hold for the six ``layer == "instruction"`` axes:

* every value of every such ontology axis has exactly one instruction, and no
  file carries a value the ontology does not declare — a new ontology value
  cannot ship without wording;
* every defect (missing file, invalid JSON, missing value) is a hard error
  naming the axis, the value and the path, never a silent fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ard.backends.axis_instruction_loader import build_axis_requirements, load_axis_instructions
from ard.core.axis_instruction import INSTRUCTION_AXES, AxisInstructionError
from ard.core.ontology import OntologyV4


def test_every_instruction_axis_covers_every_ontology_value(ontology: OntologyV4) -> None:
    """The data files and the ontology agree value for value (coverage contract)."""
    for axis in INSTRUCTION_AXES:
        instructions = load_axis_instructions(axis)
        expected = set(ontology.axis_values(axis))
        assert set(instructions) == expected, (
            f"{axis}: missing {sorted(expected - set(instructions))}, "
            f"extra {sorted(set(instructions) - expected)}"
        )
        assert all(text.strip() for text in instructions.values()), axis


def test_requirements_use_only_the_anchors_own_values() -> None:
    """The clause carries the two present axes' sentences, in axis order."""
    meta = {"difficulty": "advanced", "answer_mode": "structured_analysis"}
    clause = build_axis_requirements(meta)
    advanced = load_axis_instructions("difficulty")["advanced"]
    structured = load_axis_instructions("answer_mode")["structured_analysis"]
    assert advanced in clause
    assert structured in clause
    # Order is INSTRUCTION_AXES order: difficulty precedes answer_mode.
    assert clause.index(advanced) < clause.index(structured)


def test_a_coordinate_without_instruction_axes_adds_nothing() -> None:
    assert build_axis_requirements({}) == ""
    assert build_axis_requirements({"language": "English"}) == ""


def test_a_value_without_an_instruction_is_a_hard_error() -> None:
    with pytest.raises(AxisInstructionError, match="no instruction for difficulty='impossible'"):
        build_axis_requirements({"difficulty": "impossible"})


def test_missing_directory_is_a_hard_error_naming_the_path(tmp_path: Path) -> None:
    with pytest.raises(AxisInstructionError, match="axis wording file not found"):
        load_axis_instructions("difficulty", tmp_path)


def test_invalid_json_is_a_hard_error_naming_the_path(tmp_path: Path) -> None:
    (tmp_path / "difficulty.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(AxisInstructionError, match="invalid JSON"):
        load_axis_instructions("difficulty", tmp_path)


def test_a_document_declaring_another_axis_is_rejected(tmp_path: Path) -> None:
    payload = {"axis": "context_length", "values": {"basic": "Ask an easy question."}}
    (tmp_path / "difficulty.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AxisInstructionError, match="declares axis 'context_length'"):
        load_axis_instructions("difficulty", tmp_path)


def test_wording_directory_is_declared_once_in_the_runtime() -> None:
    """The directory constant lives in the pure module; the loader only reads it.

    A second literal in the loader would be a second source of truth (§1.4), so
    the loader must resolve the same path through
    :func:`ard.core.axis_instruction.axis_instruction_path`.
    """
    from ard.core.axis_instruction import axis_instruction_path

    for axis in INSTRUCTION_AXES:
        expected = axis_instruction_path(axis)
        assert expected.as_posix() == f"configs/prompts/axis_instruction/{axis}.json"
        assert expected.is_file()
