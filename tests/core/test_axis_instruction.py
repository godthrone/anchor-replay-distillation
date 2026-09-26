"""Contract tests for the pure axis-instruction wording module.

The module declares *where* the wording files are and *what* a file may contain;
reading them is the backend loader's job.  These tests pin the pure half: the
axis set matches the ontology's own ``layer == "instruction"`` partition, path
arithmetic is exact, malformed documents are rejected by name, and rendering is
order-preserving and empty when nothing is present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ard.core.axis_instruction import (
    AXIS_INSTRUCTION_DIR,
    AXIS_INSTRUCTION_SUFFIX,
    AXIS_REQUIREMENT_LEAD_IN,
    INSTRUCTION_AXES,
    AxisInstructionError,
    axis_instruction_path,
    render_axis_instructions,
    validate_axis_instructions,
)
from ard.core.ontology import OntologyV4


def test_instruction_axes_are_the_ontology_instruction_layer(ontology: OntologyV4) -> None:
    """``INSTRUCTION_AXES`` must equal the ontology's ``layer = instruction`` axes.

    The set is stated once in code so a consumer cannot silently drop an axis;
    this test holds it to the ontology instead of to a hand-copied list (§1.4).
    """
    declared = tuple(
        axis for axis in ontology.axis_names() if ontology.axes.spec(axis).layer == "instruction"
    )
    assert set(declared) == set(INSTRUCTION_AXES), (
        f"instruction layer {sorted(declared)} != INSTRUCTION_AXES {sorted(INSTRUCTION_AXES)}"
    )
    assert declared == INSTRUCTION_AXES, (
        "INSTRUCTION_AXES must follow the ontology's axis order so rendering is deterministic"
    )


def test_axis_instruction_path_is_the_declared_location() -> None:
    path = axis_instruction_path("answer_mode")
    assert path == AXIS_INSTRUCTION_DIR / f"answer_mode{AXIS_INSTRUCTION_SUFFIX}"
    assert path == Path("configs/prompts/axis_instruction/answer_mode.json")


def test_axis_instruction_path_rejects_a_non_instruction_axis() -> None:
    with pytest.raises(AxisInstructionError, match="not an instruction axis"):
        axis_instruction_path("language")


def _document(**overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "axis": "difficulty",
        "values": {"basic": "Ask an easy question."},
    }
    document.update(overrides)
    return document


def test_validate_returns_the_stripped_mapping() -> None:
    got = validate_axis_instructions(
        "difficulty", _document(values={"basic": "  Ask an easy question.  "}), "x.json"
    )
    assert got == {"basic": "Ask an easy question."}


def test_validate_rejects_a_document_that_declares_another_axis() -> None:
    with pytest.raises(AxisInstructionError, match="declares axis"):
        validate_axis_instructions("difficulty", _document(axis="context_length"), "x.json")


@pytest.mark.parametrize(
    ("document", "match"),
    [
        (["not", "an", "object"], "must be a JSON object"),
        ({"axis": "difficulty"}, "no non-empty 'values' object"),
        ({"axis": "difficulty", "values": {}}, "no non-empty 'values' object"),
        ({"axis": "difficulty", "values": {"basic": "   "}}, "is empty"),
        ({"axis": "difficulty", "values": {"basic": 7}}, "must be a string"),
    ],
)
def test_validate_rejects_malformed_documents(document: object, match: str) -> None:
    with pytest.raises(AxisInstructionError, match=match):
        validate_axis_instructions("difficulty", document, "configs/prompts/x.json")


def test_render_joins_in_order_under_the_lead_in() -> None:
    rendered = render_axis_instructions(["First sentence.", "Second sentence."])
    assert rendered == f"{AXIS_REQUIREMENT_LEAD_IN} First sentence. Second sentence."


def test_render_is_empty_without_instructions() -> None:
    assert render_axis_instructions([]) == ""
    assert render_axis_instructions(["   ", ""]) == ""
