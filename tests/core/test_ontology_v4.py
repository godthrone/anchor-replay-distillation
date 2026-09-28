# tests/core/test_ontology_v4.py — v4 ontology parser contract tests.
# Responsibility: verify the v4 parser accepts the real ontology and rejects
# schema violations (missing / extra / mistyped fields, absent and malformed
# files) with a message naming the field, the expectation and the received value.
#
# The file-reading loader is facility work (§1.3) and lives in
# ``ard.backends.ontology_loader``; the pure schema gate it feeds is
# ``ard.core.ontology.parse_ontology_v4``.  Both halves are exercised here.

import json
from pathlib import Path
from typing import Any

import pytest

from ard.backends.ontology_loader import load_ontology_v4
from ard.core.constraints import ConstraintEvaluator
from ard.core.ontology import (
    OntologySchemaError,
    OntologyV4,
    parse_ontology_v4,
)

ONTOLOGY_V4_PATH = Path("ontology/anchor_ontology.v4.json")

#: A minimal *v3-shaped* ontology: the top-level keys the pre-v4 loader read and
#: the v4 schema does not declare.  Inlined on purpose — the v4 loader must
#: reject it on shape alone, without depending on a v3 file that no longer ships
#: (§18.1: the old asset is gone, so nothing may reference it backwards).
V3_SHAPED_ONTOLOGY: dict[str, Any] = {
    "languages": ["English"],
    "knowledge_domains": {"science": {"sub": ["physics"]}},
    "capabilities": {"knowledge_response": ["qa"]},
    "system_prompt_presence": ["none", "present"],
    "system_prompt_style": {"minimal_persona": {"description": "one sentence"}},
    "conversation_types": {"single_turn": ["single_turn"]},
}


def _raw_v4() -> dict[str, Any]:
    """Return a fresh, mutable copy of the real v4 ontology payload."""
    payload: dict[str, Any] = json.loads(ONTOLOGY_V4_PATH.read_text(encoding="utf-8"))
    return payload


# ── Happy path ──────────────────────────────────────────────────────────────


def test_load_v4_parses_real_ontology() -> None:
    """The shipped v4 file parses into the typed model.

    **shipped-ontology contract**：下面的 12 / 8 只对当前随仓库发布的
    这份本体成立；运行时不依赖它 —— 轴数由 ``len(axis_names())`` 现算，运行时
    计数一律从叶子清单穷举。改本体时更新这个数字即可，不需要改任何生产代码。
    """
    ontology = load_ontology_v4(ONTOLOGY_V4_PATH)
    assert ontology.ontology_id == "ard-anchor-ontology"
    assert ontology.version == "4.0.0"
    assert len(ontology.axis_names()) == 12
    assert len(ontology.constraints) == 8


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        ("language", 4),
        ("knowledge_domain", 209),
        ("capability", 20),
        ("system_prompt_mode", 5),
        ("conversation_type", 7),
        ("response_style", 7),
        ("output_format", 6),
        ("difficulty", 3),
        ("context_length", 3),
        ("input_condition", 6),
        ("answer_mode", 4),
        ("visual_domain", 21),
    ],
)
def test_axis_value_cardinalities(axis: str, expected: int) -> None:
    """Every axis resolves to its declared number of coordinate values.

    **shipped-ontology contract**：这些基数（4 / 209 / 20 / 5 / 7 / 7 / 6 / 3 /
    3 / 6 / 4 / 21）只对当前随仓库发布的这份本体成立；运行时不依赖它 —— 运行时
    计数一律从叶子清单穷举。改本体时更新这些数字即可，不需要改任何生产代码。
    """
    ontology = load_ontology_v4(ONTOLOGY_V4_PATH)
    assert len(ontology.axis_values(axis)) == expected


def test_knowledge_domain_tree_shape() -> None:
    """The knowledge-domain tree is 18 domains / 36 subdomains / 209 unique leaves.

    **shipped-ontology contract**：18 / 36 / 209 只对当前随仓库发布的这份本体
    成立；运行时不依赖它 —— 运行时计数一律从叶子清单穷举。改本体时更新这些数字
    即可，不需要改任何生产代码。
    """
    ontology = load_ontology_v4(ONTOLOGY_V4_PATH)
    tree = ontology.knowledge_domain_tree.root
    assert len(tree) == 18
    subdomains = [name for subs in tree.values() for name in subs.root]
    leaves = [leaf for subs in tree.values() for group in subs.root.values() for leaf in group]
    assert len(subdomains) == 36
    assert len(leaves) == 209
    assert len(set(leaves)) == 209


def test_ontology_carries_no_handwritten_count_block() -> None:
    """Leaves are the single source: no self-reported counts exist any more.

    Both the raw payload and the typed model are checked, so a count block
    cannot come back through either door (§1.4 single source of truth, §18.1
    no leftover debt).
    """
    payload = _raw_v4()
    assert "derived_counts" not in payload
    assert "reachability" not in payload
    assert all("counts" not in spec for spec in payload["axes"].values())
    assert "derived_counts" not in OntologyV4.model_fields
    assert "reachability" not in OntologyV4.model_fields


def test_added_knowledge_domain_leaf_needs_no_other_field_touched(tmp_path: Path) -> None:
    """Adding a leaf edits the leaf list only; load + enumeration still work.

    There is no count block to keep in sync, so the new leaf flows straight
    into the axis value set and the restricted-block enumeration is unchanged
    (``knowledge_domain`` is a free axis).
    """
    payload = _raw_v4()
    payload["knowledge_domain_tree"]["science_exploration"]["frontier_questions"].append(
        "gravitational wave background"
    )
    path = tmp_path / "added_leaf.v4.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    ontology = load_ontology_v4(path)
    baseline = load_ontology_v4(ONTOLOGY_V4_PATH)
    assert "gravitational wave background" in ontology.axis_values("knowledge_domain")
    assert len(ontology.axis_values("knowledge_domain")) == (
        len(baseline.axis_values("knowledge_domain")) + 1
    )
    assert len(ConstraintEvaluator(ontology).enumerate_legal_blocks()) == len(
        ConstraintEvaluator(baseline).enumerate_legal_blocks()
    )


def test_constraint_ids_and_types() -> None:
    """All eight constraints are present with the expected discriminated types."""
    ontology = load_ontology_v4(ONTOLOGY_V4_PATH)
    expected_ids = ["R1", "R2", "R3", "R4a", "R4b", "R5", "R6", "R7"]
    assert [c.id for c in ontology.constraints] == expected_ids
    assert [c.id for c in ontology.allowed_pairs_constraints()] == [
        "R1",
        "R2",
        "R3",
        "R4a",
        "R4b",
    ]
    assert ontology.modality_gate().gated_axis == "visual_domain"


def test_axis_values_unknown_axis_raises() -> None:
    """Asking for an axis that does not exist is an error, not an empty tuple."""
    ontology = load_ontology_v4(ONTOLOGY_V4_PATH)
    with pytest.raises(KeyError):
        ontology.axis_values("no_such_axis")


def test_extra_keys_are_forbidden() -> None:
    """extra="forbid" is on: any undeclared key is a schema error."""
    with pytest.raises(ValueError, match="extra_forbidden|Extra inputs"):
        OntologyV4.model_validate({"version": "4.0.0", "undeclared": 1})


# ── The pure schema gate: already-decoded payloads, no file involved ─────────


def test_pure_parser_rejects_a_non_object_payload() -> None:
    """``parse_ontology_v4`` rejects a decoded non-object without touching disk."""
    with pytest.raises(OntologySchemaError) as excinfo:
        parse_ontology_v4([1, 2, 3], "inline-payload")
    assert "expected object" in str(excinfo.value)
    assert "inline-payload" in str(excinfo.value)


def test_pure_parser_names_the_source_and_the_field() -> None:
    """A schema mismatch names the source label and the offending field."""
    payload = _raw_v4()
    del payload["version"]
    with pytest.raises(OntologySchemaError) as excinfo:
        parse_ontology_v4(payload, "memory-payload")
    message = str(excinfo.value)
    assert "memory-payload" in message
    assert "version" in message


# ── Schema rejection ────────────────────────────────────────────────────────


def test_missing_top_level_field_is_rejected(tmp_path: Path) -> None:
    """A missing top-level field is reported with its name."""
    payload = _raw_v4()
    del payload["version"]
    path = tmp_path / "missing_version.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OntologySchemaError) as excinfo:
        load_ontology_v4(path)

    message = str(excinfo.value)
    assert "version" in message
    assert "expected" in message
    assert "<nothing>" in message


def test_missing_nested_field_is_rejected(tmp_path: Path) -> None:
    """A missing nested field is reported with its dotted path."""
    payload = _raw_v4()
    del payload["axes"]["capability"]["groups"]
    path = tmp_path / "missing_nested.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OntologySchemaError) as excinfo:
        load_ontology_v4(path)

    assert "axes.capability.groups" in str(excinfo.value)


def test_extra_field_is_rejected(tmp_path: Path) -> None:
    """An undeclared field is rejected and named."""
    payload = _raw_v4()
    payload["capability"] = ["qa"]
    path = tmp_path / "extra_field.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OntologySchemaError) as excinfo:
        load_ontology_v4(path)

    message = str(excinfo.value)
    assert "capability" in message
    assert "received" in message


def test_type_error_is_rejected(tmp_path: Path) -> None:
    """A wrongly typed field is rejected, naming field, expectation and value."""
    payload = _raw_v4()
    payload["axes"]["conversation_type"]["value_attributes"]["single_turn"]["turns"] = ["one"]
    path = tmp_path / "type_error.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OntologySchemaError) as excinfo:
        load_ontology_v4(path)

    message = str(excinfo.value)
    assert "single_turn.turns.int" in message
    assert "expected int" in message
    assert "['one']" in message


def test_flat_axis_values_must_be_a_list(tmp_path: Path) -> None:
    """A scalar where a list is expected is rejected."""
    payload = _raw_v4()
    payload["axes"]["language"]["values"] = "English"
    path = tmp_path / "scalar_values.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OntologySchemaError) as excinfo:
        load_ontology_v4(path)

    message = str(excinfo.value)
    assert "axes.language.values" in message
    assert "expected list" in message


def test_constraint_extra_field_is_rejected(tmp_path: Path) -> None:
    """An undeclared key inside a constraint is rejected (nested extra=forbid)."""
    payload = _raw_v4()
    payload["constraints"][0]["unexpected_rule"] = "x"
    path = tmp_path / "constraint_extra.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OntologySchemaError) as excinfo:
        load_ontology_v4(path)

    message = str(excinfo.value)
    assert "constraints.0" in message
    assert "unexpected_rule" in message
    assert "received" in message


def test_missing_file_raises_file_not_found() -> None:
    """A missing ontology file is a FileNotFoundError, not an empty result."""
    with pytest.raises(FileNotFoundError) as excinfo:
        load_ontology_v4("ontology/does_not_exist.v4.json")
    assert "ontology" in str(excinfo.value)


def test_malformed_json_is_rejected(tmp_path: Path) -> None:
    """Malformed JSON is rejected loudly rather than degrading silently."""
    path = tmp_path / "broken.json"
    path.write_text('{"version": "4.0.0",}', encoding="utf-8")
    with pytest.raises(OntologySchemaError) as excinfo:
        load_ontology_v4(path)
    assert "expected valid JSON" in str(excinfo.value)


def test_non_object_json_is_rejected(tmp_path: Path) -> None:
    """A JSON array is not an ontology object."""
    path = tmp_path / "array.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(OntologySchemaError) as excinfo:
        load_ontology_v4(path)
    assert "expected object" in str(excinfo.value)


def test_v3_shaped_payload_is_rejected_by_v4_loader(tmp_path: Path) -> None:
    """The real defect: feeding a v3 shape to the v4 path errors instead of yielding 0."""
    path = tmp_path / "v3_shaped.json"
    path.write_text(json.dumps(V3_SHAPED_ONTOLOGY), encoding="utf-8")
    with pytest.raises(OntologySchemaError) as excinfo:
        load_ontology_v4(path)
    assert "languages" in str(excinfo.value)
