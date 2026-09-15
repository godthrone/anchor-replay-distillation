"""Tests for ARD — domain bank, text_anchor prompts, and pipeline."""

import json
from pathlib import Path

import pytest

from ard.domain.bank import (
    AppendOutcome,
    anchor_to_dict,
    append_anchor,
    build_manifest,
    build_manifest_from_records,
    count_existing_anchors,
    count_unique_anchor_ids,
    read_anchor_bank,
    write_anchor_bank,
    write_manifest,
)
from ard.domain.anchor_shape import message_shape_error
from ard.domain.text_anchor import build_input_prompt, build_target_prompt
from ard.core.sampler import generate_anchor_id
from ard.core.types import DataSource, GeneratedAnchor, AnchorGenerationConfig


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_anchor(id="a", **kwargs):
    defaults = {
        "id": id,
        "messages": [{"role": "user", "content": "q"}],
        "target_answer": "answer",
        "target_model": "target",
        "input_generator_model": "input-gen",
        "anchor_meta": {"knowledge_domain": "math", "language": "English", "capability": "qa"},
    }
    defaults.update(kwargs)
    return GeneratedAnchor(**defaults)


# ── anchor_to_dict ──────────────────────────────────────────────────────────


def test_anchor_to_dict_format():
    """anchor_to_dict produces the unified JSONL format."""
    a = _make_anchor("test_001", reasoning="six times seven is forty-two")
    d = anchor_to_dict(a)
    assert d["id"] == "test_001"
    assert d["source"] == "ard"
    assert d["messages"] == [{"role": "user", "content": "q"}]
    assert d["targets"][0]["id"] == "primary"
    assert d["targets"][0]["output"]["content"] == "answer"
    assert d["targets"][0]["output"]["reasoning"] == "six times seven is forty-two"
    assert d["anchor_meta"] == {"knowledge_domain": "math", "language": "English", "capability": "qa"}
    assert d["teacher_id"] == "target"


def test_anchor_to_dict_reasoning_is_null_when_absent():
    """Without thinking the output carries ``reasoning: null`` — never ``""``."""
    a = _make_anchor("a")
    d = anchor_to_dict(a)
    assert d["targets"][0]["output"]["reasoning"] is None
    # The two keys are distinct: content survives, reasoning is explicitly null.
    assert d["targets"][0]["output"]["content"] == "answer"
    assert "logprobs" not in d["targets"][0]["output"]


def test_anchor_to_dict_reasoning_never_merged_into_content():
    """``content`` and ``reasoning`` stay two separate keys (§1.2 契约 2).

    Each key carries *exactly* its own source text: no concatenation in either
    direction, and no key missing from the record.
    """
    reasoning = "step one; step two; the answer is 42"
    a = _make_anchor("b", target_answer="42", reasoning=reasoning)
    output = anchor_to_dict(a)["targets"][0]["output"]
    assert output["content"] == "42", "the answer must not absorb the reasoning"
    assert output["reasoning"] == reasoning, "the reasoning must not absorb the answer"


def test_anchor_to_dict_persists_new_v3_fields():
    """``data_source`` / ``schema_version`` / ``input_generator_model`` land
    as top-level fields on the serialized record."""
    a = _make_anchor(
        "c",
        input_generator_model="input-model",
        # B3 tightened this field into the controlled vocabulary: the routing key
        # is a DataSource member now, not an arbitrary string (W-1).
        data_source=DataSource.ARD_MULTI,
    )
    d = anchor_to_dict(a)
    assert d["data_source"] == "ard_multi"
    assert d["schema_version"] == "3.0.0"
    assert d["input_generator_model"] == "input-model"
    assert d["teacher_id"] == "target"  # untouched sibling key


def test_generated_anchor_rejects_off_vocabulary_data_source():
    """A raw string outside the vocabulary cannot build an anchor (W-1)."""
    with pytest.raises(ValueError, match="must be a DataSource"):
        _make_anchor("c2", data_source="ard_multi")


def test_anchor_to_dict_defaults_data_source_to_ard_text():
    """A text-only anchor defaults to the ``ard_text`` routing key."""
    d = anchor_to_dict(_make_anchor("d"))
    assert d["data_source"] == "ard_text"
    assert d["schema_version"] == "3.0.0"


def test_persisted_record_carries_new_v3_fields(tmp_path):
    """Every record written through the real persistence path carries the new
    top-level fields, readable back from disk."""
    path = tmp_path / "bank.jsonl"
    append_anchor(_make_anchor("e", input_generator_model="pin-model"), path)
    records = read_anchor_bank(path)
    assert records[0]["data_source"] == "ard_text"
    assert records[0]["schema_version"] == "3.0.0"
    assert records[0]["input_generator_model"] == "pin-model"


# ── Bank ────────────────────────────────────────────────────────────────────


def test_write_and_read_anchor_bank(tmp_path):
    """write_anchor_bank + read_anchor_bank round-trip."""
    path = tmp_path / "bank.jsonl"
    anchors = [_make_anchor("a"), _make_anchor("b")]
    write_anchor_bank(anchors, path)
    records = read_anchor_bank(path)
    assert len(records) == 2
    assert records[0]["id"] == "a"
    assert records[0]["source"] == "ard"
    assert records[0]["teacher_id"] == "target"
    assert "targets" in records[0]
    assert records[0]["targets"][0]["output"]["content"] == "answer"


def test_write_anchor_bank_creates_parent_dir(tmp_path):
    """write_anchor_bank creates parent directories."""
    path = tmp_path / "subdir" / "nested" / "bank.jsonl"
    anchors = [_make_anchor()]
    write_anchor_bank(anchors, path)
    assert path.exists()


# ── append_anchor ───────────────────────────────────────────────────────────


def test_append_anchor_single(tmp_path):
    """append_anchor writes a single anchor to a JSONL file."""
    path = tmp_path / "bank.jsonl"
    a = _make_anchor("single")
    append_anchor(a, path)
    assert path.exists()
    records = read_anchor_bank(path)
    assert len(records) == 1
    assert records[0]["id"] == "single"


def test_append_anchor_multiple(tmp_path):
    """append_anchor appends multiple anchors correctly."""
    path = tmp_path / "bank.jsonl"
    for i in range(5):
        append_anchor(_make_anchor(f"anchor_{i}"), path)
    records = read_anchor_bank(path)
    assert len(records) == 5
    assert [r["id"] for r in records] == [f"anchor_{i}" for i in range(5)]


def test_append_anchor_creates_file(tmp_path):
    """append_anchor creates the file if it doesn't exist (parent dir must exist)."""
    path = tmp_path / "new_dir" / "bank.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    append_anchor(_make_anchor(), path)
    assert path.exists()


# ── Output gates: message shape ─────────────────────────────────────────────


def _uau_messages():
    return [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]


def test_message_shape_error_accepts_valid_shapes():
    """The exit contract accepts U, UAU and UAUAU."""
    assert message_shape_error([{"role": "user", "content": "q"}]) is None
    assert message_shape_error(_uau_messages()) is None
    assert message_shape_error(
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
        ]
    ) is None


def test_message_shape_error_rejects_uauau_and_uauu():
    """The exit contract rejects a trailing assistant and consecutive users."""
    trailing_assistant = _uau_messages() + [{"role": "assistant", "content": "a2"}]
    assert message_shape_error(trailing_assistant) is not None
    assert "last message role" in message_shape_error(trailing_assistant)

    consecutive_users = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "user", "content": "q3"},
    ]
    assert message_shape_error(consecutive_users) is not None
    assert "alternate" in message_shape_error(consecutive_users)

    assert message_shape_error([]) is not None
    assert message_shape_error([{"role": "assistant", "content": "a"}]) is not None


# ── v3.0.0 D1: optional single leading ``system`` ───────────────────────────


def _with_system(messages, content="You are a helpful assistant."):
    """Prepend a single leading ``system`` message (D1, position 0)."""
    return [{"role": "system", "content": content}, *messages]


def test_message_shape_error_accepts_optional_leading_system():
    """A single leading ``system`` is allowed for U, UAU and UAUAU (D1)."""
    assert message_shape_error(_with_system([{"role": "user", "content": "q"}])) is None
    assert message_shape_error(_with_system(_uau_messages())) is None
    assert message_shape_error(
        _with_system(
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"},
            ]
        )
    ) is None


def test_message_shape_error_rejects_system_not_in_first_position():
    """A ``system`` message anywhere but position 0 is explicitly refused."""
    misplaced = [
        {"role": "user", "content": "q"},
        {"role": "system", "content": "late system"},
    ]
    err = message_shape_error(misplaced)
    assert err is not None
    assert "system" in err and "first" in err

    # Behind an assistant/earlier slot: also refused.
    mid = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "mid system"},
        {"role": "user", "content": "q2"},
    ]
    assert message_shape_error(mid) is not None


def test_message_shape_error_rejects_multiple_system_messages():
    """More than one ``system`` message is refused (D1: at most one)."""
    dup = [
        {"role": "system", "content": "s1"},
        {"role": "system", "content": "s2"},
        {"role": "user", "content": "q"},
    ]
    err = message_shape_error(dup)
    assert err is not None
    assert "at most one" in err


def test_message_shape_error_rejects_system_only_and_system_after_strip_invalid():
    """A bare ``system`` (nothing to strip to) is refused; so is a system
    followed by a non-alternating conversation."""
    assert message_shape_error([{"role": "system", "content": "only"}]) is not None
    # system + [user, user] must still fail the alternation contract.
    non_alt = _with_system(
        [
            {"role": "user", "content": "q1"},
            {"role": "user", "content": "q2"},
        ]
    )
    err = message_shape_error(non_alt)
    assert err is not None
    assert "alternate" in err


def test_append_anchor_accepts_leading_system(tmp_path):
    """The persistence gate writes a conversation opened by a ``system``."""
    path = tmp_path / "bank.jsonl"
    outcome = append_anchor(
        _make_anchor("sys-ok", messages=_with_system(_uau_messages())),
        path,
    )
    assert outcome is AppendOutcome.APPENDED
    records = read_anchor_bank(path)
    assert len(records) == 1
    roles = [m["role"] for m in records[0]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_append_anchor_rejects_misplaced_system(tmp_path):
    """A non-leading ``system`` is refused by the real persistence gate."""
    path = tmp_path / "bank.jsonl"
    misplaced = [
        {"role": "user", "content": "q1"},
        {"role": "system", "content": "late"},
        {"role": "user", "content": "q2"},
    ]
    outcome = append_anchor(_make_anchor("sys-bad", messages=misplaced), path)
    assert outcome is AppendOutcome.INVALID_SHAPE_SKIPPED
    assert count_existing_anchors(path) == 0


def test_append_anchor_accepts_valid_shape(tmp_path):
    """The persistence gate writes a well-formed conversation."""
    path = tmp_path / "bank.jsonl"
    outcome = append_anchor(_make_anchor("ok", messages=_uau_messages()), path)
    assert outcome is AppendOutcome.APPENDED
    assert len(read_anchor_bank(path)) == 1


def test_append_anchor_rejects_invalid_shape(tmp_path, caplog):
    """A UAUU-shaped anchor is refused *and* reported — not silently dropped."""
    path = tmp_path / "bank.jsonl"
    broken = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "user", "content": "q3"},
    ]
    with caplog.at_level("WARNING"):
        outcome = append_anchor(_make_anchor("broken", messages=broken), path)

    assert outcome is AppendOutcome.INVALID_SHAPE_SKIPPED
    assert not path.exists(), "a malformed anchor must not reach the bank"
    assert any("shape gate" in record.message for record in caplog.records)
    assert any("broken" in record.getMessage() for record in caplog.records)


def test_append_anchor_rejects_trailing_assistant(tmp_path):
    """A UAUAU-shaped anchor (trailing assistant) is refused."""
    path = tmp_path / "bank.jsonl"
    messages = _uau_messages() + [{"role": "assistant", "content": "a2"}]
    outcome = append_anchor(_make_anchor("tail", messages=messages), path)
    assert outcome is AppendOutcome.INVALID_SHAPE_SKIPPED
    assert count_existing_anchors(path) == 0


def test_write_anchor_bank_refuses_invalid_shape(tmp_path):
    """The bulk writer refuses a malformed conversation rather than hiding it."""
    path = tmp_path / "bank.jsonl"
    with pytest.raises(ValueError, match="malformed"):
        write_anchor_bank(
            [
                _make_anchor("good"),
                _make_anchor("bad", messages=[{"role": "user", "content": "q"},
                                              {"role": "user", "content": "q2"}]),
            ],
            path,
        )
    assert not path.exists()


# ── Output gates: id de-duplication ─────────────────────────────────────────


def test_generate_anchor_id_is_metadata_derived():
    """Ids are a pure function of the 4 metadata dimensions (must not change)."""
    meta = {
        "language": "English",
        "knowledge_domain": "math",
        "capability": "qa",
        "conversation_type": "single_turn",
    }
    anchor_id = generate_anchor_id(meta)
    assert anchor_id.startswith("anchor_")
    assert len(anchor_id) == len("anchor_") + 16
    assert generate_anchor_id(dict(meta)) == anchor_id


def test_append_anchor_deduplicates_by_id(tmp_path, caplog):
    """Two distinct anchors sharing an id produce exactly one bank row."""
    path = tmp_path / "bank.jsonl"
    meta = {
        "language": "English",
        "knowledge_domain": "math",
        "capability": "qa",
        "conversation_type": "single_turn",
    }
    anchor_id = generate_anchor_id(meta)

    first = _make_anchor(anchor_id, anchor_meta=meta, target_answer="first answer")
    second = _make_anchor(anchor_id, anchor_meta=meta, target_answer="second answer")

    with caplog.at_level("WARNING"):
        assert append_anchor(first, path) is AppendOutcome.APPENDED
        assert append_anchor(second, path) is AppendOutcome.DUPLICATE_SKIPPED

    records = read_anchor_bank(path)
    assert len(records) == 1, "the duplicate id must not add a second row"
    assert records[0]["targets"][0]["output"]["content"] == "first answer"
    assert any("duplicate not written" in record.getMessage() for record in caplog.records)
    # Lines and real anchors agree — the property the resume logic relies on.
    assert count_existing_anchors(path) == count_unique_anchor_ids(path) == 1


def test_append_anchor_dedup_survives_across_calls_without_cache(tmp_path):
    """De-duplication is stateful per file: a rewritten bank cannot shadow new ids."""
    path = tmp_path / "bank.jsonl"
    assert append_anchor(_make_anchor("a"), path) is AppendOutcome.APPENDED
    assert append_anchor(_make_anchor("a"), path) is AppendOutcome.DUPLICATE_SKIPPED

    # A fresh, unrelated bank at the same path must accept a previously seen id.
    path.unlink()
    assert append_anchor(_make_anchor("a"), path) is AppendOutcome.APPENDED
    assert len(read_anchor_bank(path)) == 1


def test_write_anchor_bank_deduplicates_ids(tmp_path, caplog):
    """The bulk writer keeps one row per id and announces the drop."""
    path = tmp_path / "bank.jsonl"
    anchors = [_make_anchor("dup"), _make_anchor("dup"), _make_anchor("other")]
    with caplog.at_level("WARNING"):
        write_anchor_bank(anchors, path)
    records = read_anchor_bank(path)
    assert [r["id"] for r in records] == ["dup", "other"]
    assert count_unique_anchor_ids(path) == 2
    assert any("duplicate" in record.getMessage() for record in caplog.records)


# ── count_existing_anchors ──────────────────────────────────────────────────


def test_count_existing_anchors_empty(tmp_path):
    """count_existing_anchors returns 0 for non-existent file."""
    path = tmp_path / "nonexistent.jsonl"
    assert count_existing_anchors(path) == 0


def test_count_existing_anchors_with_data(tmp_path):
    """count_existing_anchors returns correct count for existing file."""
    path = tmp_path / "bank.jsonl"
    write_anchor_bank([_make_anchor(f"a{i}") for i in range(7)], path)
    assert count_existing_anchors(path) == 7


def test_count_existing_anchors_empty_file(tmp_path):
    """count_existing_anchors returns 0 for an empty file."""
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert count_existing_anchors(path) == 0


def test_count_existing_anchors_blank_lines(tmp_path):
    """count_existing_anchors skips blank lines."""
    path = tmp_path / "bank.jsonl"
    path.write_text("\n\n")
    assert count_existing_anchors(path) == 0


# ── build_manifest_from_records ─────────────────────────────────────────────


def test_build_manifest_from_records_basic(tmp_path):
    """build_manifest_from_records produces correct summary from raw records."""
    records = [
        {
            "id": "a",
            "anchor_meta": {"knowledge_domain": "math", "language": "English", "capability": "qa"},
        },
        {
            "id": "b",
            "anchor_meta": {"knowledge_domain": "math", "language": "简体中文", "capability": "qa"},
        },
        {
            "id": "c",
            "anchor_meta": {"knowledge_domain": "physics", "language": "English", "capability": "reasoning"},
        },
    ]
    manifest = build_manifest_from_records(records, tmp_path / "out")
    assert manifest["total_anchors"] == 3
    assert manifest["domains"] == {"math": 2, "physics": 1}
    assert manifest["languages"] == {"English": 2, "简体中文": 1}
    assert manifest["capabilities"] == {"qa": 2, "reasoning": 1}


def test_build_manifest_from_records_visual_domain(tmp_path):
    """build_manifest_from_records handles visual_domain for multimodal anchors."""
    records = [
        {
            "id": "v1",
            "anchor_meta": {"visual_domain": "general", "question_type": "description", "language": "English"},
        },
    ]
    manifest = build_manifest_from_records(records, tmp_path / "out")
    assert manifest["domains"] == {"general": 1}
    assert manifest["capabilities"] == {"description": 1}


# ── build_manifest ──────────────────────────────────────────────────────────


def test_build_manifest_counts(tmp_path):
    """build_manifest counts domains, languages, capabilities."""
    anchors = [
        _make_anchor("a", anchor_meta={"knowledge_domain": "math", "language": "English", "capability": "qa"}),
        _make_anchor("b", anchor_meta={"knowledge_domain": "math", "language": "简体中文", "capability": "qa"}),
        _make_anchor("c", anchor_meta={"knowledge_domain": "physics", "language": "English", "capability": "reasoning"}),
    ]
    manifest = build_manifest(anchors, tmp_path / "out")
    assert manifest["total_anchors"] == 3
    assert manifest["domains"] == {"math": 2, "physics": 1}
    assert manifest["languages"] == {"English": 2, "简体中文": 1}
    assert manifest["capabilities"] == {"qa": 2, "reasoning": 1}


def test_write_manifest(tmp_path):
    """write_manifest writes JSON."""
    path = tmp_path / "manifest.json"
    manifest = {"total_anchors": 5, "domains": {}}
    write_manifest(manifest, path)
    data = json.loads(path.read_text())
    assert data["total_anchors"] == 5


# ── Text Anchor prompts ────────────────────────────────────────────────────


def test_build_input_prompt_contains_meta():
    """build_input_prompt includes domain, capability, language."""
    meta = {
        "language": "简体中文",
        "knowledge_domain": "software_engineering",
        "capability": "coding",
        "conversation_type": "single_turn",
    }
    prompt = build_input_prompt(meta)
    assert "简体中文" in prompt
    assert "software_engineering" in prompt
    assert "coding" in prompt
    assert "single_turn" in prompt


def test_build_input_prompt_defaults():
    """build_input_prompt uses defaults for missing keys."""
    prompt = build_input_prompt({})
    assert "English" in prompt
    assert "general" in prompt


def test_build_target_prompt():
    """build_target_prompt returns a non-empty string."""
    prompt = build_target_prompt({})
    assert isinstance(prompt, str)
    assert len(prompt) > 0


# ── Pipeline (import-only) ──────────────────────────────────────────────────


def test_pipeline_imports():
    """Verify pipeline module is importable."""
    from ard.pipeline import run
    assert callable(run)