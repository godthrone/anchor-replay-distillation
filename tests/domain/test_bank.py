"""Tests for ARD — domain bank, text_anchor prompts, and pipeline."""

import json
from pathlib import Path

import pytest

from ard.domain.bank import (
    anchor_to_dict,
    append_anchor,
    build_manifest,
    build_manifest_from_records,
    count_existing_anchors,
    read_anchor_bank,
    write_anchor_bank,
    write_manifest,
)
from ard.domain.text_anchor import build_input_prompt, build_target_prompt
from ard.core.types import GeneratedAnchor, AnchorGenerationConfig


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
    a = _make_anchor("test_001", logprobs={"token_ids": [1, 2], "log_probs": [-0.1, -0.2]})
    d = anchor_to_dict(a)
    assert d["id"] == "test_001"
    assert d["source"] == "ard"
    assert d["messages"] == [{"role": "user", "content": "q"}]
    assert d["targets"][0]["id"] == "primary"
    assert d["targets"][0]["output"]["content"] == "answer"
    assert d["targets"][0]["output"]["logprobs"] == {"token_ids": [1, 2], "log_probs": [-0.1, -0.2]}
    assert d["anchor_meta"] == {"knowledge_domain": "math", "language": "English", "capability": "qa"}
    assert d["teacher_id"] == "target"


def test_anchor_to_dict_logprobs_none():
    """anchor_to_dict fills empty logprobs when None."""
    a = _make_anchor("a", logprobs=None)
    d = anchor_to_dict(a)
    assert d["targets"][0]["output"]["logprobs"] == {"token_ids": [], "log_probs": []}


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