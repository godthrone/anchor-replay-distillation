"""Tests for ARD v2 — backends, domain bank, text_anchor, pipeline."""

import json
from pathlib import Path

import pytest

from ard.backends.api_client import (
    ChatAPIClient,
    ChatAPIConfig,
    ChatRequest,
    ChatResult,
    ChatResultWithLogprobs,
    encode_image_to_base64,
)
from ard.domain.bank import (
    write_anchor_bank,
    read_anchor_bank,
    build_manifest,
    write_manifest,
)
from ard.domain.text_anchor import build_input_prompt, build_target_prompt
from ard.core.types import Anchor, AnchorGenerationConfig


# ── API Client types ────────────────────────────────────────────────────────


def test_chat_api_config_defaults():
    """ChatAPIConfig has sensible defaults."""
    c = ChatAPIConfig(api_base="https://api.example.com", model_name="m", api_key="k")
    assert c.temperature == 0.7
    assert c.max_tokens is None
    assert c.timeout == 60.0
    assert c.max_retries == 2
    assert c.chat_completions_url == "https://api.example.com/chat/completions"


def test_chat_api_config_url_no_trailing_slash():
    """chat_completions_url handles trailing slash correctly."""
    c = ChatAPIConfig(api_base="https://api.example.com/v1/", model_name="m", api_key="k")
    assert c.chat_completions_url == "https://api.example.com/v1/chat/completions"


def test_chat_request():
    """ChatRequest stores messages and temperature."""
    req = ChatRequest(messages=[{"role": "user", "content": "hi"}], temperature=0.5)
    assert req.messages == [{"role": "user", "content": "hi"}]
    assert req.temperature == 0.5


def test_chat_result_success():
    """ChatResult stores success result."""
    r = ChatResult(content="hello", success=True)
    assert r.content == "hello"
    assert r.success is True
    assert r.error is None


def test_chat_result_failure():
    """ChatResult stores failure result."""
    r = ChatResult(content="", success=False, error="timeout")
    assert r.success is False
    assert r.error == "timeout"


def test_chat_result_with_logprobs():
    """ChatResultWithLogprobs stores logprobs."""
    lp = {"token_ids": [1, 2], "log_probs": [-0.1, -0.2]}
    r = ChatResultWithLogprobs(content="hi", success=True, logprobs=lp)
    assert r.logprobs == lp
    assert r.content == "hi"


def test_chat_client_creation():
    """ChatAPIClient can be instantiated."""
    config = ChatAPIConfig(api_base="https://api.example.com", model_name="m", api_key="k")
    client = ChatAPIClient(config)
    assert client._config == config


# ── Image encoding ──────────────────────────────────────────────────────────


def test_encode_image_to_base64_png(tmp_path):
    """encode_image_to_base64 returns data URI for PNG."""
    # Create a minimal valid PNG (1x1 pixel)
    import struct
    import zlib

    def create_png(width, height):
        def chunk(chunk_type, data):
            c = chunk_type + data
            crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            return struct.pack(">I", len(data)) + c + crc

        header = b"\x89PNG\r\n\x1a\n"
        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        raw = b""
        for y in range(height):
            raw += b"\x00" + b"\xff\x00\x00" * width
        idat = chunk(b"IDAT", zlib.compress(raw))
        iend = chunk(b"IEND", b"")
        return header + ihdr + idat + iend

    png_path = tmp_path / "test.png"
    png_path.write_bytes(create_png(1, 1))
    result = encode_image_to_base64(png_path)
    assert result.startswith("data:image/png;base64,")


def test_encode_image_to_base64_file_not_found():
    """encode_image_to_base64 raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        encode_image_to_base64("nonexistent.png")


def test_encode_image_to_base64_unsupported_format(tmp_path):
    """encode_image_to_base64 raises ValueError for unsupported format."""
    bad = tmp_path / "test.xyz"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Unsupported image format"):
        encode_image_to_base64(bad)


# ── Bank ────────────────────────────────────────────────────────────────────


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
    return Anchor(**defaults)


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


# ── Config types ────────────────────────────────────────────────────────────


def test_config_section_types():
    """Verify config section types are importable and constructible."""
    from ard.config import (
        ARDConfig,
        InputGeneratorConfig,
        TargetModelConfig,
        GenerationConfig,
        OntologyConfig,
        OutputConfig,
    )

    ig = InputGeneratorConfig(api_base="https://api.example.com", model_name="m", api_key="k")
    assert ig.temperature == 0.8

    t = TargetModelConfig(api_base="https://api.example.com", model_name="m", api_key="k")
    assert t.temperature == 0.0

    g = GenerationConfig()
    assert g.target_count == 100
    assert g.seed == 42

    o = OntologyConfig()
    assert o.path == "configs/anchor_ontology.json"

    out = OutputConfig()
    assert out.directory is None
    assert out.overwrite is False


def test_ard_config_full():
    """ARDConfig composes all sections."""
    from ard.config import ARDConfig

    c = ARDConfig()
    assert c.generation.target_count == 100
    assert c.output.overwrite is False
    assert c.ontology.path == "configs/anchor_ontology.json"