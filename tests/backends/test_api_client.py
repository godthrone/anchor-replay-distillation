"""Tests for ARD — backends API client and image encoding."""

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