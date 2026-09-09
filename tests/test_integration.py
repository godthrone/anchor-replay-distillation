"""Integration tests for ARD — anchor generation pipeline and image allocation.

Tests cover:
- ``generate_text_anchors`` pipeline (single-turn, multi-turn, image spec)
- ``allocate_images`` quota function (normal, degradation)
- ``AnchorSpec.__post_init__`` validation
"""

from __future__ import annotations

import random
import struct
import zlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ard.backends.api_client import ChatAPIClient
from ard.core.quota import allocate_images
from ard.core.types import AnchorSpec, GeneratedAnchor, TurnSpec
from ard.domain.text_anchor import generate_text_anchors

# ── Helpers ──────────────────────────────────────────────────────────────────


def _create_minimal_png(directory: Path, name: str = "test.png") -> Path:
    """Create a minimal 1×1 red PNG and return its path."""

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = b"\x00\xff\x00\x00"
    idat = _chunk(b"IDAT", zlib.compress(raw))
    iend = _chunk(b"IEND", b"")

    png_path = directory / name
    png_path.write_bytes(header + ihdr + idat + iend)
    return png_path


def _make_single_turn_spec(
    spec_id: str = "test_001", language: str = "English", domain: str = "geography"
) -> AnchorSpec:
    """Create a single-turn AnchorSpec with one final user turn."""
    turns = [
        TurnSpec(
            turn_index=0,
            role="user",
            generation_instruction="Ask a question about geography",
            is_final=True,
        ),
    ]
    return AnchorSpec(
        id=spec_id,
        anchor_meta={
            "language": language,
            "knowledge_domain": domain,
            "capability": "qa",
            "conversation_type": "single_turn",
        },
        turns=turns,
        input_generator_id="input-gen",
    )


def _make_multiturn_spec(spec_id: str = "multi_001") -> AnchorSpec:
    """Create a 3-turn AnchorSpec (user, assistant, user-is_final)."""
    turns = [
        TurnSpec(
            turn_index=0,
            role="user",
            generation_instruction="Start a conversation about AI",
        ),
        TurnSpec(
            turn_index=1,
            role="assistant",
            generation_instruction="Respond to the user",
        ),
        TurnSpec(
            turn_index=2,
            role="user",
            generation_instruction="Ask a follow-up question",
            is_final=True,
        ),
    ]
    return AnchorSpec(
        id=spec_id,
        anchor_meta={
            "language": "English",
            "knowledge_domain": "ai",
            "capability": "qa",
            "conversation_type": "multi_turn",
        },
        turns=turns,
        input_generator_id="input-gen",
    )


def _make_spec_with_n_turns(spec_id: str, num_turns: int) -> AnchorSpec:
    """Create an AnchorSpec with *num_turns* alternating user/assistant/.../user turns."""
    turns = []
    for i in range(num_turns):
        role = "user" if i % 2 == 0 else "assistant"
        is_final = i == num_turns - 1
        turns.append(
            TurnSpec(
                turn_index=i,
                role=role,
                generation_instruction=f"Turn {i}",
                is_final=is_final,
            )
        )
    return AnchorSpec(
        id=spec_id,
        anchor_meta={},
        turns=turns,
        input_generator_id=None,
    )


# ── generate_text_anchors tests ──────────────────────────────────────────────


class TestGenerateTextAnchors:
    """Integration tests for the ``generate_text_anchors`` pipeline."""

    def test_single_turn(self) -> None:
        """``generate_text_anchors`` produces one ``GeneratedAnchor`` for a single-turn spec."""
        spec = _make_single_turn_spec()

        mock_input = MagicMock(spec=ChatAPIClient)
        mock_input.chat.return_value = "What is the capital of France?"

        mock_target = MagicMock(spec=ChatAPIClient)
        mock_target.chat_with_logprobs.return_value = {
            "content": "Paris is the capital of France.",
            "logprobs": {
                "token_ids": [1, 2, 3],
                "log_probs": [-0.1, -0.2, -0.3],
            },
        }

        results = generate_text_anchors(
            specs=[spec],
            input_client=mock_input,
            target_client=mock_target,
            input_model_name="input-model",
            target_model_name="target-model",
            concurrency=1,
        )

        assert isinstance(results, list)
        assert len(results) == 1
        anchor = results[0]
        assert isinstance(anchor, GeneratedAnchor)
        # Single-turn: 1 user message (no intermediate assistant)
        assert len(anchor.messages) == 1
        assert anchor.messages[0]["role"] == "user"
        assert isinstance(anchor.messages[0]["content"], str)
        assert len(anchor.messages[0]["content"]) > 0
        assert anchor.target_answer == "Paris is the capital of France."
        assert len(anchor.target_answer) > 0
        assert anchor.logprobs is not None
        assert len(anchor.logprobs["token_ids"]) > 0
        assert len(anchor.logprobs["log_probs"]) > 0

    def test_multiturn(self) -> None:
        """``generate_text_anchors`` handles a multi-turn conversation producing 5 messages."""
        spec = _make_multiturn_spec()

        mock_input = MagicMock(spec=ChatAPIClient)
        mock_input.chat.side_effect = [
            "What is machine learning?",
            "Can you explain neural networks?",
            "How do transformers work?",
        ]

        mock_target = MagicMock(spec=ChatAPIClient)
        mock_target.chat.side_effect = [
            "Machine learning is a subset of AI.",
            "Neural networks are computational models inspired by the brain.",
        ]
        mock_target.chat_with_logprobs.return_value = {
            "content": "Transformers use self-attention mechanisms.",
            "logprobs": {
                "token_ids": [10, 20, 30],
                "log_probs": [-0.1, -0.2, -0.3],
            },
        }

        results = generate_text_anchors(
            specs=[spec],
            input_client=mock_input,
            target_client=mock_target,
            input_model_name="input-model",
            target_model_name="target-model",
            concurrency=1,
        )

        assert len(results) == 1
        anchor = results[0]
        # 3 turns (user, assistant, user-is_final) →
        #   5 messages: user, assistant, user, assistant, user
        assert len(anchor.messages) == 5
        roles = [m["role"] for m in anchor.messages]
        assert roles == ["user", "assistant", "user", "assistant", "user"]
        # Final turn has logprobs
        assert anchor.logprobs is not None
        assert anchor.target_answer == "Transformers use self-attention mechanisms."

    def test_with_image_spec(self, tmp_path: Path) -> None:
        """``generate_text_anchors`` produces list-format content for turns with images."""
        png_path = _create_minimal_png(tmp_path)

        turns = [
            TurnSpec(
                turn_index=0,
                role="user",
                generation_instruction="Describe this image",
                image_path=str(png_path),
                is_final=True,
            ),
        ]
        spec = AnchorSpec(
            id="img_001",
            anchor_meta={
                "language": "English",
                "knowledge_domain": "vision",
                "capability": "description",
                "conversation_type": "single_turn",
            },
            turns=turns,
            input_generator_id="input-gen",
        )

        mock_input = MagicMock(spec=ChatAPIClient)
        mock_input.chat.return_value = "What do you see in this image?"

        mock_target = MagicMock(spec=ChatAPIClient)
        mock_target.chat_with_logprobs.return_value = {
            "content": "I see a red pixel.",
            "logprobs": {"token_ids": [1], "log_probs": [-0.1]},
        }

        results = generate_text_anchors(
            specs=[spec],
            input_client=mock_input,
            target_client=mock_target,
            input_model_name="input-model",
            target_model_name="target-model",
            concurrency=1,
        )

        assert len(results) == 1
        anchor = results[0]
        assert len(anchor.messages) == 1
        assert anchor.messages[0]["role"] == "user"
        content = anchor.messages[0]["content"]
        # Content should be a list (multimodal format) when an image is present
        assert isinstance(content, list)
        has_image = any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in content
        )
        assert has_image, "content list should contain an image_url part"
        has_text = any(
            isinstance(part, dict) and part.get("type") == "text"
            for part in content
        )
        assert has_text, "content list should contain a text part"


# ── allocate_images tests ────────────────────────────────────────────────────


class TestAllocateImages:
    """Integration tests for the ``allocate_images`` quota function."""

    def test_normal_allocation(self) -> None:
        """``allocate_images`` assigns images to specs when the pool is sufficient."""
        specs = [_make_spec_with_n_turns(f"spec_{i}", 1) for i in range(3)]
        image_pool = ["/img/a.png", "/img/b.png", "/img/c.png"]
        rng = random.Random(42)

        result = allocate_images(specs, image_pool, max_turns_with_image=1, rng=rng)

        # Returns the same list, modified in-place
        assert result is specs
        images_assigned = sum(1 for s in specs if s.anchor_meta.get("has_image"))
        assert images_assigned > 0
        for s in specs:
            if s.anchor_meta.get("has_image"):
                assert s.turns[0].image_path is not None
                assert s.anchor_meta["image_count"] >= 1

    def test_degradation_small_pool(self) -> None:
        """``allocate_images`` cycles through a small pool — all specs get images."""
        specs = [_make_spec_with_n_turns(f"spec_{i}", 1) for i in range(5)]
        image_pool = ["/img/only.png"]  # only 1 image
        rng = random.Random(42)

        result = allocate_images(specs, image_pool, max_turns_with_image=1, rng=rng)

        assert result is specs
        images_assigned = sum(1 for s in specs if s.anchor_meta.get("has_image"))
        no_images = sum(1 for s in specs if not s.anchor_meta.get("has_image"))
        # All specs get images via cycle — the single image is reused
        assert images_assigned == len(specs), "All specs should get images via cycle"
        assert no_images == 0, "No specs should be text-only when image pool is non-empty"
        # All specs that got images should have image_path set
        for s in specs:
            if s.anchor_meta.get("has_image"):
                assert s.turns[0].image_path is not None


# ── AnchorSpec validation tests ──────────────────────────────────────────────


class TestAnchorSpecValidation:
    """Validation tests for ``AnchorSpec.__post_init__``."""

    def test_empty_turns_raises(self) -> None:
        """``AnchorSpec`` with empty turns raises ``ValueError``."""
        with pytest.raises(ValueError, match="must not be empty"):
            AnchorSpec(
                id="bad",
                anchor_meta={},
                turns=[],
                input_generator_id=None,
            )

    def test_first_not_user_raises(self) -> None:
        """``AnchorSpec`` whose first turn is not ``"user"`` raises ``ValueError``."""
        turns = [
            TurnSpec(turn_index=0, role="assistant", generation_instruction="A1"),
            TurnSpec(
                turn_index=1,
                role="user",
                generation_instruction="Q1",
                is_final=True,
            ),
        ]
        with pytest.raises(ValueError, match="First turn must be user"):
            AnchorSpec(
                id="bad",
                anchor_meta={},
                turns=turns,
                input_generator_id=None,
            )

    def test_last_not_user_raises(self) -> None:
        """``AnchorSpec`` whose last turn is not ``"user"`` raises ``ValueError``."""
        turns = [
            TurnSpec(turn_index=0, role="user", generation_instruction="Q1"),
            TurnSpec(
                turn_index=1,
                role="assistant",
                generation_instruction="A1",
                is_final=True,
            ),
        ]
        with pytest.raises(ValueError, match="Last turn must be user"):
            AnchorSpec(
                id="bad",
                anchor_meta={},
                turns=turns,
                input_generator_id=None,
            )

    def test_consecutive_same_role_raises(self) -> None:
        """``AnchorSpec`` rejects consecutive same-role turns."""
        turns = [
            TurnSpec(turn_index=0, role="user", generation_instruction="Q1"),
            TurnSpec(
                turn_index=1,
                role="user",
                generation_instruction="Q2",
                is_final=True,
            ),
        ]
        with pytest.raises(ValueError, match="same role"):
            AnchorSpec(
                id="bad",
                anchor_meta={},
                turns=turns,
                input_generator_id=None,
            )
