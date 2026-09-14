"""Integration tests for ARD — anchor generation pipeline and image allocation.

Tests cover:
- ``generate_text_anchors`` pipeline (single-turn, multi-turn, image spec)
- ``allocate_images`` quota function (normal, degradation)
- ``AnchorSpec.__post_init__`` validation
"""

from __future__ import annotations

import json
import random
import struct
import zlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ard.backends.api_client import ARDTimeoutError, ChatAPIClient, ChatResponse
from ard.core.quota import allocate_images
from ard.core.sampler import generate_anchor_id
from ard.core.types import AnchorSpec, GeneratedAnchor, TurnSpec
from ard.domain.anchor_shape import message_shape_error
from ard.domain.bank import append_anchor, read_anchor_bank
from ard.domain.text_anchor import AnchorGenerationStats, generate_text_anchors

# ── Helpers ──────────────────────────────────────────────────────────────────


#: What a *non-final* target turn answers.  The final turn is the one that also
#: carries the teacher's reasoning, so it is scripted separately below.
_MID_TURN_REPLY = "an intermediate assistant reply"

#: The final target turn of every scripted target client: a real answer, plus the
#: reasoning trace that now lands in ``targets[0].output.reasoning``.
_FINAL_ANSWER = "the final target answer"
_FINAL_REASONING = "Step one: read the question. Step two: answer it."


def _final_target_response(
    content: str = _FINAL_ANSWER,
    reasoning: str | None = _FINAL_REASONING,
) -> ChatResponse:
    """A final-turn response: answer and reasoning as two separate fields."""
    return ChatResponse(content=content, reasoning=reasoning)


def _scripted_chat_client(
    content: str = _MID_TURN_REPLY,
    final: ChatResponse | None = None,
    *,
    total_calls: int = 2,
) -> MagicMock:
    """A ``ChatAPIClient`` double whose last call is the one that answers.

    ``chat()`` serves every assistant turn *and* the final answer, so the number
    of calls depends on the spec.  This double distinguishes the two roles by
    position: calls ``1..total_calls-1`` return *content* (intermediate assistant
    replies) and call *total_calls* returns *final* — the response that carries
    the teacher's reasoning.  Every spec exercised here has at most one
    intermediate assistant turn, so ``total_calls`` is 1 or 2.
    """
    client = MagicMock(spec=ChatAPIClient)
    chat_response = final if final is not None else _final_target_response()
    state = {"seen": 0}

    def _chat(messages: list[dict], temperature: float | None = None) -> ChatResponse:
        state["seen"] += 1
        if state["seen"] == total_calls:
            return chat_response
        return ChatResponse(content=content)

    client.chat.side_effect = _chat
    return client


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
        mock_input.chat.return_value = ChatResponse(content="What is the capital of France?")

        mock_target = _scripted_chat_client(
            total_calls=1,
            final=_final_target_response("Paris is the capital of France."),
        )

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
        # The teacher's reasoning is persisted as its own field, never merged
        # into the answer (§1.2 契约 2).
        assert anchor.reasoning == _FINAL_REASONING
        assert anchor.reasoning not in anchor.target_answer

    def test_multiturn(self) -> None:
        """A 3-turn spec produces ``UAU`` — one message per turn, ending with user.

        This test previously asserted ``["user", "assistant", "user",
        "assistant", "user"]`` (5 messages) and thereby *froze the bug*: the
        old loop never read ``turn.role``, so it generated a user message for
        the assistant turn too and the final user turn became the second
        assistant reply.  The correct shape has one message per turn and the
        final turn (which the target model answers) is a ``user`` turn.
        """
        spec = _make_multiturn_spec()

        mock_input = MagicMock(spec=ChatAPIClient)
        # Only the two *user* turns may ask the input generator for a message.
        mock_input.chat.side_effect = [
            ChatResponse(content="What is machine learning?"),
            ChatResponse(content="How do transformers work?"),
        ]

        mock_target = _scripted_chat_client(
            # The single *assistant* turn is answered by the target model.
            content="Machine learning is a subset of AI.",
            final=_final_target_response(
                "Transformers use self-attention mechanisms."
            ),
        )

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
        # 3 turns (user, assistant, user-is_final) → 3 messages: user, assistant, user
        assert len(anchor.messages) == 3
        roles = [m["role"] for m in anchor.messages]
        assert roles == ["user", "assistant", "user"]
        # Assistant turns must not be routed through the input generator.
        assert mock_input.chat.call_count == 2
        # The target model serves the assistant turn *and* the final answer.
        assert mock_target.chat.call_count == 2
        # The assistant message really is the target model's reply
        assert anchor.messages[1]["content"] == "Machine learning is a subset of AI."
        # Final turn carries the answer and the reasoning, as two fields
        assert anchor.target_answer == "Transformers use self-attention mechanisms."
        assert anchor.reasoning == _FINAL_REASONING

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
        mock_input.chat.return_value = ChatResponse(content="What do you see in this image?")

        mock_target = _scripted_chat_client(
            total_calls=1, final=_final_target_response("I see a red pixel.")
        )

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


class TestRoleDrivenGeneration:
    """Role-driven generation: one message per turn, assistants are target-side.

    Regression suite for the ``UAUAU`` defect (every turn was routed through
    the input generator, so assistant turns became user messages) and for the
    ``UAUU`` defect (a timed-out turn was skipped with ``continue``, leaving
    two consecutive user messages).
    """

    @pytest.mark.parametrize(
        "num_turns, expected_shape",
        [
            (1, "U"),
            (3, "UAU"),
            (5, "UAUAU"),
            (9, "UAUAUAUAU"),
        ],
    )
    def test_multiturn_shape_is_user_first_user_last(
        self, num_turns: int, expected_shape: str
    ) -> None:
        """Multi-turn anchors are ``U``/``UAU``/``UAUAU`` — never ``UAUAU`` for 3 turns."""
        spec = _make_spec_with_n_turns(f"shape_{num_turns}", num_turns)

        mock_input = MagicMock(spec=ChatAPIClient)
        mock_input.chat.return_value = ChatResponse(content="a user question")
        mock_target = _scripted_chat_client(
            content="an assistant reply",
            total_calls=2 if num_turns >= 3 else 1,
        )

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
        assert num_turns == len(spec.turns)
        assert len(anchor.messages) == num_turns
        shape = "".join("U" if m["role"] == "user" else "A" for m in anchor.messages)
        assert shape == expected_shape
        assert anchor.messages[0]["role"] == "user"
        assert anchor.messages[-1]["role"] == "user"
        # The exit-boundary contract, checked with the same predicate the bank uses.
        assert message_shape_error(anchor.messages) is None

    def test_request_counts_for_three_turns(self) -> None:
        """A 3-turn anchor costs 2 input calls + 1 assistant + 1 final-answer call.

        The old loop spent 3 input calls and 2 target calls for the same spec in
        the wrong places: one wasted request per anchor plus a structurally
        broken conversation.
        """
        spec = _make_spec_with_n_turns("counts_3", 3)

        mock_input = MagicMock(spec=ChatAPIClient)
        mock_input.chat.side_effect = [
            ChatResponse(content="question one"), ChatResponse(content="question two")
        ]
        mock_target = _scripted_chat_client(
            content="assistant reply", final=_final_target_response("final target answer")
        )

        results = generate_text_anchors(
            specs=[spec],
            input_client=mock_input,
            target_client=mock_target,
            input_model_name="input-model",
            target_model_name="target-model",
            concurrency=1,
        )

        assert len(results) == 1
        assert mock_input.chat.call_count == 2, "one input call per user turn"
        # One target call per assistant turn + one for the final answer.
        assert mock_target.chat.call_count == 2
        assert [m["role"] for m in results[0].messages] == ["user", "assistant", "user"]

    def test_assistant_turn_never_calls_input_client(self) -> None:
        """The input generator is only asked to impersonate the *user*."""
        spec = _make_spec_with_n_turns("no_input_for_assistant", 5)

        mock_input = MagicMock(spec=ChatAPIClient)
        mock_input.chat.return_value = ChatResponse(content="a user question")
        mock_target = _scripted_chat_client(
            content="an assistant reply", final=_final_target_response("final target answer")
        )

        generate_text_anchors(
            specs=[spec],
            input_client=mock_input,
            target_client=mock_target,
            input_model_name="input-model",
            target_model_name="target-model",
            concurrency=1,
        )

        # user turns: 3 → input calls: 3 (and no more)
        user_turns = sum(1 for t in spec.turns if t.role == "user")
        assistant_turns = sum(1 for t in spec.turns if t.role == "assistant")
        assert mock_input.chat.call_count == user_turns == 3
        # 2 assistant turns + the final answer.
        assert mock_target.chat.call_count == assistant_turns + 1 == 3

    def test_timeout_on_assistant_turn_discards_anchor(self) -> None:
        """A timed-out turn abandons the anchor instead of leaving ``UAUU``."""
        spec = _make_spec_with_n_turns("timeout_assistant", 3)

        mock_input = MagicMock(spec=ChatAPIClient)
        mock_input.chat.side_effect = [
            ChatResponse(content="question one"), ChatResponse(content="question two")
        ]
        mock_target = MagicMock(spec=ChatAPIClient)
        mock_target.chat.side_effect = ARDTimeoutError("timed out")

        results = generate_text_anchors(
            specs=[spec],
            input_client=mock_input,
            target_client=mock_target,
            input_model_name="input-model",
            target_model_name="target-model",
            concurrency=1,
        )

        # No anchor at all — never an anchor whose history is UAUU.
        assert results == []

    def test_timeout_on_final_user_turn_discards_anchor(self) -> None:
        """A timeout on the final turn discards the anchor."""
        spec = _make_spec_with_n_turns("timeout_final", 3)

        mock_input = MagicMock(spec=ChatAPIClient)
        mock_input.chat.side_effect = [
            ChatResponse(content="question one"), ARDTimeoutError("timed out")
        ]
        mock_target = _scripted_chat_client(
            content="assistant reply", final=_final_target_response("final target answer")
        )

        results = generate_text_anchors(
            specs=[spec],
            input_client=mock_input,
            target_client=mock_target,
            input_model_name="input-model",
            target_model_name="target-model",
            concurrency=1,
        )

        assert results == []

    def test_empty_user_message_discards_anchor(self) -> None:
        """An empty user message abandons the anchor (it would shift the history)."""
        spec = _make_spec_with_n_turns("empty_user", 3)

        mock_input = MagicMock(spec=ChatAPIClient)
        mock_input.chat.return_value = ChatResponse(content="   ")
        mock_target = _scripted_chat_client(
            content="assistant reply", final=_final_target_response("final target answer")
        )

        results = generate_text_anchors(
            specs=[spec],
            input_client=mock_input,
            target_client=mock_target,
            input_model_name="input-model",
            target_model_name="target-model",
            concurrency=1,
        )

        assert results == []

    def test_image_paths_stay_aligned_with_their_turns(self, tmp_path: Path) -> None:
        """``_convert_images_to_paths`` rewrites the image of *its own* turn.

        The conversion pairs message *i* with ``spec.turns[i]``.  That pairing
        only holds because the loop now appends exactly one message per turn;
        with two image-bearing user turns, a structural mismatch would show up
        as the second image path landing on the wrong message.
        """
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        first = _create_minimal_png(images_dir, "first.png")
        second = _create_minimal_png(images_dir, "second.png")

        turns = [
            TurnSpec(
                turn_index=0,
                role="user",
                generation_instruction="Describe this image",
                image_path=str(first),
            ),
            TurnSpec(
                turn_index=1,
                role="assistant",
                generation_instruction="Respond",
            ),
            TurnSpec(
                turn_index=2,
                role="user",
                generation_instruction="And this one?",
                image_path=str(second),
                is_final=True,
            ),
        ]
        spec = AnchorSpec(
            id="img_alignment",
            anchor_meta={
                "language": "English",
                "knowledge_domain": "vision",
                "capability": "description",
                "conversation_type": "multi_turn",
            },
            turns=turns,
            input_generator_id="input-gen",
        )

        mock_input = MagicMock(spec=ChatAPIClient)
        mock_input.chat.side_effect = [
            ChatResponse(content="What is in this image?"),
            ChatResponse(content="What about this one?"),
        ]
        mock_target = _scripted_chat_client(
            content="A red pixel.", final=_final_target_response("Also a red pixel.")
        )

        results = generate_text_anchors(
            specs=[spec],
            input_client=mock_input,
            target_client=mock_target,
            input_model_name="input-model",
            target_model_name="target-model",
            concurrency=1,
        )

        assert len(results) == 1
        messages = results[0].messages
        assert len(messages) == 3
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]

        def image_parts(msg: dict) -> list[str]:
            content = msg["content"]
            if not isinstance(content, list):
                return []
            return [
                part["image"]
                for part in content
                if isinstance(part, dict) and part.get("type") == "image"
            ]

        # Turn 0 carries the first image, turn 2 the second; the assistant turn
        # carries none.  Absolute paths are rewritten to images/<name>.
        assert image_parts(messages[0]) == ["images/first.png"]
        assert image_parts(messages[1]) == []
        assert image_parts(messages[2]) == ["images/second.png"]
        # No base64 payload survives into the output format.
        assert "image_url" not in json.dumps(messages)

    def test_output_path_receives_only_valid_shapes(self, tmp_path: Path) -> None:
        """Shapes written to the bank satisfy the conversation contract."""
        spec = _make_spec_with_n_turns("banked", 3)
        output_path = tmp_path / "anchor_bank.jsonl"

        mock_input = MagicMock(spec=ChatAPIClient)
        mock_input.chat.return_value = ChatResponse(content="a user question")
        mock_target = _scripted_chat_client(
            content="an assistant reply", final=_final_target_response("final target answer")
        )

        generate_text_anchors(
            specs=[spec],
            input_client=mock_input,
            target_client=mock_target,
            input_model_name="input-model",
            target_model_name="target-model",
            concurrency=1,
            output_path=output_path,
        )

        records = read_anchor_bank(output_path)
        assert len(records) == 1
        assert [m["role"] for m in records[0]["messages"]] == ["user", "assistant", "user"]


    def test_duplicate_ids_across_specs_are_banked_once(self, tmp_path: Path) -> None:
        """Anchors whose metadata collides share an id and must not double the bank.

        Anchor ids are derived from 4-dimensional metadata
        (``generate_anchor_id``), so two genuinely different specs can carry the
        same id.  A historical run produced 70 bank lines for 55 real anchors;
        the streaming writer must now keep one row per id and say so.
        """
        output_path = tmp_path / "anchor_bank.jsonl"
        colliding_meta = {
            "language": "English",
            "knowledge_domain": "math",
            "capability": "qa",
            "conversation_type": "single_turn",
        }
        anchor_id = generate_anchor_id(colliding_meta)
        specs = [
            AnchorSpec(
                id=anchor_id,
                anchor_meta=dict(colliding_meta),
                turns=[
                    TurnSpec(
                        turn_index=0,
                        role="user",
                        generation_instruction="Ask something",
                        is_final=True,
                    )
                ],
                input_generator_id=None,
            )
            for _ in range(3)
        ]

        mock_input = MagicMock(spec=ChatAPIClient)
        mock_input.chat.return_value = ChatResponse(content="a user question")
        mock_target = _scripted_chat_client(total_calls=1)

        results = generate_text_anchors(
            specs=specs,
            input_client=mock_input,
            target_client=mock_target,
            input_model_name="input-model",
            target_model_name="target-model",
            concurrency=1,
            output_path=output_path,
        )

        # All three anchors were generated...
        assert len(results) == 3
        # ...but the bank holds exactly one row for that id.
        records = read_anchor_bank(output_path)
        assert [r["id"] for r in records] == [anchor_id]
        assert len(records) == len({r["id"] for r in records})


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


# ── pipeline → manifest wiring ───────────────────────────────────────────────


def _toml_dump(data: dict) -> str:
    """Minimal TOML writer for the flat/nested-scalar config used in these tests."""

    def _value(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return f'"{value}"'
        return str(value)

    lines: list[str] = []
    for section, body in data.items():
        if isinstance(body, dict):
            lines.append(f"[{section}]")
            lines.extend(
                f"{key} = {_value(value)}"
                for key, value in body.items()
                if not isinstance(value, dict)
            )
            lines.append("")
        else:
            lines.append(f"{section} = {_value(body)}")
    return "\n".join(lines)


def _scripted_generate_text_anchors(**kwargs: object) -> list[GeneratedAnchor]:
    """Stand-in for the real generator: reports 1 written, 1 duplicate, 2 abandons.

    The real generator is exercised in
    ``tests/domain/test_text_anchor_backpressure.py``; what is under test here is
    the *plumbing* — that whatever it reports reaches ``manifest.json``.  The
    report is therefore scripted, and the only real call is
    :func:`append_anchor`, which is exactly what the counters describe.
    """
    stats = kwargs["stats"]
    output_path = kwargs["output_path"]
    assert isinstance(stats, AnchorGenerationStats)
    assert isinstance(output_path, Path)
    anchor = GeneratedAnchor(
        id="pipeline_wiring",
        messages=[{"role": "user", "content": "q"}],
        target_answer="an answer long enough",
        target_model="target-model",
        input_generator_model="input-model",
        anchor_meta={"language": "English", "knowledge_domain": "geography"},
        reasoning="the plumbing test's reasoning trace",
    )
    # Offered twice on purpose: the second offer must hit the id gate.
    append_anchor(anchor, output_path)
    append_anchor(anchor, output_path)
    stats.requested = 3
    stats.succeeded = 1
    stats.written = 1
    stats.duplicate_ids = 1
    stats.abandoned_total = 2
    stats.abandoned_by_reason = {"timeout": 2}
    stats.backpressure_events = 1
    return [anchor]


class TestManifestGenerationReport:
    """``pipeline.run()`` must fold the generation counters into ``manifest.json``.

    The counters existed after WP-F2/F3 but nobody read them, so a run whose
    anchors were all dropped still produced a healthy-looking manifest.  These
    tests freeze the wiring, not the counter implementation.
    """

    def test_manifest_reports_written_and_abandoned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tomllib

        from ard.config import load_config
        from ard.pipeline import run

        output_dir = tmp_path / "out"
        config_path = tmp_path / "config.toml"
        base = tomllib.loads(
            (Path(__file__).resolve().parents[1] / "configs" / "config.toml").read_text(
                encoding="utf-8"
            )
        )
        base["generation"]["target_count"] = 3
        base["generation"]["concurrency"] = 1
        base["output"]["directory"] = str(output_dir)
        # API endpoints are required to construct the clients; no call reaches them.
        for section in ("input_generator", "target_model"):
            base[section]["api_base"] = "http://127.0.0.1:1/v1"
            base[section]["model_name"] = f"{section}-model"
            base[section]["api_key"] = "unused"
        config_path.write_text(_toml_dump(base), encoding="utf-8")

        monkeypatch.setattr(
            "ard.pipeline.generate_text_anchors", _scripted_generate_text_anchors
        )

        result_dir = run(load_config(config_path))
        manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))

        assert manifest["total_anchors"] == 1, "the bank holds exactly one record"
        counters = manifest["generation"]["counters"]
        assert counters["requested"] == 3
        assert counters["succeeded"] == 1
        assert counters["written"] == 1
        assert counters["duplicate_ids"] == 1
        assert counters["backpressure_events"] == 1
        # Zero-valued counters are omitted rather than published as 0, so a
        # clean aspect of the run adds no noise (see docs/architecture.md §8.2).
        assert "rejected_invalid_shape" not in counters
        assert counters["abandoned_by_reason"] == {"timeout": 2}
        # The pre-existing fields still describe the bank itself.
        assert manifest["domains"] == {"geography": 1}
        assert manifest["output_dir"] == str(output_dir)
        assert manifest["config"]["generation"]["target_count"] == 3

    def test_clean_run_publishes_no_generation_noise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run where nothing was dropped gains no ``generation`` field."""
        import tomllib

        from ard.config import load_config
        from ard.pipeline import run

        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True)
        bank = output_dir / "anchor_bank.jsonl"
        record = {
            "id": "already_here",
            "source": "ard",
            "messages": [{"role": "user", "content": "q"}],
            "targets": [{"id": "primary", "output": {"content": "a", "reasoning": None}}],
            "anchor_meta": {"language": "English", "knowledge_domain": "math"},
            "teacher_id": "target-model",
        }
        bank.write_text(json.dumps(record) + "\n", encoding="utf-8")

        config_path = tmp_path / "config.toml"
        base = tomllib.loads(
            (Path(__file__).resolve().parents[1] / "configs" / "config.toml").read_text(
                encoding="utf-8"
            )
        )
        base["generation"]["target_count"] = 1  # already satisfied → resume path
        base["output"]["directory"] = str(output_dir)
        config_path.write_text(_toml_dump(base), encoding="utf-8")

        monkeypatch.setattr(
            "ard.pipeline.generate_text_anchors",
            lambda **kwargs: pytest.fail("no generation must run on the resume path"),
        )

        result_dir = run(load_config(config_path))
        manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["total_anchors"] == 1
        assert manifest.get("generation") is None, (
            "a missing generation report means 'not measured', never 'all zero'"
        )

    def test_reasoning_failure_counters_reach_the_manifest(self) -> None:
        """The reasoning/empty-content counters are published under ``failures``.

        The counters that used to be merged from two families are now one, so
        what matters is that the surviving family still reaches the manifest
        unchanged — a run that lost answers to thinking must not look healthy.
        """
        from ard.domain.bank import with_generation_report

        manifest: dict = {}
        with_generation_report(
            manifest,
            stats={},
            failures={"responses": 5, "empty_content": 2, "truncated_empty": 2},
        )
        assert manifest["generation"]["failures"] == {
            "responses": 5,
            "empty_content": 2,
            "truncated_empty": 2,
        }
