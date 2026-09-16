"""Tests for the image-part rewrite in the text anchor pipeline (v3.0.0 B3).

Covers the defect where ``_convert_images_to_paths`` paired message *i* with
``spec.turns[i]`` and therefore skipped the rewrite whenever a message existed
that is not a turn — the image then stayed in the API's inline ``image_url``
(base64) form and reached the bank that way.

Scope note: the pipeline as committed here does **not** yet emit a leading
``system`` message, so the offset is exercised by calling
``_convert_images_to_paths`` with the message list it will receive once a system
message exists, rather than through ``generate_text_anchors``.  The function is
called for real — nothing is patched or stubbed — so these assertions are about
the production code path, not a manufactured shape.
"""

from __future__ import annotations

import json
from pathlib import Path

from ard.core.types import AnchorSpec, TurnSpec
from ard.domain.text_anchor import _convert_images_to_paths

#: A minimal base64 data URI, i.e. the inline form the API call needs.
INLINE_IMAGE_URL = "data:image/png;base64,iVBORw0KGgo="


def _image_spec(image_path: str = "/tmp/ard-probe/images/probe.png") -> AnchorSpec:
    """A single-turn spec whose only turn owns an image."""
    return AnchorSpec(
        id="img_001",
        anchor_meta={
            "language": "English",
            "knowledge_domain": "vision",
            "capability": "qa",
            "conversation_type": "single_turn",
        },
        turns=[
            TurnSpec(
                turn_index=0,
                role="user",
                generation_instruction="Describe this image",
                image_path=image_path,
                is_final=True,
            )
        ],
        input_generator_id="input-gen",
    )


def _inline_image_message() -> dict:
    """One conversation message carrying the inline image form."""
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": INLINE_IMAGE_URL}},
            {"type": "text", "text": "What do you see?"},
        ],
    }


def test_inline_image_is_rewritten_to_a_path() -> None:
    """The base case: the image owner is the first message, so it is rewritten."""
    converted = _convert_images_to_paths([_inline_image_message()], _image_spec())

    assert converted[0]["content"][0] == {"type": "image", "image": "images/probe.png"}
    assert "base64" not in json.dumps(converted)


def test_leading_system_message_does_not_hide_the_image_owner() -> None:
    """A leading non-turn message must not shift the message/turn pairing.

    With the pairing read as ``spec.turns[msg_idx]``, ``messages[1]`` — the
    image-owning turn 0 — is looked up as ``turns[1]``, which does not exist
    here, so the rewrite is skipped and the base64 travels on to the bank.  The
    offset has to be derived from the message list instead of assumed.

    A ``system`` message is not a turn, and the shape contract allows it only at
    position 0, so it is exactly one message of prefix.
    """
    messages = [{"role": "system", "content": "You are a meticulous debugger."}]
    messages.append(_inline_image_message())

    converted = _convert_images_to_paths(messages, _image_spec())

    # The system message passes through untouched and keeps position 0.
    assert converted[0] == {"role": "system", "content": "You are a meticulous debugger."}
    # The image-owning conversation message is still paired with turns[0].
    assert converted[1]["content"][0] == {"type": "image", "image": "images/probe.png"}
    assert "base64" not in json.dumps(converted)
    assert "image_url" not in json.dumps(converted)


def test_messages_beyond_the_turn_list_are_carried_over() -> None:
    """The rewrite is length-preserving and never silently drops history."""
    messages = [_inline_image_message(), {"role": "assistant", "content": "A pixel."}]

    converted = _convert_images_to_paths(messages, _image_spec())

    assert len(converted) == 2
    assert converted[1] == {"role": "assistant", "content": "A pixel."}
    # ``converted`` is a fresh list of shallow copies: the caller's input is not
    # mutated in place, which is what lets ``_generate_one_anchor`` compare the
    # pre- and post-conversion shapes.
    assert messages[0]["content"][0]["type"] == "image_url"


def test_an_image_path_outside_an_images_directory_is_kept() -> None:
    """A path with no ``/images/`` segment cannot be made relative; keep it."""
    spec = _image_spec(image_path=str(Path("/tmp/elsewhere/probe.png")))

    converted = _convert_images_to_paths([_inline_image_message()], spec)

    assert converted[0]["content"][0] == {"type": "image", "image": "/tmp/elsewhere/probe.png"}
