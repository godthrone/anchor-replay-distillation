"""Tests for multi-turn anchor types, quota, and config boundaries."""

import pytest

from ard.core.types import (
    AnchorSpec,
    GeneratedAnchor,
    TurnSpec,
)

# ── TurnSpec ────────────────────────────────────────────────────────────────


def test_turn_spec_construction():
    """TurnSpec can be constructed with required fields."""
    t = TurnSpec(turn_index=0, role="user", generation_instruction="Ask a question")
    assert t.turn_index == 0
    assert t.role == "user"
    assert t.generation_instruction == "Ask a question"
    assert t.image_path is None
    assert t.is_final is False


def test_turn_spec_with_image():
    """TurnSpec accepts optional image_path."""
    t = TurnSpec(
        turn_index=1,
        role="user",
        generation_instruction="Describe this image",
        image_path="/path/to/img.png",
        is_final=True,
    )
    assert t.image_path == "/path/to/img.png"
    assert t.is_final is True


# ── AnchorSpec ──────────────────────────────────────────────────────────────


def test_anchor_spec_valid_construction():
    """AnchorSpec constructs with valid turns."""
    turns = [
        TurnSpec(turn_index=0, role="user", generation_instruction="Q1", is_final=True),
    ]
    spec = AnchorSpec(
        id="test_001",
        anchor_meta={"lang": "en"},
        turns=turns,
        input_generator_id="gpt-4",
    )
    assert spec.id == "test_001"
    assert len(spec.turns) == 1
    assert spec.turns[0].role == "user"


def test_anchor_spec_multi_turn():
    """AnchorSpec supports multi-turn conversations (user/assistant/.../user)."""
    turns = [
        TurnSpec(turn_index=0, role="user", generation_instruction="Q1"),
        TurnSpec(turn_index=1, role="assistant", generation_instruction="A1"),
        TurnSpec(turn_index=2, role="user", generation_instruction="Q2", is_final=True),
    ]
    spec = AnchorSpec(
        id="multi_001",
        anchor_meta={},
        turns=turns,
        input_generator_id=None,
    )
    assert len(spec.turns) == 3


def test_anchor_spec_empty_turns_raises():
    """AnchorSpec with empty turns raises ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        AnchorSpec(
            id="bad",
            anchor_meta={},
            turns=[],
            input_generator_id=None,
        )


def test_anchor_spec_first_not_user_raises():
    """AnchorSpec whose first turn is not user raises ValueError."""
    turns = [
        TurnSpec(turn_index=0, role="assistant", generation_instruction="A1"),
        TurnSpec(turn_index=1, role="user", generation_instruction="Q1", is_final=True),
    ]
    with pytest.raises(ValueError, match="First turn must be user"):
        AnchorSpec(
            id="bad",
            anchor_meta={},
            turns=turns,
            input_generator_id=None,
        )


def test_anchor_spec_last_not_user_raises():
    """AnchorSpec whose last turn is not user raises ValueError."""
    turns = [
        TurnSpec(turn_index=0, role="user", generation_instruction="Q1"),
        TurnSpec(turn_index=1, role="assistant", generation_instruction="A1", is_final=True),
    ]
    with pytest.raises(ValueError, match="Last turn must be user"):
        AnchorSpec(
            id="bad",
            anchor_meta={},
            turns=turns,
            input_generator_id=None,
        )


def test_anchor_spec_role_alternation():
    """AnchorSpec rejects consecutive same-role turns."""
    turns = [
        TurnSpec(turn_index=0, role="user", generation_instruction="Q1"),
        TurnSpec(turn_index=1, role="user", generation_instruction="Q2", is_final=True),
    ]
    with pytest.raises(ValueError, match="same role"):
        AnchorSpec(
            id="bad",
            anchor_meta={},
            turns=turns,
            input_generator_id=None,
        )


# ── GeneratedAnchor ─────────────────────────────────────────────────────────


def test_generated_anchor_construction():
    """GeneratedAnchor constructs with required fields."""
    ga = GeneratedAnchor(
        id="gen_001",
        messages=[{"role": "user", "content": "hello"}],
        target_answer="world",
        target_model="target-model",
        input_generator_model="input-gen",
        anchor_meta={"lang": "en"},
    )
    assert ga.id == "gen_001"
    assert ga.target_answer == "world"
    assert ga.reasoning is None


def test_generated_anchor_with_reasoning():
    """GeneratedAnchor stores the reasoning trace when provided."""
    ga = GeneratedAnchor(
        id="g",
        messages=[],
        target_answer="x",
        target_model="m",
        input_generator_model="m",
        anchor_meta={},
        reasoning="thinking about x",
    )
    assert ga.reasoning == "thinking about x"


# ── Config validation ───────────────────────────────────────────────────────


def test_config_has_no_system_persona_field():
    """`system_persona` is gone — the ontology samples the system prompt (B3).

    Kept as a test rather than as a comment because the field was reachable
    from three layers (config → generation config → generator) and any of them
    could reintroduce it: with ``extra="forbid"`` a config file that still
    carries the key now fails loudly instead of silently doing nothing.
    """
    from pydantic import ValidationError

    from ard.config import GenerationConfig

    with pytest.raises(ValidationError):
        GenerationConfig(system_persona="none")  # type: ignore[call-arg]
    assert "system_persona" not in GenerationConfig.model_fields


def test_config_has_no_turn_knobs():
    """Turn counts come from the ontology, so the old knobs are refused.

    ``max_turns`` / ``max_turns_with_image`` used to drive the plan's turn
    distribution; the v4 rule reads each entry's turns from
    ``conversation_type.value_attributes.turns`` instead.  ``extra="forbid"`` is
    what makes a config file that still carries them fail loudly rather than
    silently do nothing (§18.1 不留负债).
    """
    from pydantic import ValidationError

    from ard.config import GenerationConfig

    fields = set(GenerationConfig.model_fields)
    assert "max_turns" not in fields
    assert "max_turns_with_image" not in fields
    with pytest.raises(ValidationError):
        GenerationConfig(max_turns=3)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        GenerationConfig(max_turns_with_image=2)  # type: ignore[call-arg]


def test_config_system_prompt_is_not_configurable():
    """No system-prompt *config* key exists — it is a sampling dimension (B3).

    The old ``system_persona`` switch only ever had four fixed values and
    reached nothing, so it was deleted rather than migrated (§18.1 不留负债).
    The behaviour it was supposed to produce is covered by
    ``tests/core/test_system_prompt.py`` (dimension) and
    ``tests/domain/test_system_prompt_anchor.py`` (generation + persistence).
    """
    from ard.config import GenerationConfig

    config_fields = set(GenerationConfig.model_fields)
    assert "system_persona" not in config_fields
    assert not any("system_prompt" in name for name in config_fields)
