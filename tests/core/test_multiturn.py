"""Tests for multi-turn anchor types, quota, and sampling."""

import json
import random
from pathlib import Path

import pytest

from ard.core.types import (
    AnchorSpec,
    AnchorGenerationConfig,
    GeneratedAnchor,
    TurnSpec,
)
from ard.core.quota import compute_turn_distribution
from ard.core.sampler import sample_anchors, _get_leaf_conversation_types
from ard.core.ontology import load_ontology


# ── Helpers ─────────────────────────────────────────────────────────────────


def _minimal_ontology_payload() -> dict:
    return {
        "languages": ["English"],
        "knowledge_domains": {
            "domain_a": {"topic": ["alpha"]},
            "domain_b": {"topic": ["beta"]},
            "domain_c": {"topic": ["gamma"]},
        },
        "capabilities": {
            "knowledge_response": ["qa"],
            "reasoning": ["reasoning"],
            "coding_and_data": ["coding"],
        },
        "conversation_types": {
            "single_turn": ["single_turn"],
            "clarification": ["clarification_2_turn"],
            "troubleshooting": ["troubleshooting_3_turn"],
        },
        "language_features": {
            "style": ["concise"],
            "format": ["paragraph"],
            "difficulty": ["basic"],
            "context_length": ["short"],
            "noise": ["clean"],
            "answer_expectation": ["direct_answer"],
        },
    }


def _large_ontology_payload() -> dict:
    """Ontology large enough to support 100+ unique anchor combinations."""
    return {
        "languages": ["English", "简体中文"],
        "knowledge_domains": {
            f"domain_{i:02d}": {"topic": [f"topic_{i}"]}
            for i in range(10)
        },
        "capabilities": {
            "knowledge_response": ["qa", "explanation", "comparison"],
            "reasoning": ["reasoning", "math_solving", "planning"],
            "coding_and_data": ["coding", "debugging", "data_analysis"],
        },
        "conversation_types": {
            "single_turn": ["single_turn"],
            "clarification": ["clarification_2_turn"],
            "troubleshooting": ["troubleshooting_3_turn"],
        },
        "language_features": {
            "style": ["concise"],
            "format": ["paragraph"],
            "difficulty": ["basic"],
            "context_length": ["short"],
            "noise": ["clean"],
            "answer_expectation": ["direct_answer"],
        },
    }


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


# ── compute_turn_distribution ───────────────────────────────────────────────


def test_compute_turn_distribution_even():
    """Even distribution across max_turns."""
    rng = random.Random(42)
    result = compute_turn_distribution(100, 3, rng)
    assert len(result) == 3
    assert sum(result) == 100
    # Should be roughly 33/33/34
    assert min(result) >= 33
    assert max(result) <= 34


def test_compute_turn_distribution_single_turn():
    """max_turns=1 returns all in one bucket."""
    rng = random.Random(42)
    result = compute_turn_distribution(50, 1, rng)
    assert result == [50]


def test_compute_turn_distribution_deterministic():
    """Same seed produces same distribution."""
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    r1 = compute_turn_distribution(100, 5, rng1)
    r2 = compute_turn_distribution(100, 5, rng2)
    assert r1 == r2


# ── sample_anchors ───────────────────────────────────────────────────────────


def test_sample_anchors_count():
    """sample_anchors returns correct count."""
    ontology = load_ontology(Path("ontology/anchor_ontology.json"))
    config = AnchorGenerationConfig(
        target_count=100, seed=42, max_turns=3,
    )
    rng = random.Random(config.seed)
    specs = sample_anchors(ontology, config, rng)
    assert len(specs) == 100
    assert all(isinstance(s, AnchorSpec) for s in specs)


def test_sample_anchors_turn_distribution():
    """Turn counts are distributed across 1..max_turns."""
    ontology = load_ontology(Path("ontology/anchor_ontology.json"))
    config = AnchorGenerationConfig(
        target_count=100, seed=42, max_turns=3,
    )
    rng = random.Random(config.seed)
    specs = sample_anchors(ontology, config, rng)

    # Count anchors by number of turns
    turn_counts: dict[int, int] = {}
    for s in specs:
        n = len(s.turns)
        turn_counts[n] = turn_counts.get(n, 0) + 1

    # All turn counts should be odd (1, 3, 5, ...) since last turn must be user
    assert all(k % 2 == 1 for k in turn_counts)
    # With max_turns=3, valid odd counts are 1 and 3 → ~50/50 distribution
    for count in turn_counts.values():
        assert 40 <= count <= 60  # roughly 50/50


def test_sample_anchors_single_turn():
    """max_turns=1 produces all single-turn specs."""
    ontology = load_ontology(Path("ontology/anchor_ontology.json"))
    config = AnchorGenerationConfig(
        target_count=50, seed=42, max_turns=1,
    )
    rng = random.Random(config.seed)
    specs = sample_anchors(ontology, config, rng)
    assert len(specs) == 50
    assert all(len(s.turns) == 1 for s in specs)
    assert all(s.turns[0].role == "user" for s in specs)


def test_sample_anchors_turns_alternate():
    """All generated specs have valid role alternation."""
    ontology = load_ontology(Path("ontology/anchor_ontology.json"))
    config = AnchorGenerationConfig(
        target_count=30, seed=7, max_turns=4,
    )
    rng = random.Random(config.seed)
    specs = sample_anchors(ontology, config, rng)

    for s in specs:
        # Verify __post_init__ passes (no exception)
        for i in range(len(s.turns) - 1):
            assert s.turns[i].role != s.turns[i + 1].role
        assert s.turns[0].role == "user"
        assert s.turns[-1].role == "user"


def test_sample_anchors_ids_unique():
    """Each AnchorSpec has a unique id."""
    ontology = load_ontology(Path("ontology/anchor_ontology.json"))
    config = AnchorGenerationConfig(
        target_count=50, seed=42, max_turns=2,
    )
    rng = random.Random(config.seed)
    specs = sample_anchors(ontology, config, rng)
    ids = {s.id for s in specs}
    assert len(ids) == len(specs)


# ── _get_leaf_conversation_types ────────────────────────────────────────────


def test_leaf_conversation_types():
    """Extracts all leaf conversation types from ontology."""
    ontology = _minimal_ontology_payload()
    leaves = _get_leaf_conversation_types(ontology)
    expected = {"single_turn", "clarification_2_turn", "troubleshooting_3_turn"}
    assert set(leaves) == expected


def test_leaf_conversation_types_real_ontology():
    """Real ontology has 7 leaf conversation types."""
    ontology = load_ontology(Path("ontology/anchor_ontology.json"))
    leaves = _get_leaf_conversation_types(ontology)
    assert len(leaves) == 7
    assert "single_turn" in leaves


def test_leaf_conversation_types_fallback():
    """Empty conversation_types returns fallback."""
    ontology: dict = {"conversation_types": {}}
    leaves = _get_leaf_conversation_types(ontology)
    assert leaves == ["single_turn", "multi_turn"]


# ── Config validation ───────────────────────────────────────────────────────


def test_config_max_turns_default():
    """GenerationConfig defaults max_turns=1."""
    from ard.config import GenerationConfig
    g = GenerationConfig()
    assert g.max_turns == 1
    assert g.max_turns_with_image == 1


def test_config_has_no_system_persona_field():
    """`system_persona` is gone — the ontology samples the system prompt (B3).

    Kept as a test rather than as a comment because the field was reachable
    from three layers (config → generation config → generator) and any of them
    could reintroduce it: with ``extra="forbid"`` a config file that still
    carries the key now fails loudly instead of silently doing nothing.
    """
    from ard.config import GenerationConfig
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GenerationConfig(system_persona="none")  # type: ignore[call-arg]
    assert "system_persona" not in GenerationConfig.model_fields


def test_config_rejects_image_turns_exceeds_max():
    """max_turns_with_image > max_turns fails validation."""
    from ard.config import GenerationConfig
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GenerationConfig(max_turns=2, max_turns_with_image=5)


def test_config_accepts_valid_combination():
    """Valid max_turns/max_turns_with_image combination passes."""
    from ard.config import GenerationConfig
    g = GenerationConfig(max_turns=3, max_turns_with_image=2)
    assert g.max_turns == 3
    assert g.max_turns_with_image == 2


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


# ── load_config with new fields ─────────────────────────────────────────────


def test_load_config_with_new_fields(tmp_path):
    """load_config reads new generation fields from TOML."""
    from ard.config import load_config, ARDConfig

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[input_generator]\n"
        'api_base = "https://api.example.com/v1"\n'
        'model_name = "test-model"\n'
        'api_key = "sk-test"\n'
        "[target_model]\n"
        'api_base = "https://api.example.com/v1"\n'
        'model_name = "target-model"\n'
        'api_key = "sk-target"\n'
        "[generation]\n"
        "max_turns = 3\n"
        "max_turns_with_image = 2\n"
    )
    config = load_config(str(config_path))
    assert isinstance(config, ARDConfig)
    assert config.generation.max_turns == 3
    assert config.generation.max_turns_with_image == 2