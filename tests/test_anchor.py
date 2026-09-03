"""Tests for ARD v2 — core types, ontology, config, sampler, embeddings, CLI."""

import json
from pathlib import Path

import pytest

from ard.core.types import Anchor, AnchorGenerationConfig
from ard.core.ontology import load_ontology
from ard.core.sampler import sample_anchors, generate_anchor_id
from ard.core.embeddings import load_embeddings, farthest_point_sampling


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
        "conversation_types": {"single_turn": ["single_turn"]},
        "language_features": {
            "style": ["concise"],
            "format": ["paragraph"],
            "difficulty": ["basic"],
            "context_length": ["short"],
            "noise": ["clean"],
            "answer_expectation": ["direct_answer"],
        },
    }


# ── Types ───────────────────────────────────────────────────────────────────


def test_anchor_type_defaults():
    """Anchor dataclass creates with required fields."""
    a = Anchor(
        id="test_001",
        messages=[{"role": "user", "content": "hello"}],
        target_answer="world",
        target_model="gpt-4",
        input_generator_model="claude",
    )
    assert a.id == "test_001"
    assert a.target_answer == "world"
    assert a.anchor_meta == {}
    assert a.logprobs is None


def test_anchor_type_with_logprobs():
    """Anchor stores logprobs when provided."""
    logprobs = {"token_ids": [1, 2, 3], "log_probs": [-0.1, -0.2, -0.3]}
    a = Anchor(
        id="a",
        messages=[],
        target_answer="x",
        target_model="m",
        input_generator_model="m",
        logprobs=logprobs,
    )
    assert a.logprobs == logprobs


def test_anchor_generation_config_defaults():
    """AnchorGenerationConfig has sensible defaults."""
    c = AnchorGenerationConfig()
    assert c.target_count == 100
    assert c.seed == 42
    assert c.languages == []
    assert c.task_types == []


def test_anchor_generation_config_custom():
    """AnchorGenerationConfig accepts custom values."""
    c = AnchorGenerationConfig(
        target_count=50, seed=7, languages=["English"], task_types=["qa", "coding"]
    )
    assert c.target_count == 50
    assert c.seed == 7
    assert c.languages == ["English"]
    assert c.task_types == ["qa", "coding"]


# ── Ontology ────────────────────────────────────────────────────────────────


def test_ontology_loads_valid(tmp_path):
    """load_ontology returns parsed dict for valid JSON."""
    path = tmp_path / "ontology.json"
    path.write_text(json.dumps(_minimal_ontology_payload()), encoding="utf-8")
    result = load_ontology(path)
    assert result["languages"] == ["English"]
    assert "domain_a" in result["knowledge_domains"]


def test_ontology_loads_real_file():
    """load_ontology loads the real anchor_ontology.json."""
    ontology = load_ontology(Path("configs/anchor_ontology.json"))
    assert isinstance(ontology, dict)
    assert "languages" in ontology
    assert "knowledge_domains" in ontology
    assert "capabilities" in ontology


def test_ontology_file_not_found():
    """load_ontology raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_ontology("nonexistent.json")


# ── Sampler ─────────────────────────────────────────────────────────────────


def test_sample_anchors_returns_list(tmp_path):
    """sample_anchors returns a list of meta dicts."""
    path = tmp_path / "ontology.json"
    path.write_text(json.dumps(_minimal_ontology_payload()), encoding="utf-8")
    ontology = load_ontology(path)
    config = AnchorGenerationConfig(target_count=4, seed=1, languages=["English"], task_types=["qa"])
    result = sample_anchors(ontology, config)
    assert isinstance(result, list)
    assert len(result) <= 4
    assert all(isinstance(item, dict) for item in result)


def test_sample_anchors_deterministic():
    """Same seed+config produces same output."""
    ontology = _minimal_ontology_payload()
    config = AnchorGenerationConfig(target_count=4, seed=42, languages=["English"], task_types=["qa"])
    r1 = sample_anchors(ontology, config)
    r2 = sample_anchors(ontology, config)
    assert r1 == r2


def test_generate_anchor_id_stable():
    """generate_anchor_id produces stable hashes."""
    meta = {"language": "English", "knowledge_domain": "math", "capability": "qa"}
    id1 = generate_anchor_id(meta)
    id2 = generate_anchor_id(meta)
    assert id1 == id2
    assert id1.startswith("anchor_")
    assert len(id1) == 23  # "anchor_" + 16 hex chars


def test_generate_anchor_id_different_inputs():
    """Different meta produces different IDs."""
    id1 = generate_anchor_id({"language": "English", "knowledge_domain": "a", "capability": "x"})
    id2 = generate_anchor_id({"language": "English", "knowledge_domain": "b", "capability": "x"})
    assert id1 != id2


# ── Embeddings ──────────────────────────────────────────────────────────────


def test_farthest_point_sampling_basic():
    """farthest_point_sampling returns n indices."""
    import numpy as np
    emb = np.random.randn(100, 64).astype(np.float32)
    indices = farthest_point_sampling(emb, 10, seed=42)
    assert len(indices) == 10
    assert len(set(indices)) == 10  # all unique
    assert all(0 <= i < 100 for i in indices)


def test_farthest_point_sampling_n_equals_N():
    """When n == N, returns all indices."""
    import numpy as np
    emb = np.random.randn(5, 8).astype(np.float32)
    indices = farthest_point_sampling(emb, 5, seed=42)
    assert sorted(indices) == [0, 1, 2, 3, 4]


def test_farthest_point_sampling_n_one():
    """When n == 1, returns a single index."""
    import numpy as np
    emb = np.random.randn(10, 8).astype(np.float32)
    indices = farthest_point_sampling(emb, 1, seed=42)
    assert len(indices) == 1


def test_farthest_point_sampling_deterministic():
    """Same seed produces same result."""
    import numpy as np
    emb = np.random.randn(50, 16).astype(np.float32)
    r1 = farthest_point_sampling(emb, 5, seed=123)
    r2 = farthest_point_sampling(emb, 5, seed=123)
    assert r1 == r2


def test_farthest_point_sampling_empty_raises():
    """Empty array raises ValueError."""
    import numpy as np
    with pytest.raises(ValueError, match="empty"):
        farthest_point_sampling(np.array([]).reshape(0, 8), 1)


def test_farthest_point_sampling_n_out_of_range():
    """n out of range raises ValueError."""
    import numpy as np
    emb = np.random.randn(10, 8).astype(np.float32)
    with pytest.raises(ValueError, match="n must be"):
        farthest_point_sampling(emb, 0)
    with pytest.raises(ValueError, match="n must be"):
        farthest_point_sampling(emb, 11)


# ── Config ──────────────────────────────────────────────────────────────────


def test_config_load_minimal(tmp_path):
    """load_config loads a minimal config.toml."""
    from ard.config import load_config, ARDConfig

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[input_generator]\n"
        'api_base = "https://api.example.com/v1"\n'
        'model_name = "test-model"\n'
        'api_key = "sk-test"\n'
        "[target]\n"
        'api_base = "https://api.example.com/v1"\n'
        'model_name = "target-model"\n'
        'api_key = "sk-target"\n'
    )
    config = load_config(str(config_path))
    assert isinstance(config, ARDConfig)
    assert config.input_generator.api_base == "https://api.example.com/v1"
    assert config.target.model_name == "target-model"
    # Defaults
    assert config.generation.target_count == 100
    assert config.generation.seed == 42


def test_config_load_with_override(tmp_path):
    """load_config deep-merges override config."""
    from ard.config import load_config

    base = tmp_path / "config.toml"
    base.write_text(
        "[input_generator]\n"
        'api_base = ""\n'
        'model_name = ""\n'
        'api_key = ""\n'
        "[target]\n"
        'api_base = ""\n'
        'model_name = ""\n'
        'api_key = ""\n'
    )
    override = tmp_path / "config.override.toml"
    override.write_text(
        "[input_generator]\n"
        'api_base = "https://real.example.com/v1"\n'
        'model_name = "real-model"\n'
        'api_key = "real-key"\n'
        "[target]\n"
        'api_base = "https://real.example.com/v1"\n'
        'model_name = "real-target"\n'
        'api_key = "real-target-key"\n'
    )
    config = load_config(str(base), str(override))
    assert config.input_generator.api_base == "https://real.example.com/v1"
    assert config.target.model_name == "real-target"


def test_config_validation_rejects_unknown_fields(tmp_path):
    """load_config rejects unknown fields in config."""
    from ard.config import load_config

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[input_generator]\n"
        'api_base = "https://api.example.com"\n'
        'model_name = "m"\n'
        'api_key = "k"\n'
        "[target]\n"
        'api_base = "https://api.example.com"\n'
        'model_name = "m"\n'
        'api_key = "k"\n'
        "[unknown_section]\n"
        "foo = 1\n"
    )
    with pytest.raises(ValueError):
        load_config(str(config_path))


def test_config_missing_file():
    """load_config raises FileNotFoundError for missing base."""
    from ard.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.toml")


# ── CLI ─────────────────────────────────────────────────────────────────────


def test_cli_requires_config():
    """CLI parser requires --config."""
    from ard.cli import main as _  # ensure importable

    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "ard", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0
    assert "--config" in result.stdout