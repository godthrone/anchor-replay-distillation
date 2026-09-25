"""Tests for ARD — core types, config, sampling ids, CLI."""

from pathlib import Path

import pytest

from ard.core.sampling import generate_anchor_id
from ard.core.types import AnchorGenerationConfig, GeneratedAnchor

# ── Types ───────────────────────────────────────────────────────────────────


def test_anchor_type_defaults():
    """Anchor dataclass creates with required fields."""
    a = GeneratedAnchor(
        id="test_001",
        messages=[{"role": "user", "content": "hello"}],
        target_answer="world",
        target_model="gpt-4",
        input_generator_model="claude",
        anchor_meta={},
    )
    assert a.id == "test_001"
    assert a.target_answer == "world"
    assert a.anchor_meta == {}
    assert a.reasoning is None


def test_anchor_type_with_reasoning():
    """Anchor stores the teacher's reasoning trace when provided."""
    a = GeneratedAnchor(
        id="a",
        messages=[],
        target_answer="x",
        target_model="m",
        input_generator_model="m",
        anchor_meta={},
        reasoning="six times seven is forty-two",
    )
    assert a.reasoning == "six times seven is forty-two"
    # Reasoning is not the answer: the two fields stay independent.
    assert a.target_answer == "x"


def test_anchor_generation_config_defaults():
    """AnchorGenerationConfig carries only the run seed and the concurrency."""
    c = AnchorGenerationConfig()
    assert c.seed == 42
    assert c.concurrency == 4


def test_anchor_generation_config_custom():
    """AnchorGenerationConfig accepts custom values."""
    c = AnchorGenerationConfig(seed=7, concurrency=8)
    assert c.seed == 7
    assert c.concurrency == 8


# ── Anchor ids ──────────────────────────────────────────────────────────────


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
        "[target_model]\n"
        'api_base = "https://api.example.com/v1"\n'
        'model_name = "target-model"\n'
        'api_key = "sk-target"\n'
    )
    config = load_config(str(config_path))
    assert isinstance(config, ARDConfig)
    assert config.input_generator.api_base == "https://api.example.com/v1"
    assert config.target_model.model_name == "target-model"
    # Defaults.  There is no ``target_count`` any more: WP-S2a derives the anchor
    # count from the ontology's construction rule, so asserting a default count
    # would assert a number the config no longer owns.
    assert config.ontology.path == "ontology/anchor_ontology.v4.json"
    # `seed` is unset here → the config layer resolves it to a concrete int
    # drawn from the system random source (2026-09-15: unset = random,
    # explicit int = pinned). It is never left as None downstream.
    assert isinstance(config.generation.seed, int)
    assert config.generation.concurrency == 4


def test_config_unset_seed_draws_a_fresh_seed_per_config():
    """未配置 seed → 每个 config 各取一个新的随机 int（未配置 = 随机）。"""
    from ard.config import GenerationConfig

    seeds = {GenerationConfig().seed for _ in range(8)}
    assert all(isinstance(s, int) for s in seeds)
    # 8 次抽取在 2**32 空间上全部碰撞的概率约 7 * 2**-32 —— 不是 flaky 断言。
    assert len(seeds) > 1


def test_config_explicit_seed_is_pinned(tmp_path):
    """显式 seed（代码或 TOML）→ 原样保留，保证同 seed 可复现。"""
    from ard.config import GenerationConfig, load_config

    assert GenerationConfig(seed=42).seed == 42
    assert GenerationConfig(seed=0).seed == 0

    config_path = tmp_path / "config.toml"
    config_path.write_text("[generation]\nseed = 42\n")
    assert load_config(str(config_path)).generation.seed == 42


def test_config_resolved_seed_is_the_effective_int():
    """`resolved_seed` 是 config 层的强类型生效值（即落盘 config.json 的值）。

    核心层取的是 `resolved_seed` 而不是 `seed`（后者的静态类型是
    `int | None`），因此它必须返回 int；而绕过校验的 config 不得静默把
    None 传下去。
    """
    from ard.config import GenerationConfig

    assert GenerationConfig(seed=42).resolved_seed == 42

    g = GenerationConfig()
    assert isinstance(g.resolved_seed, int)
    assert g.resolved_seed == g.seed

    # 绕过校验的实例（model_construct）必须显式报错，而不是把 None 交给下游
    # 的 random.Random(None) 静默变成不可复现。
    with pytest.raises(RuntimeError, match="did not run"):
        _ = GenerationConfig.model_construct(seed=None).resolved_seed


def test_config_load_with_override(tmp_path):
    """load_config deep-merges override config."""
    from ard.config import load_config

    base = tmp_path / "config.toml"
    base.write_text(
        "[input_generator]\n"
        'api_base = ""\n'
        'model_name = ""\n'
        'api_key = ""\n'
        "[target_model]\n"
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
        "[target_model]\n"
        'api_base = "https://real.example.com/v1"\n'
        'model_name = "real-target"\n'
        'api_key = "real-target-key"\n'
    )
    config = load_config(str(base), str(override))
    assert config.input_generator.api_base == "https://real.example.com/v1"
    assert config.target_model.model_name == "real-target"


def test_config_validation_rejects_unknown_fields(tmp_path):
    """load_config rejects unknown fields in config."""
    from ard.config import load_config

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[input_generator]\n"
        'api_base = "https://api.example.com"\n'
        'model_name = "m"\n'
        'api_key = "k"\n'
        "[target_model]\n"
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
    assert t.temperature == 0.1

    g = GenerationConfig()
    # Neither the count nor the turn counts are configurable: the v4 construction
    # rule derives the count and the ontology supplies each entry's turns, so the
    # model has no such field to default.
    # Unset seed is resolved to a concrete int at construction time.
    assert isinstance(g.seed, int)
    assert g.concurrency == 4

    o = OntologyConfig()
    assert o.path == "ontology/anchor_ontology.v4.json"

    out = OutputConfig()
    assert out.directory is None
    assert out.overwrite is False


def test_ard_config_full():
    """ARDConfig composes all sections."""
    from ard.config import ARDConfig

    c = ARDConfig()
    assert c.output.overwrite is False
    assert c.ontology.path == "ontology/anchor_ontology.v4.json"
