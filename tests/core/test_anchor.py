"""Tests for ARD — core types, config, sampling ids, CLI."""

from pathlib import Path

import pytest

from ard.core.sampling import format_anchor_id
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


# ── Anchor ids (v5: the id is a plan position, not a coordinate digest) ─────
#
# ``generate_anchor_id`` / ``ANCHOR_ID_DIMENSIONS`` are gone.  An id no longer
# hashes the anchor's coordinates; it is the sample's serial number in the plan:
# ``run_key`` mixes only (ontology fingerprint, seed, sampling algorithm), and
# ``format_anchor_id`` appends the zero-padded ``(cycle, position)``.  The two
# properties the rest of the system is built on — pure determinism and stable
# prefixes as N grows — live in ``run_key``/``format_anchor_id`` and are covered
# in depth by ``tests/core/test_sampling.py``; what follows pins the format
# contract at this module's boundary.


def test_anchor_id_format_is_a_position_serial():
    """id 形如 ``<run_key>-c<cycle:05d>p<position:05d>``，是位置的纯函数。"""
    anchor_id = format_anchor_id("c4f4aa6f", 0, 959)
    assert anchor_id == "c4f4aa6f-c00000p00959"
    # Pure: the same (run, cycle, position) is byte-identical every call.
    assert format_anchor_id("c4f4aa6f", 0, 959) == anchor_id


def test_anchor_id_distinguishes_position_cycle_and_run():
    """位置 / 轮次 / run 任一不同 ⇒ id 不同（id 是主键，不承载坐标）。"""
    anchor_id = format_anchor_id("c4f4aa6f", 0, 959)
    assert format_anchor_id("c4f4aa6f", 0, 960) != anchor_id
    assert format_anchor_id("c4f4aa6f", 1, 959) != anchor_id
    assert format_anchor_id("1a2b3c4d", 0, 959) != anchor_id


def test_anchor_id_is_not_derived_from_coordinates():
    """同一坐标在不同位置的 id 必须**不同** —— 坐标相同 ≠ 样本重复（v5 裁定）。

    旧口径下 id 是坐标的哈希，同一个坐标处处得到同一个 id；v5 把 id 换成位置
    序号后，重复坐标是新样本，两个位置各自持有自己的 id，银行两条都留下。
    """
    first_occurrence = format_anchor_id("c4f4aa6f", 0, 3)
    second_occurrence = format_anchor_id("c4f4aa6f", 7, 3)
    assert first_occurrence != second_occurrence


# ── Config ──────────────────────────────────────────────────────────────────


def test_config_load_minimal(tmp_path):
    """load_config loads a minimal config.toml."""
    from ard.config import ARDConfig, load_config

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
    # Defaults.  There is no ``target_count`` any more: the anchor count is
    # derived from the ontology's construction rule, so asserting a default count
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
    """`resolved_seed` 是 config 层的强类型生效值（即落盘 config.toml 的值）。

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
    import subprocess
    import sys

    from ard.cli import main as _  # noqa: F401 — importing it is the assertion

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
        GenerationConfig,
        InputGeneratorConfig,
        OntologyConfig,
        OutputConfig,
        TargetModelConfig,
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
