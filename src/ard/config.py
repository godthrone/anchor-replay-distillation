# config.py — Configuration model definitions and loading.
# Responsibility: define Pydantic models for all config sections,
# load and validate TOML config files, deep-merge override configs.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ── TOML parsing ──────────────────────────────────────────────────────────
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore[no-recheck]
    except ImportError as exc:
        raise ImportError(
            "tomli is required for Python < 3.11. Install with: pip install tomli"
        ) from exc


# ── Section configs ───────────────────────────────────────────────────────


class _LLMConfig(BaseModel):
    """Common base for LLM API configurations.

    Shared fields for both input generator and target model backends.
    """

    model_config = ConfigDict(extra="forbid")

    api_base: str | None = None
    model_name: str | None = None
    api_key: str | None = None  # secret — override in config.override.toml
    max_tokens: int | None = None
    connect_timeout: float = 10.0
    first_token_timeout: float = 300.0
    inter_token_timeout: float = 15.0
    max_retries: int = 3
    retry_on_timeout: bool = False
    temperature: float
    """Sampling temperature — different defaults for input vs target."""


class InputGeneratorConfig(_LLMConfig):
    """Configuration for the input generator (question creator) LLM API."""

    temperature: float = 0.8


class TargetModelConfig(_LLMConfig):
    """Configuration for the target model (answer provider) LLM API.

    Temperature defaults to 0.0 for deterministic answers.
    """

    temperature: float = 0.0
    enable_thinking: bool = False
    """Enable thinking/reasoning mode (Qwen3, DeepSeek-R1, etc.).

    When True, the model outputs reasoning before the final answer.
    Set to True only when distilling to a reasoning-capable student model.
    Default False for deterministic output.
    """


class OntologyConfig(BaseModel):
    """Configuration for the anchor ontology used for question generation."""

    model_config = ConfigDict(extra="forbid")

    path: str = "ontology/anchor_ontology.json"


class GenerationConfig(BaseModel):
    """Configuration for the question generation pipeline."""

    model_config = ConfigDict(extra="forbid")

    target_count: int = 100
    seed: int = 42
    concurrency: int = 4
    languages: list[str] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)
    max_turns: int = Field(default=1, ge=1, le=10)
    system_persona: Literal["none", "one_sentence", "appropriate", "detailed"] = "none"
    max_turns_with_image: int = Field(default=1, ge=0, le=5)
    embeddings_path: str = "ontology/anchor_ontology_embeddings.json"
    backpressure_threshold: int = 3       # 连续超时触发冷却的阈值
    backpressure_cooldown: float = 60.0   # 冷却暂停秒数

    @model_validator(mode="after")
    def _validate_image_turns(self) -> "GenerationConfig":
        if self.max_turns_with_image > self.max_turns:
            raise ValueError(
                f"max_turns_with_image ({self.max_turns_with_image}) cannot exceed "
                f"max_turns ({self.max_turns})"
            )
        return self


class OutputConfig(BaseModel):
    """Configuration for dataset output."""

    model_config = ConfigDict(extra="forbid")

    directory: str | None = None
    overwrite: bool = False


# ── Top-level config ──────────────────────────────────────────────────────


class ARDConfig(BaseModel):
    """Top-level ARD configuration, composed of all section configs.

    Loaded from one or two TOML files. See :func:`load_config`.
    """

    model_config = ConfigDict(extra="forbid")

    input_generator: InputGeneratorConfig = Field(default_factory=InputGeneratorConfig)
    target_model: TargetModelConfig = Field(default_factory=TargetModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    ontology: OntologyConfig = Field(default_factory=OntologyConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


# Resolve forward references — required because of `from __future__ import annotations`.
ARDConfig.model_rebuild()


# ── Empty string normalization ─────────────────────────────────────────────


def _replace_empty_str_with_none(d: dict) -> dict:
    """Recursively replace all empty string ``""`` values with ``None``.

    TOML files often use ``api_key = ""`` as a placeholder for secret fields.
    Pydantic models with ``str | None = None`` will not trigger their
    ``None`` default when ``""`` is loaded — downstream ``is None`` checks
    miss the empty string. This normalizer ensures that ``""`` is treated
    as "not provided" throughout the config.
    """
    result: dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _replace_empty_str_with_none(v)
        elif v == "":
            result[k] = None
        else:
            result[k] = v
    return result


# ── Deep merge helper ─────────────────────────────────────────────────────


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two dictionaries. Leaf values in *override* replace those in *base*.

    Returns a new dict; neither input is mutated.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ── Public loader ─────────────────────────────────────────────────────────


def load_config(
    base_path: str | Path,
    override_path: str | Path | None = None,
) -> ARDConfig:
    """Load and validate ARD configuration.

    Args:
        base_path: Path to the base ``config.toml`` file.
        override_path: Optional path to an override TOML file for deep merge.

    Returns:
        Validated :class:`ARDConfig` instance.

    Raises:
        FileNotFoundError: If *base_path* does not exist.
        ValueError: If configuration validation fails.
    """
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    # 1. Load base config
    with open(base_path, "rb") as fh:
        base_dict = tomllib.load(fh)

    # 2. Deep-merge override if provided
    if override_path is not None:
        override_path = Path(override_path)
        if not override_path.exists():
            raise FileNotFoundError(f"Override config not found: {override_path}")
        with open(override_path, "rb") as fh:
            override_dict = tomllib.load(fh)
        merged_dict = _deep_merge(base_dict, override_dict)
    else:
        merged_dict = base_dict

    # 3. Normalize empty strings to None
    merged_dict = _replace_empty_str_with_none(merged_dict)

    # 4. Validate
    try:
        return ARDConfig.model_validate(merged_dict)
    except Exception as exc:
        raise ValueError(
            f"Configuration validation failed. "
            f"Check that all fields match the expected schema. "
            f"Details: {exc}"
        ) from exc
