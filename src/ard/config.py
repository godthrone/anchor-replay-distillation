from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

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


class InputGeneratorConfig(BaseModel):
    """Configuration for the input generator (question creator) LLM API."""

    model_config = ConfigDict(extra="forbid")

    api_base: str = ""
    model_name: str = ""
    api_key: str = ""  # secret — override in config.override.toml
    temperature: float = 0.8
    max_tokens: int = 4096
    timeout: float = 120.0
    max_retries: int = 3


class TargetConfig(BaseModel):
    """Configuration for the target (answer provider) LLM API.

    Same fields as InputGeneratorConfig, but temperature defaults to 0.0
    for deterministic answers.
    """

    model_config = ConfigDict(extra="forbid")

    api_base: str = ""
    model_name: str = ""
    api_key: str = ""  # secret — override in config.override.toml
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 120.0
    max_retries: int = 3


class OntologyConfig(BaseModel):
    """Configuration for the anchor ontology used for question generation."""

    model_config = ConfigDict(extra="forbid")

    path: str = "configs/anchor_ontology.json"


class GenerationConfig(BaseModel):
    """Configuration for the question generation pipeline."""

    model_config = ConfigDict(extra="forbid")

    target_count: int = 100
    seed: int = 42
    languages: list[str] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)


class OutputConfig(BaseModel):
    """Configuration for dataset output."""

    model_config = ConfigDict(extra="forbid")

    dir: str = ""
    overwrite: bool = False


# ── Top-level config ──────────────────────────────────────────────────────


class ARDConfig(BaseModel):
    """Top-level ARD configuration, composed of all section configs.

    Loaded from one or two TOML files. See :func:`load_config`.
    """

    model_config = ConfigDict(extra="forbid")

    input_generator: InputGeneratorConfig = Field(default_factory=InputGeneratorConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    ontology: OntologyConfig = Field(default_factory=OntologyConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


# Resolve forward references — required because of `from __future__ import annotations`.
ARDConfig.model_rebuild()


# ── Deep merge helper ─────────────────────────────────────────────────────


def _deep_merge(base: dict, override: dict) -> dict:
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
    override_path: Optional[str | Path] = None,
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

    # 3. Validate
    try:
        return ARDConfig.model_validate(merged_dict)
    except Exception as exc:
        raise ValueError(
            f"Configuration validation failed. "
            f"Check that all fields match the expected schema. "
            f"Details: {exc}"
        ) from exc