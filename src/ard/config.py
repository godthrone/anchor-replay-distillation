# config.py — Configuration model definitions and loading.
# Responsibility: define Pydantic models for all config sections,
# load and validate TOML config files, deep-merge override configs.

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


# ── Boundary validation (§2.3) ────────────────────────────────────────────


class ConfigError(ValueError):
    """A configuration value is missing or unusable at a boundary.

    Raised where the value is *first* consumed — config validation or the start
    of a run — and never after a side effect has been performed, so an unusable
    config cannot leave a half-built output directory behind (§2.3 边界校验即防呆).
    Subclasses :class:`ValueError` because that is what a caller supplying a bad
    config value would expect; the CLI catches it explicitly to print the
    message instead of a traceback.
    """


class LLMEndpoint(BaseModel):
    """The non-optional view of one LLM section's credentials.

    :meth:`_LLMConfig.resolved_endpoint` builds this after checking the fields
    the pipeline cannot run without.  Downstream code therefore sees plain
    ``str`` instead of ``str | None``: "this might be unset" is refused once, at
    the boundary, rather than re-checked (or silently assumed away) at every
    use site (§2.1 契约即防呆).
    """

    model_config = ConfigDict(frozen=True)

    api_base: str
    model_name: str
    api_key: str | None = None
    """Secret — never logged.  ``None`` is legal: many OpenAI-compatible local
    servers (vLLM, llama.cpp) need no bearer token, and the shipped base config
    leaves the field empty on purpose."""


# ── Section configs ───────────────────────────────────────────────────────


class _LLMConfig(BaseModel):
    """Common base for LLM API configurations.

    Shared fields for both input generator and target model backends.
    """

    model_config = ConfigDict(extra="forbid")

    SECTION: ClassVar[str] = "llm"
    """TOML table this section is loaded from — used to make a boundary error
    point at the right table instead of saying "somewhere in the config"."""

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

    @field_validator("api_base", "model_name", "api_key", mode="before")
    @classmethod
    def _empty_string_means_unset(cls, value: object) -> object:
        """Normalize the TOML "left blank" placeholder to ``None`` (§2.2).

        ``configs/config.toml`` ships ``api_base = ""`` for the user to fill in,
        and TOML has no ``null``.  ``""`` must therefore become ``None`` at the
        model boundary — otherwise a blank endpoint satisfies every ``is None``
        check and only fails deep inside the HTTP client (the original
        ``ValueError: api_base must not be None`` traceback).  Doing it here also
        covers sections built directly in Python, not just TOML loads.
        """
        if value == "":
            return None
        return value

    def resolved_endpoint(self) -> LLMEndpoint:
        """Return this section's credentials, or refuse with a field-level error.

        Returns:
            The endpoint with ``api_base`` and ``model_name`` narrowed to ``str``.

        Raises:
            ConfigError: If ``api_base`` or ``model_name`` is unset.  The message
                names the missing field(s), the TOML table, and how to set them.

        The pipeline calls this **before** it creates its output directory, so
        an unusable config cannot leave an empty directory (or a config snapshot)
        behind — the failure is a readable boundary rejection, not a traceback
        from somewhere deep in the HTTP layer.
        """
        api_base = self.api_base
        model_name = self.model_name
        if api_base is None or model_name is None:
            missing = [
                name
                for name, value in (("api_base", api_base), ("model_name", model_name))
                if value is None
            ]
            listed = ", ".join(f"`{name}`" for name in missing)
            example = "\n".join(
                (
                    f"[{self.SECTION}]",
                    'api_base = "http://<host>:<port>/v1"',
                    'model_name = "<served-model-name>"',
                )
            )
            raise ConfigError(
                f"[{self.SECTION}] is missing {listed}. The LLM endpoint cannot be "
                f"called. Set the missing field(s) in the config passed to --config, "
                f"or in the override file next to it (config.override.toml), e.g.:\n"
                f"{example}"
            )
        return LLMEndpoint(api_base=api_base, model_name=model_name, api_key=self.api_key)


class InputGeneratorConfig(_LLMConfig):
    """Configuration for the input generator (question creator) LLM API."""

    SECTION: ClassVar[str] = "input_generator"

    temperature: float = 0.8


class TargetModelConfig(_LLMConfig):
    """Configuration for the target answering model (= teacher side) LLM API.

    The target model emits the anchor's target answer
    (``targets[0].output``) — the supervision signal downstream SFT/OPD call
    the "teacher".  Temperature defaults to 0.1 (user ruling, R12: was 0.0 —
    deterministic — which under-sampled answer diversity; 0.1 keeps answers
    near-greedy while letting the teacher vary phrasing between anchors).
    """

    SECTION: ClassVar[str] = "target_model"

    temperature: float = 0.1
    enable_thinking: bool = False
    """Enable thinking/reasoning mode (Qwen3, DeepSeek-R1, etc.).

    When True, the model outputs reasoning before the final answer.
    Set to True only when distilling to a reasoning-capable student model.
    Default False.
    """


class OntologyConfig(BaseModel):
    """Configuration for the anchor ontology used for question generation."""

    model_config = ConfigDict(extra="forbid")

    path: str = "ontology/anchor_ontology.v4.json"


class GenerationConfig(BaseModel):
    """Configuration for the question generation pipeline.

    The anchor **count is not a configurable field**: the v4 construction rule
    derives it from the ontology (one sample per legal restricted block, one per
    rotated leaf — see :mod:`ard.core.sampling`).  A knob here would be a second
    source of truth for a number the rule already fixes (§7.4).
    """

    model_config = ConfigDict(extra="forbid")

    # Unset (``None``, the default) = draw a fresh seed for this run from the
    # system random source; an explicit int pins that run's sampling order.
    seed: int | None = None
    concurrency: int = 4
    max_turns: int = Field(default=1, ge=1, le=10)
    max_turns_with_image: int = Field(default=1, ge=0, le=5)
    backpressure_threshold: int = 3       # 连续超时触发冷却的阈值
    backpressure_cooldown: float = 60.0   # 冷却暂停秒数

    @model_validator(mode="after")
    def _resolve_seed(self) -> GenerationConfig:
        """Draw a run seed when the user did not pin one.

        ``seed`` is optional: leaving it unset means "let this run sample
        freely", so the effective seed is drawn once per config load from the
        OS entropy source and every run differs.  Setting an integer keeps the
        sampling order pinned, i.e. reproducible.

        Resolving here — instead of forwarding ``None`` downstream — keeps
        exactly one authoritative seed per run: it is the value the pipeline
        snapshots into ``<output_dir>/config.json``, so even a randomised run
        stays replicable after the fact, and the core layer keeps its plain
        ``int`` contract.
        """
        if self.seed is None:
            self.seed = random.SystemRandom().randrange(2**32)
        return self

    @property
    def resolved_seed(self) -> int:
        """The effective sampling seed of this run — always a concrete ``int``.

        This is the strongly typed view of :attr:`seed` for downstream code
        that cannot handle "unset": :meth:`_resolve_seed` has already drawn a
        value whenever the user left ``seed`` unset, and it is this value that
        the pipeline snapshots into ``<output_dir>/config.json``.

        The ``None`` branch below is therefore unreachable for any normally
        validated config, and it raises instead of quietly passing ``None`` on:
        a ``None`` here means validation was bypassed (``model_construct``, a
        future refactor), and handing it to ``random.Random(None)`` would turn
        a supposedly reproducible run random without telling anyone (§2.2
        explicit over implicit, §3.4 no silent degradation).
        """
        seed = self.seed
        if seed is None:
            raise RuntimeError(
                "GenerationConfig.seed is None — _resolve_seed did not run. "
                "Configs must be built through validation (parse/validate), not "
                "model_construct()."
            )
        return seed

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
