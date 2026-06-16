from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ard.anchor.api_client import ChatAPIConfig

# ── .env key → ARDConfig field mapping ──────────────────────────────────
_KEY_MAP: dict[str, tuple[str, type, Any]] = {
    # (env_key, attr_name, type, default) — default=None means required
    # --- required API ---
    "ARD_INPUT_GENERATOR_API_BASE": ("input_generator_api_base", str, None),
    "ARD_INPUT_GENERATOR_MODEL_NAME": ("input_generator_model_name", str, None),
    "ARD_INPUT_GENERATOR_API_KEY": ("input_generator_api_key", str, None),
    "ARD_TARGET_API_BASE": ("target_api_base", str, None),
    "ARD_TARGET_MODEL_NAME": ("target_model_name", str, None),
    "ARD_TARGET_API_KEY": ("target_api_key", str, None),
    # --- generation ---
    "ARD_TARGET_COUNT": ("target_count", int, 10),
    "ARD_SEED": ("seed", int, 42),
    "ARD_EXACT_FINAL_COUNT_ENABLED": ("exact_final_count_enabled", bool, False),
    "ARD_EXACT_FINAL_COUNT_BATCH_SIZE": ("exact_final_count_batch_size", int, 10),
    "ARD_EXACT_FINAL_COUNT_MAX_BATCHES": ("exact_final_count_max_batches", int, 3),
    # --- output ---
    "ARD_OUTPUT_DIR": ("output_dir", str, ""),
    "ARD_OVERWRITE_OUTPUT": ("overwrite_output", bool, False),
    # --- ontology ---
    "ARD_ONTOLOGY_PATH": ("ontology_path", str, "configs/anchor_ontology.json"),
    "ARD_ONTOLOGY_EMBEDDINGS_PATH": (
        "ontology_embeddings_path",
        str,
        "configs/anchor_ontology_embeddings.json",
    ),
    "ARD_SAMPLING_STRATEGY": ("sampling_strategy", str, "farthest"),
    "ARD_LANGUAGES": ("languages", str, ""),
    "ARD_TASK_TYPES": ("task_types", str, ""),
    # --- API behavior ---
    "ARD_TEMPERATURE": ("temperature", float, 0.7),
    "ARD_TARGET_TEMPERATURE": ("target_temperature", float, 0.0),
    "ARD_TOP_P": ("top_p", float, 0.95),
    "ARD_TIMEOUT": ("timeout", float, 60.0),
    "ARD_MAX_RETRIES": ("max_retries", int, 2),
    "ARD_INPUT_GENERATOR_CONCURRENCY": ("input_generator_concurrency", int, 100),
    "ARD_TARGET_CONCURRENCY": ("target_concurrency", int, 100),
    # --- filtering ---
    "ARD_MIN_TARGET_ANSWER_CHARS": ("min_target_answer_chars", int, 8),
    "ARD_MAX_TARGET_ANSWER_CHARS": ("max_target_answer_chars", int, 0),
    # --- system persona ---
    "ARD_SYSTEM_PERSONAS": ("system_personas", str, ""),
    # --- reasoning ---
    "ARD_TARGET_REASONING_EFFORT": ("target_reasoning_effort", str, ""),
    # --- token caps ---
    "ARD_INPUT_GENERATOR_MAX_TOKENS": ("input_generator_max_tokens", int, 0),
    "ARD_TARGET_MAX_TOKENS": ("target_max_tokens", int, 0),
}

_PLACEHOLDER_VALUES: frozenset[str] = frozenset({"", "...", "replace-me", "your-api-key", "sk-..."})
_BOOL_TRUE: frozenset[str] = frozenset({"1", "true", "yes", "y", "on"})
_BOOL_FALSE: frozenset[str] = frozenset({"0", "false", "no", "n", "off"})


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a KEY=VALUE .env file, stripping comments and quotes."""
    values: dict[str, str] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip().lstrip("﻿")
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SystemExit(f"Invalid .env line {line_no}: expected KEY=VALUE.")
        key, value = line.split("=", 1)
        key = key.strip().lstrip("﻿")
        value = value.strip().strip('"').strip("'")
        if not key:
            raise SystemExit(f"Invalid .env line {line_no}: empty key.")
        values[key] = value
    return values


def _cast_value(env_key: str, raw: str, typ: type, default: Any) -> Any:
    """Cast a raw string from .env to the expected type, or raise SystemExit."""
    if typ is bool:
        lower = raw.lower()
        if lower in _BOOL_TRUE:
            return True
        if lower in _BOOL_FALSE:
            return False
        if not raw and default is not None:
            return default
        raise SystemExit(f"{env_key} must be a boolean (true/false), got {raw!r}.")
    if typ is int:
        if not raw and default is not None:
            return default
        try:
            return int(raw)
        except ValueError:
            raise SystemExit(f"{env_key} must be an integer, got {raw!r}.")
    if typ is float:
        if not raw and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            raise SystemExit(f"{env_key} must be a number, got {raw!r}.")
    # str — empty string is valid
    return raw


def _validate_required(env_key: str, value: str, attr_name: str) -> None:
    """Raise SystemExit if a required value is missing or placeholder."""
    if not value or value.lower() in _PLACEHOLDER_VALUES:
        raise SystemExit(
            f"Set {env_key} in .env before running. "
            f"(Current value: {value!r} — looks like a placeholder.)"
        )


@dataclass(slots=True)
class ARDConfig:
    """All ARD configuration, loaded from a single .env file."""

    # --- required API ---
    input_generator_api_base: str = ""
    input_generator_model_name: str = ""
    input_generator_api_key: str = ""
    target_api_base: str = ""
    target_model_name: str = ""
    target_api_key: str = ""
    # --- generation ---
    target_count: int = 10
    seed: int = 42
    exact_final_count_enabled: bool = False
    exact_final_count_batch_size: int = 10
    exact_final_count_max_batches: int = 3
    # --- output ---
    output_dir: str = ""
    overwrite_output: bool = False
    # --- ontology ---
    ontology_path: str = "configs/anchor_ontology.json"
    ontology_embeddings_path: str = "configs/anchor_ontology_embeddings.json"
    sampling_strategy: str = "farthest"
    languages: str = ""
    task_types: str = ""
    # --- API behavior ---
    temperature: float = 0.7
    target_temperature: float = 0.0
    top_p: float = 0.95
    timeout: float = 60.0
    max_retries: int = 2
    input_generator_concurrency: int = 100
    target_concurrency: int = 100
    # --- filtering ---
    min_target_answer_chars: int = 8
    max_target_answer_chars: int = 0
    # --- system persona ---
    system_personas: str = ""
    # --- reasoning ---
    target_reasoning_effort: str = ""
    # --- token caps ---
    input_generator_max_tokens: int = 0
    target_max_tokens: int = 0

    # project root (set by load())
    _project_root: Path = field(default_factory=Path.cwd, repr=False)

    @classmethod
    def load(cls, env_path: str | Path = ".env") -> "ARDConfig":
        """Load configuration from a .env file, validate, and return an ARDConfig.

        By default reads ``./.env`` relative to the current working directory.
        Every value is read from that one file — no CLI overrides, no env-var fallback.
        """
        env_path = Path(env_path).resolve()
        if not env_path.exists():
            raise SystemExit(
                "No .env file found. Copy .env-example to .env and fill in your API values."
            )

        raw = _parse_env_file(env_path)

        # Build keyword arguments for the dataclass constructor
        kwargs: dict[str, Any] = {"_project_root": env_path.parent}

        for env_key, (attr_name, typ, default) in _KEY_MAP.items():
            str_value = raw.get(env_key, "")
            if default is None:
                # required
                _validate_required(env_key, str_value, attr_name)
                kwargs[attr_name] = _cast_value(env_key, str_value, typ, default)
            else:
                if not str_value:
                    kwargs[attr_name] = default
                else:
                    kwargs[attr_name] = _cast_value(env_key, str_value, typ, default)

        config = cls(**kwargs)
        return config

    @property
    def input_generator_config(self) -> ChatAPIConfig:
        return ChatAPIConfig(
            api_base=self.input_generator_api_base,
            model_name=self.input_generator_model_name,
            api_key=self.input_generator_api_key,
            reasoning_effort=None,  # input gen doesn't use reasoning control
            temperature=self.temperature,
        )

    @property
    def target_config(self) -> ChatAPIConfig:
        return ChatAPIConfig(
            api_base=self.target_api_base,
            model_name=self.target_model_name,
            api_key=self.target_api_key,
            reasoning_effort=self.target_reasoning_effort or None,
            temperature=self.target_temperature,
        )

    @property
    def project_root(self) -> Path:
        return self._project_root

    def resolve_output_dir(self) -> Path:
        """Resolve the output directory, creating a timestamped one if not set."""
        if self.output_dir:
            p = Path(self.output_dir)
            return p if p.is_absolute() else self._project_root / p
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        return (
            self._project_root
            / "outputs"
            / f"ard_anchor_dataset_{timestamp}_n{self.target_count}_seed{self.seed}"
        )

    def resolve_path(self, relative: str) -> Path:
        """Resolve a path that may be relative to the project root."""
        p = Path(relative)
        return p if p.is_absolute() else self._project_root / p

    @property
    def system_personas_list(self) -> list[str] | None:
        """Parse ARD_SYSTEM_PERSONAS into a list, or None to use all."""
        if not self.system_personas:
            return None  # None = use all four
        return [s.strip() for s in self.system_personas.split(",") if s.strip()]

    @property
    def languages_list(self) -> list[str] | None:
        if not self.languages:
            return None
        return [s.strip() for s in self.languages.split(",") if s.strip()]

    @property
    def task_types_list(self) -> list[str] | None:
        if not self.task_types:
            return None
        return [s.strip() for s in self.task_types.split(",") if s.strip()]
