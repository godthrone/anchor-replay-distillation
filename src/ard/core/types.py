"""Core data types for ARD — Anchor Replay Distillation.

All types are plain dataclasses with slots=True. No external dependencies.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Anchor:
    """A single anchor data point."""

    id: str
    messages: list[dict[str, Any]]
    target_answer: str
    target_model: str
    input_generator_model: str
    anchor_meta: dict[str, Any] = field(default_factory=dict)
    logprobs: dict[str, Any] | None = None


@dataclass(slots=True)
class AnchorGenerationConfig:
    """Lightweight config for anchor generation (core layer, not pydantic)."""

    target_count: int = 100
    seed: int = 42
    languages: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)