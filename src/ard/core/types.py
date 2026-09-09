"""Core data types for ARD — Anchor Replay Distillation.

All types are plain dataclasses with slots=True. No external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TurnSpec:
    """A single turn in a multi-turn conversation.

    Each turn represents either a user question or an assistant response
    in the conversation flow.  The final turn (``is_final=True``) is the
    last user question that the target model will answer.
    """

    turn_index: int
    role: str  # "user" | "assistant"
    generation_instruction: str | None = None
    image_path: str | None = None
    is_final: bool = False


@dataclass(slots=True)
class AnchorSpec:
    """Unified anchor specification covering all complexity levels.

    An AnchorSpec defines the full conversation structure — how many turns,
    what roles, and what generation instructions — before any API calls are
    made.  It is the input to the anchor generation pipeline.

    Validation:
        * ``turns`` must not be empty.
        * The first turn must be ``"user"``.
        * The last turn must be ``"user"``.
        * Roles must alternate (no two consecutive turns with the same role).
    """

    id: str
    anchor_meta: dict[str, Any]
    turns: list[TurnSpec]
    input_generator_id: str | None = None

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError("AnchorSpec.turns must not be empty")
        if self.turns[0].role != "user":
            raise ValueError("First turn must be user")
        if self.turns[-1].role != "user":
            raise ValueError("Last turn must be user")
        for i in range(len(self.turns) - 1):
            if self.turns[i].role == self.turns[i + 1].role:
                raise ValueError(
                    f"Turns {i} and {i+1} have same role "
                    f"'{self.turns[i].role}'"
                )


@dataclass(slots=True)
class GeneratedAnchor:
    """A fully generated anchor, ready for serialization.

    This is the output of the anchor generation pipeline — all messages,
    the target answer, and optional logprobs have been produced by the
    respective models.
    """

    id: str
    messages: list[dict[str, Any]]
    target_answer: str
    target_model: str
    input_generator_model: str
    anchor_meta: dict[str, Any]
    logprobs: dict[str, Any] | None = None


@dataclass(slots=True)
class AnchorGenerationConfig:
    """Lightweight config for anchor generation (core layer, not pydantic)."""

    target_count: int = 100
    seed: int = 42
    concurrency: int = 4
    languages: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)
    max_turns: int = 1
    max_turns_with_image: int = 1
    system_persona: str = "none"
    embeddings_path: str = "data/anchor_ontology_embeddings.json"
