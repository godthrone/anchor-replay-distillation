"""Core data types for ARD — Anchor Replay Distillation.

All types are plain dataclasses with slots=True. No external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DataSource(StrEnum):
    """Controlled vocabulary for the per-record OPD routing key (§2.1 契约即防呆).

    ``data_source`` tells the training side (verl / slime route by this key)
    which ARD sub-corpus a record came from.  It is an enum rather than a free
    string because a typo used to be written to the bank silently — the
    training framework would then route on a value nobody produces.

    Attributes:
        ARD_TEXT: A text-only anchor, produced by the text pipeline.
        ARD_MULTI: A multimodal anchor (at least one turn carries an image).
    """

    ARD_TEXT = "ard_text"
    ARD_MULTI = "ard_multi"


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

    This is the output of the anchor generation pipeline — all messages, the
    target answer, and (when the teacher thought before answering) the teacher's
    reasoning trace have been produced by the respective models.
    """

    id: str
    messages: list[dict[str, Any]]
    target_answer: str
    target_model: str
    input_generator_model: str
    anchor_meta: dict[str, Any]
    reasoning: str | None = None
    """The teacher's reasoning trace for the final answer, or ``None``.

    ``None`` — not ``""`` — is the empty value (§2.2): it is what a run with
    ``enable_thinking = false`` produces, because the server then emits no
    reasoning at all.  Serialized as ``targets[0].output.reasoning``.
    """
    data_source: DataSource = DataSource.ARD_TEXT
    """OPD multi-teacher routing key for the record (controlled vocabulary).

    Identifies which ARD source sub-corpus a record belongs to.  The value is
    checked in :meth:`__post_init__` **and** at the write gate
    (:func:`ard.domain.bank.data_source_error`), because an annotation is not a
    contract: a plain enum annotation still accepts a raw string at run time (it
    is a dataclass, not a pydantic model), so without the check a typo would
    reach the bank as silently as it did when the field was a bare ``str``.  A
    record whose routing key is out of vocabulary is one no consumer ever picks
    up, so it must not be constructible quietly (§2.1 契约即防呆).  Serialized
    as a top-level field, set once here (single source of truth, §1.4).
    """

    def __post_init__(self) -> None:
        if not isinstance(self.data_source, DataSource):
            raise ValueError(
                f"data_source must be a DataSource, got {self.data_source!r} "
                f"(allowed: {[member.value for member in DataSource]})"
            )


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
    embeddings_path: str = "ontology/anchor_ontology_embeddings.json"
