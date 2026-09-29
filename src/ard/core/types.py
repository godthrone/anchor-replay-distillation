"""Core data types for ARD — Anchor Replay Distillation.

All types are plain dataclasses with slots=True. No external dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeAlias


class DataSource(StrEnum):
    """Controlled vocabulary for the per-record routing key the training side reads
    (§2.1 契约即防呆).

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
    """The conversation *prefix*, one :class:`TurnSpec` per message, ending on
    the final user question — the answer is the generation target, not a turn
    here, so the count is **odd** (``user`` first, ``user`` last, roles
    alternating; see Validation).

    This is **not** the number an ontology ``conversation_type`` declares: its
    ``turns`` attribute counts *exchanges* (one user question plus the answer it
    receives), so ``n`` ontology turns are ``2 * n - 1`` spec turns —
    ``single_turn`` 1 → 1, ``clarification`` 2 → 3, ``constraint_update`` 4 → 7
    (see :func:`ard.core.sampling.turn_counts_by_conversation_type`).  Reading
    the ontology count as a message count is impossible: an even count could not
    start and end on ``user`` and would be rejected by ``__post_init__``.
    """
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
                raise ValueError(f"Turns {i} and {i + 1} have same role '{self.turns[i].role}'")


@dataclass(slots=True)
class GeneratedAnchor:
    """A fully generated anchor, ready for serialization.

    This is the output of the anchor generation pipeline — all messages, the
    target answer, and (when the teacher thought before answering) the teacher's
    reasoning trace have been produced by the respective models.
    """

    id: str
    messages: ChatMessageList
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
    """Multi-teacher routing key the training side reads (controlled vocabulary).

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
    """Lightweight config for anchor generation (core layer, not pydantic).

    Deliberately tiny: everything that defines *what* the plan contains is
    derived from the ontology and the construction rule (§1.4 single source of
    truth).  Only the run seed (deterministic sampling) and the generation
    concurrency belong here.
    """

    seed: int = 42
    concurrency: int = 4


# ── Shared type aliases ─────────────────────────────────────────────────────
#
# §12.1 rule 4 forbids anonymous nested container types: a shape written inline
# at every signature has no name to state what it means, and changing it means
# editing every caller.  These aliases name the shapes that cross the
# core / domain / backends / pipeline boundaries — one name per shape, so there
# is a single source of truth for it (§1.4).  They live in the core layer
# because core imports nothing from the rest of the project (§1.3 层次边界), so
# every layer can import them without a cycle.  Each definition is a runtime
# ``types.GenericAlias``, i.e. exactly the expression it replaces, with no
# runtime cost and no behaviour change.

JsonObject: TypeAlias = dict[str, Any]
"""One decoded JSON object: a bank record, a manifest fragment, a plan entry."""

JsonObjectMapping: TypeAlias = Mapping[str, Any]
"""The same object seen through a read-only, mapping-like view."""

JsonObjectList: TypeAlias = list[JsonObject]
"""Decoded JSON objects in file or plan order."""

JsonObjectSequence: TypeAlias = Sequence[JsonObjectMapping]
"""A read-only run of decoded JSON objects, in file or plan order.

The element type is the mapping view, not ``JsonObject``: every consumer only
reads, so callers may hand over real ``dict``s *or* mapping-like doubles.
"""

ChatContentPart: TypeAlias = dict[str, Any]
"""One typed part of a multimodal chat content list.

``{"type": "text", "text": ...}`` or ``{"type": "image_url", "image_url": ...}``
— the endpoint owns this shape, so it is named rather than modelled.
"""

ChatContentParts: TypeAlias = list[ChatContentPart]
"""A message's content when it is a list of parts instead of a plain string."""

ChatMessage: TypeAlias = dict[str, Any]
"""One OpenAI-compatible chat message: a ``role`` plus its ``content``.

The content is either a plain string or, for multimodal turns, a
:data:`ChatContentParts` list.  It stays a dict on purpose: this is the
provider's **wire format**, serialised verbatim into the request payload and
parsed straight out of the response, so the shape is owned by the endpoint
rather than by this project.  §12.1's ban on anonymous nested types is
satisfied by naming the shape; wrapping it in a pydantic model would add a
translation layer whose only possible behaviour is to drift from the format it
mirrors.
"""

ChatMessageList: TypeAlias = list[ChatMessage]
"""A conversation: the messages of one request, in turn order."""

StringList: TypeAlias = list[str]
"""A list of strings: anchor ids, names, legal values."""

StringSet: TypeAlias = set[str]
"""An unordered, hashable set of strings: anchor ids already seen, and the like."""

StringTuple: TypeAlias = tuple[str, ...]
"""An ordered, hashable run of strings — one axis's values."""

StringPair: TypeAlias = tuple[str, str]
"""Two strings whose order carries the meaning (``axis -> value``)."""

StringPairs: TypeAlias = tuple[StringPair, ...]
"""An ordered, hashable run of :data:`StringPair`.

This is how a coordinate is identified: its ``(axis, value)`` pairs, sorted, so
two coordinates built in a different order still compare equal.
"""

AxisValuesByAxis: TypeAlias = dict[str, StringTuple]
"""For each axis name, the values that axis may take."""

AnchorSpecList: TypeAlias = list[AnchorSpec]
"""The anchors of one plan — or of one plan subgroup — in plan order."""
