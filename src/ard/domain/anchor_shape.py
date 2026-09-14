"""Message-shape invariants for generated anchors (gate, not retreat).

The anchor format has one structural contract, stated in
:mod:`ard.core.types`: a conversation starts with ``user``, ends with
``user`` and alternates roles strictly — and, since v3.0.0, may be preceded
by an *optional single* ``system`` message at position 0 (D1: the ``messages``
array is the single source of truth for the system prompt).  ``AnchorSpec``
enforces the turn half of it on the way *in*, but the entry gate alone never
protected the output — that is how ``UAUAU``-shaped anchors reached a
published anchor bank.  This module holds the **single implementation** of
that contract so every entry/exit check (including the build-time gate in
:mod:`ard.domain.text_anchor`) can never drift apart (§1.4 单一真相源).

It is a boundary check, not a fallback (§2.3): a violation means the anchor
must be discarded, loudly.
"""

from __future__ import annotations

from typing import Any

from ard.core.types import AnchorSpec

__all__ = ["expected_message_roles", "message_shape_error"]


def expected_message_roles(spec: AnchorSpec) -> list[str]:
    """Return the expected *conversation* role sequence for *spec*.

    One message per ``TurnSpec``: user turns are produced by the input
    generator, assistant turns by the target model.  Because every turn yields
    exactly one message, the expected sequence is simply the declared turn
    roles.  ``AnchorSpec`` already guarantees this sequence starts with
    ``user``, ends with ``user`` and alternates.

    The sequence deliberately contains **no** ``system``: an optional leading
    system message is an orthogonal, message-level concern handled by
    :func:`message_shape_error` (D1), not part of the turn list.

    Args:
        spec: Anchor specification.

    Returns:
        List of ``"user"`` / ``"assistant"`` strings, one per turn.
    """
    return [turn.role for turn in spec.turns]


def message_shape_error(messages: list[Any]) -> str | None:
    """Validate the structural shape of a message list.

    This is the **single implementation** of the anchor shape contract
    (``src/ard/core/types.py`` + v3.0.0 D1):

    * an *optional single* ``system`` message may open the conversation at
      position 0 — the OpenAI ``messages`` format, where the array is the
      only source of truth for the system prompt;
    * once that optional leading system is stripped, the conversation must
      start with ``user``, end with ``user`` and alternate roles strictly.

    ``AnchorSpec`` enforces the turn half of this at the *entry* boundary;
    this function exists so the same contract can be checked at every *exit* /
    build-time gate (bank persistence, post-conversion, in-progress build)
    before an anchor is released (§1.4 单一真相源, §2.3 边界校验即防呆).

    Args:
        messages: Message dicts (or any objects with a ``role``/``get``).

    Returns:
        ``None`` when the shape is valid, otherwise a human-readable
        description of the first violation found.
    """
    if not messages:
        return "no messages"

    def _role(msg: Any) -> Any:
        # Messages are plain dicts everywhere in this project; tolerate any
        # mapping exposing ``role`` so the check never raises on bad input.
        if isinstance(msg, dict):
            return msg.get("role")
        return getattr(msg, "role", None)

    roles = [_role(m) for m in messages]

    # Vocabulary is ``user`` / ``assistant`` plus the optional ``system``.
    if any(r not in ("user", "assistant", "system") for r in roles):
        index = next(
            i for i, r in enumerate(roles) if r not in ("user", "assistant", "system")
        )
        return f"message {index} has unknown role {roles[index]!r}"

    # system: at most one, and only at position 0 (D1).  Checked *before* the
    # conversation shape so a misplaced system is reported as such, not as a
    # garbled conversation shape.
    if roles.count("system") > 1:
        return "at most one 'system' message is allowed"
    if "system" in roles and roles[0] != "system":
        return "a 'system' message must be the first message"

    # Conversation shape: strip the (validated) single leading system, then
    # require first ``user``, last ``user`` and strict alternation on the rest.
    base = 1 if roles and roles[0] == "system" else 0
    conversation = roles[base:]
    if not conversation:
        return "a 'system' message must be followed by conversation messages"
    if conversation[0] != "user":
        return f"first message role must be 'user', got {conversation[0]!r}"
    if conversation[-1] != "user":
        return f"last message role must be 'user', got {conversation[-1]!r}"
    for i in range(len(conversation) - 1):
        if conversation[i] == conversation[i + 1]:
            return (
                f"messages {base + i} and {base + i + 1} share role "
                f"{conversation[i]!r} (roles must alternate)"
            )
    return None
