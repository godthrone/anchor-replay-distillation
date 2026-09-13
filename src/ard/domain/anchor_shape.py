"""Message-shape invariants for generated anchors (gate, not retreat).

The anchor format has one structural contract, stated in
:mod:`ard.core.types`: a conversation starts with ``user``, ends with
``user`` and alternates roles strictly.  ``AnchorSpec`` enforces it on the
way *in*, but the entry gate alone never protected the output — that is how
``UAUAU``-shaped anchors reached a published anchor bank.  This module holds
the single implementation of that contract so entry and exit checks can never
drift apart (§1.4 单一真相源).

It is a boundary check, not a fallback (§2.3): a violation means the anchor
must be discarded, loudly.
"""

from __future__ import annotations

from typing import Any

from ard.core.types import AnchorSpec

__all__ = ["expected_message_roles", "message_shape_error"]


def expected_message_roles(spec: AnchorSpec) -> list[str]:
    """Return the role sequence ``messages`` must have for *spec*.

    One message per ``TurnSpec``: user turns are produced by the input
    generator / with logprobs, assistant turns by the target model.  Because
    every turn yields exactly one message, the expected sequence is simply
    the declared turn roles.  ``AnchorSpec`` already guarantees this
    sequence starts with ``user``, ends with ``user`` and alternates.

    Args:
        spec: Anchor specification.

    Returns:
        List of ``"user"`` / ``"assistant"`` strings, one per turn.
    """
    return [turn.role for turn in spec.turns]


def message_shape_error(messages: list[Any]) -> str | None:
    """Validate the structural shape of a message list.

    The anchor format invariants (``src/ard/core/types.py``) are:
    first message ``user``, last message ``user``, roles strictly
    alternating.  ``AnchorSpec`` enforces this at the *entry* boundary;
    this function exists so the same contract can be checked at the
    *exit* boundary, before an anchor is handed to persistence.

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
    if any(r not in ("user", "assistant") for r in roles):
        index = next(i for i, r in enumerate(roles) if r not in ("user", "assistant"))
        return f"message {index} has unknown role {roles[index]!r}"
    if roles[0] != "user":
        return f"first message role must be 'user', got {roles[0]!r}"
    if roles[-1] != "user":
        return f"last message role must be 'user', got {roles[-1]!r}"
    for i in range(len(roles) - 1):
        if roles[i] == roles[i + 1]:
            return (
                f"messages {i} and {i + 1} share role {roles[i]!r} "
                f"(roles must alternate)"
            )
    return None
