"""Outcome vocabulary of a bank append attempt.

This module owns the enum that :func:`ard.domain.bank.append_anchor` returns.
It lives in its own module so that the class name and the file name mirror each
other exactly (§12.2 类名与路径互映): ``AppendOutcome`` ↔ ``append_outcome.py``.
An ``Enum`` is not one of the §12.2 data-container exemption cases (the
exemption text lists frozen dataclasses and pydantic models only), so the
function-family attachment argument does not apply and the class is filed
under its own name.

Consumers import :class:`AppendOutcome` from here — ``ard.domain.bank`` does
**not** re-export it (§18.1 不留负债: no compatibility shim).
"""

from enum import Enum


class AppendOutcome(Enum):
    """What happened when an anchor was offered to the bank."""

    APPENDED = "appended"
    """The anchor was written to the bank."""

    DUPLICATE_SKIPPED = "duplicate_skipped"
    """An anchor with this id is already in the bank — nothing was written."""

    INVALID_SHAPE_SKIPPED = "invalid_shape_skipped"
    """The anchor's messages violated the conversation-shape contract."""

    INVALID_DATA_SOURCE_SKIPPED = "invalid_data_source_skipped"
    """The anchor's ``data_source`` was outside the controlled vocabulary."""
