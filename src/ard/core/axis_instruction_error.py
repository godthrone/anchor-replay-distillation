"""Axis-instruction wording failure: the one exception its contract raises.

``AxisInstructionError`` is the single failure type of the axis-instruction
wording contract.  By §12.2 class-path mirroring its file name is the class
name in snake_case; ``ard.core.axis_instruction`` re-exports it, so callers are
unaffected by the split.
"""

__all__ = ["AxisInstructionError"]


class AxisInstructionError(RuntimeError):
    """Raised when an axis's wording file is missing, empty or malformed.

    The message always names the file and the axis value it expected.  It never
    contains credential material: a wording path and an axis value are not
    secret (§15).
    """
