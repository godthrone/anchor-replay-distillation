"""End-of-run summary warning for :mod:`ard.pipeline`.

Responsibility: freeze the one operator-facing WARNING a short run must end
with — a dataset holding fewer anchors than its plan is announced on the
terminal as well as in the logs (§3.2 透明退路) — and that a complete run stays
quiet, so the warning keeps meaning something.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ard.logging import get_logger
from ard.pipeline import _warn_if_run_is_short

#: The channel the end-of-run accounting logs through (see ``ard.logging``).
OPERATOR_LOGGER = "ard.operator"


def _manifest(planned: int, written: int, reasons: dict[str, int] | None = None) -> dict:
    """A manifest shaped like the ones ``run`` builds, with only what the check reads."""
    counters: dict = {}
    if reasons:
        counters = {"abandoned_total": sum(reasons.values()), "abandoned_by_reason": reasons}
    return {
        "plan": {"planned_anchors": planned, "written_anchors": written},
        "generation": {"counters": counters},
    }


def _seen(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == OPERATOR_LOGGER]


def _write_readout(output_dir: Path) -> None:
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    (output_dir / "results" / "coverage.json").write_text("{}", encoding="utf-8")


def test_a_short_run_warns_with_the_counts_and_the_readout_pointer(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The operator gets written/planned, the reasons, the shortfall and the fix."""
    output_dir = tmp_path / "run"
    _write_readout(output_dir)

    with caplog.at_level(logging.WARNING, logger=OPERATOR_LOGGER):
        _warn_if_run_is_short(
            _manifest(100, 94, {"empty_content": 5, "transport_error": 1}), output_dir
        )

    messages = _seen(caplog)
    assert len(messages) == 1, messages
    text = messages[0]
    assert "INCOMPLETE RUN" in text
    assert "94/100 planned anchor(s) written" in text
    assert "abandoned 6" in text
    assert "empty_content=5, transport_error=1" in text
    assert "6 planned coordinate(s) missing" in text
    assert "results/coverage.json" in text
    assert "Re-run the same command" in text


def test_a_short_run_warns_even_without_the_acceptance_readout(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The dataset is short whether or not ``[coverage] enabled`` is set."""
    with caplog.at_level(logging.WARNING, logger=OPERATOR_LOGGER):
        _warn_if_run_is_short(_manifest(10, 8), tmp_path)

    messages = _seen(caplog)
    assert len(messages) == 1, messages
    assert "8/10 planned anchor(s) written" in messages[0]
    assert "2 planned coordinate(s) missing" in messages[0]
    # No readout on disk, so no pointer to one — but the warning still fires.
    assert "results/coverage.json" not in messages[0]


def test_a_complete_run_warns_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """No shortfall, no abandoned anchor ⇒ silence (a warning always on is not read)."""
    with caplog.at_level(logging.WARNING, logger=OPERATOR_LOGGER):
        _warn_if_run_is_short(_manifest(100, 100), tmp_path)

    assert _seen(caplog) == []


def test_the_warning_is_printed_on_the_terminal(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """The terminal copy is the point of the channel, in the console's own format."""
    logger = logging.getLogger(OPERATOR_LOGGER)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)  # bind a handler to *this* test's stdout
    try:
        _warn_if_run_is_short(_manifest(100, 94, {"empty_content": 6}), tmp_path)
        captured = capfd.readouterr().out
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    assert "WARNING: INCOMPLETE RUN: 94/100 planned anchor(s) written" in captured
    assert "empty_content=6" in captured
    # The same record propagates to the ``ard`` namespace, where the run's file
    # handlers live — ``configure_file_logging`` creates that logger at run start.
    namespace = logging.getLogger("ard")
    assert get_logger(OPERATOR_LOGGER).parent is namespace
