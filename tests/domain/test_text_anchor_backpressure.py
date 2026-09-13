"""Tests for anchor-generation backpressure and per-run failure accounting (WP-F5).

Two defects are frozen here:

* **Unreachable backpressure.**  ``_generate_one_anchor`` used to catch
  ``ARDTimeoutError`` per turn and ``return None``, so the counter in
  ``generate_text_anchors`` could never increment and the cooldown never ran
  (two WARNINGs lived in that dead code path).  The tests below fail if a turn
  failure stops reaching the scheduler again.
* **Failures visible only in logs.**  The counters that say "this run dropped
  anchors / had no log-probs / triggered cooldowns" are asserted here to reach
  ``AnchorGenerationStats`` and hence ``manifest.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

from ard.backends.api_client import (
    ARDEmptyContentError,
    ARDLogprobsError,
    ARDTimeoutError,
    ChatAPIStats,
)
from ard.core.types import AnchorSpec, GeneratedAnchor, TurnSpec
from ard.domain.bank import (
    AppendOutcome,
    append_anchor,
    build_manifest_from_records,
    with_generation_report,
)
from ard.domain.text_anchor import (
    AnchorGenerationStats,
    failure_reason,
    generate_text_anchors,
    is_server_instability,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


class _ScriptedClient:
    """A ``ChatAPIClient`` double driven by per-call outcomes.

    Each outcome is either a ``str`` (returned as content), an ``Exception``
    instance (raised), or a ``dict`` (returned whole, for
    ``chat_with_logprobs``).  Outcomes are consumed in call order; the last one
    repeats, so a scripted client is written as "3 timeouts then an answer"
    rather than as a fixed call count.
    """

    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[str, dict]] = []

    def _next(self, kind: str) -> object:
        if not self._outcomes:
            raise AssertionError(f"{kind} called with no scripted outcome left")
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        self.calls.append((kind, {}))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        return str(self._next("chat"))

    def chat_with_logprobs(
        self, messages: list[dict], temperature: float | None = None
    ) -> dict:
        outcome = self._next("chat_with_logprobs")
        if not isinstance(outcome, dict):
            raise AssertionError("chat_with_logprobs needs a dict outcome")
        return outcome


def _ok_logprobs(content: str = "the final target answer") -> dict:
    return {
        "content": content,
        "logprobs": {"token_ids": ["the", " final"], "log_probs": [-0.1, -0.2]},
    }


def _spec(spec_id: str) -> AnchorSpec:
    """A single-turn spec (one user turn, answered with log-probs)."""
    return AnchorSpec(
        id=spec_id,
        anchor_meta={"language": "English", "knowledge_domain": "geography"},
        turns=[
            TurnSpec(
                turn_index=0,
                role="user",
                generation_instruction="Ask about geography",
                is_final=True,
            )
        ],
        input_generator_id="input-gen",
    )


def _specs(count: int) -> list[AnchorSpec]:
    return [_spec(f"bp_{index}") for index in range(count)]


class _SleepSpy:
    """Records cooldown pauses instead of sleeping through them."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _run(specs: list[AnchorSpec], input_outcomes, target_outcomes, **kwargs):
    """Run the generator with scripted clients and collect the warnings."""
    stats = kwargs.pop("stats", AnchorGenerationStats())
    spy = _SleepSpy()
    input_client = _ScriptedClient(input_outcomes)
    target_client = _ScriptedClient(target_outcomes)
    anchors = generate_text_anchors(
        specs=specs,
        input_client=input_client,  # type: ignore[arg-type]
        target_client=target_client,  # type: ignore[arg-type]
        input_model_name="input-model",
        target_model_name="target-model",
        concurrency=1,
        stats=stats,
        sleep=spy,
        disable_progress=True,
        **kwargs,
    )
    return anchors, stats, spy


# ── Failure classification ───────────────────────────────────────────────────


def test_timeout_and_transport_are_server_instability() -> None:
    """Only timeout/transport failures justify a cooldown."""
    assert failure_reason(ARDTimeoutError("timed out")) == "timeout"
    assert failure_reason(httpx.ConnectError("connection refused")) == "transport_error"
    assert is_server_instability(ARDTimeoutError("timed out")) is True
    assert is_server_instability(
        httpx.ConnectError("connection refused")
    ) is True


def test_model_output_failures_are_not_server_instability() -> None:
    """Empty content / missing log-probs are deterministic — sleeping cannot fix them."""
    stats = ChatAPIStats(content_chars=0, reasoning_chars=120)
    empty = ARDEmptyContentError("reasoning ate the budget", stats)
    missing = ARDLogprobsError("no log-probs", reason="key_missing")
    assert failure_reason(empty) == "empty_content"
    assert failure_reason(missing) == "logprobs_error"
    assert is_server_instability(empty) is False
    assert is_server_instability(missing) is False
    # An unexpected bug must not be disguised as server load.
    assert failure_reason(ValueError("boom")) == "unexpected_error"
    assert is_server_instability("unexpected_error") is False


# ── Backpressure ─────────────────────────────────────────────────────────────


def test_consecutive_timeouts_trigger_cooldown_and_reset_counter(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """3 consecutive timeouts → one cooldown pause, a WARNING, and a reset counter."""
    stats = AnchorGenerationStats()
    with caplog.at_level(logging.WARNING):
        anchors, stats, spy = _run(
            _specs(3),
            [ARDTimeoutError("timed out")],
            [_ok_logprobs()],
            backpressure_threshold=3,
            backpressure_cooldown=60.0,
            stats=stats,
        )

    assert anchors == []
    assert spy.calls == [60.0], "the cooldown must actually be entered once"
    assert stats.backpressure_events == 1
    assert stats.abandoned_by_reason == {"timeout": 3}
    assert stats.abandoned_total == 3
    # The counter is reset after the cooldown, so the run's final value is 0.
    assert stats.consecutive_server_failures == 0

    # Both WARNINGs that used to live in unreachable code are asserted here.
    warnings = [record.getMessage() for record in caplog.records]
    assert any(
        "Backpressure triggered: 3 consecutive server failures" in m for m in warnings
    ), warnings
    assert any("pausing generation for 60.0s" in m for m in warnings), warnings
    assert any("resetting the" in m and "resuming generation" in m for m in warnings), warnings
    # Even below the threshold the per-anchor streak is observable in the log.
    assert any(
        "Anchor abandoned (timeout)" in m and "consecutive server failure(s)" in m
        for m in warnings
    ), warnings


def test_transport_error_counts_towards_backpressure() -> None:
    """A connection failure is server instability, exactly like a timeout."""
    anchors, stats, spy = _run(
        _specs(2),
        [httpx.ConnectError("connection refused")],
        [_ok_logprobs()],
        backpressure_threshold=2,
        backpressure_cooldown=5.0,
    )
    assert anchors == []
    assert spy.calls == [5.0]
    assert stats.backpressure_events == 1
    assert stats.abandoned_by_reason == {"transport_error": 2}


def test_backpressure_fires_again_after_counter_reset(caplog: pytest.LogCaptureFixture) -> None:
    """Once reset, the counter starts over — the next N failures pause again."""
    with caplog.at_level(logging.WARNING):
        _, stats, spy = _run(
            _specs(6),
            [ARDTimeoutError("timed out")],
            [_ok_logprobs()],
            backpressure_threshold=3,
            backpressure_cooldown=1.0,
        )
    assert stats.backpressure_events == 2, "4x would be wrong: the counter resets"
    assert spy.calls == [1.0, 1.0]


def test_model_output_failures_never_start_a_cooldown() -> None:
    """Empty content / missing log-probs abandon anchors but do not pause the run."""
    stats = ChatAPIStats(content_chars=0, reasoning_chars=64)
    _, stats, spy = _run(
        _specs(3),
        # spec 0: the input generator returns nothing usable.
        ["a user question"],
        # spec 1 and 2: the final turn carries no log-probs.
        [_ok_logprobs(), ARDLogprobsError("no log-probs", reason="key_missing")],
        backpressure_threshold=2,
        backpressure_cooldown=60.0,
    )
    assert spy.calls == [], "a deterministic model-output failure must not sleep"
    assert stats.backpressure_events == 0
    assert stats.abandoned_by_reason == {"logprobs_error": 2}
    assert stats.succeeded == 1

    # Now the empty-content path, which abandons every anchor.
    _, empty_stats, empty_spy = _run(
        _specs(2),
        [ARDEmptyContentError("no content", stats)],
        [_ok_logprobs()],
        backpressure_threshold=1,
        backpressure_cooldown=60.0,
    )
    assert empty_spy.calls == []
    assert empty_stats.abandoned_by_reason == {"empty_content": 2}


def test_empty_answer_is_counted_but_does_not_pause() -> None:
    """A too-short answer is a soft abandon: counted, no cooldown."""
    _, stats, spy = _run(
        _specs(3),
        ["a user question"],
        [_ok_logprobs("x")],
        backpressure_threshold=1,
        backpressure_cooldown=60.0,
        min_answer_chars=8,
    )
    assert spy.calls == []
    assert stats.abandoned_by_reason == {"answer_too_short": 3}


def test_success_between_failures_resets_the_streak() -> None:
    """A produced anchor clears the streak, so 2+2 failures stay below 3."""
    _, stats, spy = _run(
        _specs(5),
        [
            ARDTimeoutError("timed out"),
            ARDTimeoutError("timed out"),
            "a user question",
            ARDTimeoutError("timed out"),
            ARDTimeoutError("timed out"),
        ],
        [_ok_logprobs()],
        backpressure_threshold=3,
        backpressure_cooldown=60.0,
    )
    assert spy.calls == [], "a success in between must clear the counter"
    assert stats.backpressure_events == 0
    assert stats.abandoned_by_reason == {"timeout": 4}
    assert stats.succeeded == 1
    # The run ends on a streak of 2 — below threshold, so it stays visible.
    assert stats.consecutive_server_failures == 2


def test_abandoned_anchor_is_reported_with_its_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The failure is logged with its traceback instead of vanishing into a None."""
    with caplog.at_level(logging.WARNING):
        _run(
            _specs(1),
            [ARDTimeoutError("timed out")],
            [_ok_logprobs()],
            backpressure_threshold=99,
        )
    records = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert records, "an abandoned anchor must be announced"
    assert any(record.exc_info is not None for record in records), (
        "the traceback of the abandoning turn must reach the log"
    )


# ── Stats / manifest plumbing ────────────────────────────────────────────────


def test_zero_requested_run_reports_zeros() -> None:
    """A run with no specs still reports a consistent, all-zero accounting."""
    anchors, stats, spy = _run([], ["a user question"], [_ok_logprobs()])
    assert anchors == []
    assert spy.calls == []
    assert stats.requested == 0
    assert stats.abandoned_total == 0
    assert stats.to_manifest_dict()["written"] == 0


def test_stats_count_written_rejected_and_duplicate(tmp_path: Path) -> None:
    """Bank rejections are counted: 1 written, 1 duplicate id."""
    output_path = tmp_path / "anchor_bank.jsonl"
    # Same id twice: the second offer must be refused by the id-uniqueness gate.
    specs = [_spec("dup_id"), _spec("dup_id"), _spec("fresh_id")]
    _, stats, _ = _run(
        specs,
        ["a user question"],
        [_ok_logprobs()],
        output_path=output_path,
    )
    assert stats.requested == 3
    assert stats.succeeded == 3
    assert stats.written == 2
    assert stats.duplicate_ids == 1
    assert stats.rejected_invalid_shape == 0
    assert stats.abandoned_total == 0


def test_shape_gate_rejection_is_recorded() -> None:
    """The bank's own shape gate refuses a malformed anchor and says so."""
    broken = GeneratedAnchor(
        id="broken",
        messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "q2"},
            {"role": "user", "content": "q3"},
        ],
        target_answer="a",
        target_model="target-model",
        input_generator_model="input-model",
        anchor_meta={},
    )
    assert append_anchor(broken, Path("/nonexistent/never-written.jsonl")) is (
        AppendOutcome.INVALID_SHAPE_SKIPPED
    )


def test_manifest_records_every_failure_signal(tmp_path: Path) -> None:
    """manifest.json carries rejected / duplicate / logprobs / empty / backpressure counts."""
    records = [
        {
            "id": "anchor_a",
            "anchor_meta": {"language": "English", "knowledge_domain": "math"},
        }
    ]
    manifest = build_manifest_from_records(records, tmp_path)
    stats = AnchorGenerationStats(
        requested=10,
        succeeded=7,
        written=7,
        abandoned_total=3,
        abandoned_by_reason={"timeout": 2, "empty_content": 1},
        rejected_invalid_shape=2,
        duplicate_ids=1,
        backpressure_events=1,
    )
    with_generation_report(
        manifest,
        stats=stats.to_manifest_dict(),
        failures={
            "key_missing": 4,
            "partial": 1,
            "empty_content": 3,
            "reasoning_only_responses": 3,
            "truncated_empty": 3,
        },
    )
    write_path = tmp_path / "manifest.json"
    write_path.write_text(json.dumps(manifest), encoding="utf-8")
    published = json.loads(write_path.read_text(encoding="utf-8"))

    generation = published["generation"]
    assert generation["counters"]["rejected_invalid_shape"] == 2
    assert generation["counters"]["duplicate_ids"] == 1
    assert generation["counters"]["backpressure_events"] == 1
    assert generation["counters"]["abandoned_by_reason"] == {
        "empty_content": 1,
        "timeout": 2,
    }
    assert generation["failures"]["key_missing"] == 4
    assert generation["failures"]["empty_content"] == 3
    assert generation["failures"]["reasoning_only_responses"] == 3
    # The pre-existing fields are untouched.
    assert published["total_anchors"] == 1
    assert published["domains"] == {"math": 1}


def test_manifest_omits_zero_valued_counters(tmp_path: Path) -> None:
    """Zero-valued counters are dropped; a fully clean run publishes nothing."""
    records = [
        {
            "id": "anchor_a",
            "anchor_meta": {"language": "English", "knowledge_domain": "math"},
        }
    ]
    manifest = build_manifest_from_records(records, tmp_path)
    with_generation_report(
        manifest,
        stats={"requested": 1, "written": 1, "duplicate_ids": 0, "backpressure_events": 0},
        failures={"key_missing": 0},
    )
    assert manifest["generation"] == {"counters": {"requested": 1, "written": 1}}

    # A run with nothing to report gains no field at all.
    clean = build_manifest_from_records(records, tmp_path)
    with_generation_report(clean, stats={"requested": 0}, failures={})
    assert "generation" not in clean

    # Backward compatibility: a manifest written before this field existed must
    # still be consumable — readers must treat the field as optional.
    legacy = build_manifest_from_records(records, tmp_path)
    assert "generation" not in legacy
    assert legacy["total_anchors"] == 1
    assert legacy.get("generation") is None


def test_manifest_generation_report_is_idempotent(tmp_path: Path) -> None:
    """Re-attaching a report replaces it instead of nesting a second copy."""
    manifest = build_manifest_from_records([], tmp_path)
    with_generation_report(manifest, stats={"written": 1})
    with_generation_report(manifest, stats={"written": 2})
    assert manifest["generation"] == {"counters": {"written": 2}}
