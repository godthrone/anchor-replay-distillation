"""Checkpoint/resume contract of ``pipeline.run`` and the bank it resumes from (R5).

User requirement under test, verbatim:

    "不需要跨 run 覆盖，每个 run 独立，不过之前支持断点继续，如果 output
     目录下面是未完成的应该支持继续。"

Translated into the three behaviours frozen here:

1. an output directory holding fewer records than the rule-derived plan
   resumes: only the plan coordinates whose stable id is **absent from the bank**
   are asked for (identity, not record count — a bank missing a middle
   coordinate is not a short prefix of the plan), they are **appended** to the
   bank, and the records already on disk survive byte-for-byte;
2. an output directory that already satisfies the plan generates nothing at all
   and returns without touching the bank (idempotent re-run) — unless it holds
   anchor ids outside the plan, which is a different dataset and is refused
   (section 2, S22);
3. ``output.overwrite`` is the explicit opt-in that clears the bank (§3.3
   预授权退路) — without it, a second run never destroys the first one's records.

Plus the failure mode that made (1) untrustworthy: an append interrupted mid-line
left a fragment without its trailing newline, after which the record *count* and
the record *reader* disagreed and every following record was concatenated onto the
fragment.  Sections 5–7 pin that down at the bank level; section 7 is keyed on the
**plan identity** recorded in ``manifest.json`` (S22), never on ``[generation]
seed`` — a seed is a process-level config value that does not name a plan.

Scope note: the pipeline half is deliberately *not* a second sampler test.  ``run``
is exercised with the spec plan replaced by a small deterministic double (the
``generate_specs`` seam of ``pipeline.run``), so what the assertions read is the
resume arithmetic and the bank on disk — not the v4 construction rule (covered by
``tests/core/test_sampling.py``) and not the turn generator (covered by
``tests/domain/test_text_anchor_backpressure.py``).  Since WP-S2a the target is
``len(plan)`` — derived from the ontology — so the double *is* the plan, and the
pipeline decides the shortfall by comparing the plan's ids against the bank's.  No
network call is reachable: the generator double never touches the API clients the
pipeline constructs, and ``api_base`` points at a closed local port.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from ard.core.types import AnchorSpec, GeneratedAnchor, TurnSpec
from ard.domain.bank import (
    append_anchor,
    count_existing_anchors,
    count_unique_anchor_ids,
    read_anchor_bank,
)
from ard.domain.text_anchor import AnchorGenerationStats

#: Plan size used by the "incomplete bank" cases: small enough to read at a
#: glance, large enough that "2 existing + 3 missing" is not a coincidence of
#: the bank.  The production plan is 1,826 entries (WP-S2a contract test), so a
#: five-entry plan is also the guard that resume never assumes a hard-coded size.
_PLAN_SIZE = 5
_TARGET_COUNT = _PLAN_SIZE


def _anchor(anchor_id: str, *, answer: str | None = None) -> GeneratedAnchor:
    """A minimal anchor that satisfies the bank's exit gates (UAU shape)."""
    return GeneratedAnchor(
        id=anchor_id,
        messages=[{"role": "user", "content": f"question {anchor_id}"}],
        target_answer=answer or f"answer {anchor_id}",
        target_model="target-model",
        input_generator_model="input-model",
        anchor_meta={"language": "English", "knowledge_domain": "math"},
        reasoning=None,
    )


def _plan() -> list[AnchorSpec]:
    """A small deterministic plan — the pipeline hands its tail to the generator.

    It stands in for the rule-derived 1,826-coordinate plan; ``_PLAN_SIZE`` is
    what the resume arithmetic is measured against.  The ids are indexed by
    *plan position*; resume is keyed on these ids, so a test can name exactly
    which coordinate a previous run left unfinished.
    """
    return [
        AnchorSpec(
            id=f"resumed-{index}",
            anchor_meta={"language": "English", "knowledge_domain": "math"},
            turns=[TurnSpec(turn_index=0, role="user", is_final=True)],
        )
        for index in range(_PLAN_SIZE)
    ]


def _write_config(path: Path, output_dir: Path, *, overwrite: bool, seed: int = 7) -> None:
    """Minimal valid config; the API endpoints are dummies that are never called."""
    lines = [
        "[input_generator]",
        'api_base = "http://127.0.0.1:1/v1"',
        'model_name = "input-model"',
        'api_key = "unused"',
        "",
        "[target_model]",
        'api_base = "http://127.0.0.1:1/v1"',
        'model_name = "target-model"',
        'api_key = "unused"',
        "",
        "[generation]",
        # Neither the count nor the turn counts are config fields: the plan size
        # and each entry's turns come from the ontology's construction rule, and
        # the plan double below supplies the size here.
        f"seed = {seed}",
        "concurrency = 1",
        "",
        "[ontology]",
        'path = "unused.json"',
        "",
        "[output]",
        f'directory = "{output_dir}"',
        f"overwrite = {'true' if overwrite else 'false'}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


class _RunSpy:
    """Records what the pipeline asked for and appends one real record per spec."""

    def __init__(self, tag: str, abandoned: set[str] | None = None) -> None:
        self.tag = tag
        self.abandoned = abandoned or set()
        self.requested: list[int] = []
        self.requested_ids: list[list[str]] = []

    def __call__(self, **kwargs: object) -> list[GeneratedAnchor]:
        specs = kwargs["specs"]
        output_path = kwargs["output_path"]
        stats = kwargs["stats"]
        assert isinstance(specs, list)
        assert isinstance(output_path, Path)
        assert isinstance(stats, AnchorGenerationStats)
        self.requested.append(len(specs))
        self.requested_ids.append([spec.id for spec in specs if isinstance(spec, AnchorSpec)])
        written: list[GeneratedAnchor] = []
        for spec in specs:
            assert isinstance(spec, AnchorSpec)
            if spec.id in self.abandoned:
                continue  # the generation failure this test needs to reproduce
            anchor = _anchor(spec.id, answer=f"answer from {self.tag}")
            outcome = append_anchor(anchor, output_path)
            assert outcome.value == "appended", (
                f"a spec the pipeline asked for was refused by the bank: {outcome}"
            )
            written.append(anchor)
        stats.requested = len(specs)
        stats.written = len(written)
        stats.succeeded = len(written)
        return written


def _resume_rig(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    existing: int,
    overwrite: bool = False,
    tag: str = "resumed",
    abandoned: set[str] | None = None,
) -> tuple[Path, Path, _RunSpy, Callable[[object], list[AnchorSpec]]]:
    """Prepare an output dir with *existing* records and return the run rig.

    The fourth element is the plan double the caller must hand to ``run``
    through its ``generate_specs`` seam.
    """
    from ard import pipeline

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)
    bank = output_dir / "anchor_bank.jsonl"
    # The records a previous run wrote carry the *plan's* ids: resume is decided
    # by coordinate identity (``plan - bank``), so a bank whose ids are not in
    # the plan is a foreign bank, not a short prefix of this plan.  Positions
    # beyond the plan stand in for "the bank holds extra records".
    plan_ids = [spec.id for spec in _plan()]
    for index in range(existing):
        record_id = plan_ids[index] if index < len(plan_ids) else f"overfill-{index}"
        append_anchor(_anchor(record_id), bank)

    config_path = tmp_path / "config.toml"
    _write_config(config_path, output_dir, overwrite=overwrite)

    spy = _RunSpy(tag, abandoned=abandoned)
    # The ontology is never read: the plan double is handed to ``run`` through
    # its ``generate_specs`` seam, so the expensive v4 load is skipped.
    monkeypatch.setattr(
        pipeline,
        "sample_specs",
        lambda config: pytest.fail("the plan double must replace the v4 sampler"),
    )
    monkeypatch.setattr(pipeline, "generate_text_anchors", spy)
    return output_dir, bank, spy, (lambda config: _plan())


def _records(bank: Path) -> list[dict]:
    """Parse the bank the way a consumer does: unreadable lines are not records."""
    records: list[dict] = []
    for line in bank.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


#: What an append looks like when the process died inside ``write()``: valid JSON
#: so far, no closing brace, no newline.  Written by hand rather than by patching
#: ``write`` because the test must model the *outcome* of an interruption, not one
#: particular way of producing it.
_FRAGMENT = '{"id": "interrupted", "source": "ard", "messages": [{"role": "user"'


def _append_fragment(bank: Path) -> None:
    with bank.open("a", encoding="utf-8") as f:
        f.write(_FRAGMENT)


# ── 5. line-boundary tolerance (the bank level) ─────────────────────────────


def test_fragment_does_not_swallow_the_records_appended_after_it(tmp_path: Path) -> None:
    """Direct regression: a fragment must not absorb the next records (bug 2).

    Without the line-boundary repair in :func:`append_anchor`, the next records
    were concatenated onto the newline-less fragment and became part of one
    unparseable line — they were written but never read back.
    """
    bank = tmp_path / "bank.jsonl"
    append_anchor(_anchor("before"), bank)
    _append_fragment(bank)
    append_anchor(_anchor("after-1"), bank)
    append_anchor(_anchor("after-2"), bank)

    assert [r["id"] for r in read_anchor_bank(bank)] == ["before", "after-1", "after-2"]
    assert count_existing_anchors(bank) == 3


def test_append_keeps_the_fragment_on_disk(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The fragment is separated, announced — and *not* erased (§2.4)."""
    bank = tmp_path / "bank.jsonl"
    append_anchor(_anchor("before"), bank)
    _append_fragment(bank)

    with caplog.at_level("WARNING"):
        append_anchor(_anchor("after"), bank)

    assert any("did not end with a newline" in record.getMessage() for record in caplog.records)
    lines = bank.read_text(encoding="utf-8").splitlines()
    assert _FRAGMENT in lines, "the evidence of the interrupted write still exists"
    assert len(lines) == 3
    assert [r["id"] for r in read_anchor_bank(bank)] == ["before", "after"]


def test_fragment_is_not_counted_as_an_anchor(tmp_path: Path) -> None:
    """Reader and counter must agree on what a record is (bug 1)."""
    bank = tmp_path / "bank.jsonl"
    append_anchor(_anchor("before"), bank)
    _append_fragment(bank)

    # The pre-fix counter returned 2 here and read_anchor_bank raised, so the
    # resume path computed a shortfall from a line it could not read.
    assert count_existing_anchors(bank) == 1
    assert count_unique_anchor_ids(bank) == 1
    assert [r["id"] for r in read_anchor_bank(bank)] == ["before"]


def test_mid_file_corruption_is_skipped_and_the_rest_is_read(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A damaged line costs exactly that line — the other records are still usable."""
    bank = tmp_path / "bank.jsonl"
    append_anchor(_anchor("first"), bank)
    with bank.open("a", encoding="utf-8") as f:
        f.write("{not json at all}\n")
    append_anchor(_anchor("second"), bank)

    with caplog.at_level("WARNING"):
        records = read_anchor_bank(bank)

    assert [r["id"] for r in records] == ["first", "second"]
    assert count_existing_anchors(bank) == 2
    assert any("Ignoring unreadable line" in record.getMessage() for record in caplog.records)


def test_an_empty_bank_is_still_a_bank(tmp_path: Path) -> None:
    """The repair must not invent a leading newline on an empty/missing file."""
    bank = tmp_path / "bank.jsonl"
    append_anchor(_anchor("only"), bank)
    assert bank.read_text(encoding="utf-8").startswith('{"id": "only"')
    assert count_existing_anchors(tmp_path / "missing.jsonl") == 0
    assert read_anchor_bank(tmp_path / "missing.jsonl") == []


# ── 1. unfinished run → resume ───────────────────────────────────────────────


def test_incomplete_bank_generates_only_the_missing_anchors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N records on disk + a plan of M → exactly M-N new records, appended."""
    from ard.config import load_config
    from ard.pipeline import run

    existing = 2
    output_dir, bank, spy, plan = _resume_rig(tmp_path, monkeypatch, existing=existing)
    before = bank.read_text(encoding="utf-8")

    run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    assert spy.requested == [_TARGET_COUNT - existing], (
        "the run must ask for the shortfall, not for the full target again"
    )
    records = _records(bank)
    assert len(records) == _TARGET_COUNT
    assert [r["id"] for r in records[:existing]] == [
        f"resumed-{index}" for index in range(existing)
    ], "the pre-existing records must not be reordered or rewritten"
    assert [r["id"] for r in records[existing:]] == [
        f"resumed-{index}" for index in range(existing, _TARGET_COUNT)
    ]
    assert bank.read_text(encoding="utf-8").startswith(before), (
        "resume must append: the bytes of the previous run are a prefix of the new bank"
    )


def test_resume_does_not_touch_the_clients_before_generating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resume path performs no network call — the generator is the only caller."""
    from ard.config import load_config
    from ard.pipeline import run

    _, bank, spy, plan = _resume_rig(tmp_path, monkeypatch, existing=3)

    def _forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("the resume path must not reach the API clients")

    monkeypatch.setattr("ard.backends.api_client.ChatAPIClient.chat", _forbidden)
    run(load_config(tmp_path / "config.toml"), generate_specs=plan)
    assert spy.requested == [_TARGET_COUNT - 3]
    assert count_existing_anchors(bank) == _TARGET_COUNT


# ── 2. finished run → nothing to do ─────────────────────────────────────────


def test_complete_bank_is_left_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """N ≥ M → no generation, no rewrite, same bytes (a re-run is idempotent)."""
    from ard.config import load_config
    from ard.pipeline import run

    output_dir, bank, spy, plan = _resume_rig(tmp_path, monkeypatch, existing=_TARGET_COUNT)
    before = bank.read_text(encoding="utf-8")

    result_dir = run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    assert spy.requested == [], "a satisfied bank must not trigger generation"
    assert bank.read_text(encoding="utf-8") == before
    assert count_existing_anchors(bank) == _TARGET_COUNT
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_anchors"] == _TARGET_COUNT


def test_overfilled_bank_with_foreign_records_is_refused_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An over-filled bank is unverifiable, not "nothing to do" (S22).

    More records than the plan means the bank necessarily holds anchor ids that
    are *not* in this run's plan.  Rewriting this run's plan identity over such
    a bank would make ``manifest.json`` describe a plan the bank does not hold —
    exactly the divergence the identity guard exists to prevent.  The refusal
    happens before any side effect, so the bank is byte-identical afterwards.
    """
    from ard.config import ConfigError, load_config
    from ard.pipeline import run

    _, bank, spy, plan = _resume_rig(tmp_path, monkeypatch, existing=_TARGET_COUNT + 2)
    before = bank.read_text(encoding="utf-8")

    with pytest.raises(ConfigError, match="not part of this run's plan"):
        run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    assert spy.requested == []
    assert bank.read_text(encoding="utf-8") == before


# ── 3. overwrite is the explicit opt-in (§3.3 预授权退路) ───────────────────


def test_overwrite_clears_the_bank_and_generates_the_full_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing earlier records only happens when ``output.overwrite = true``."""
    from ard.config import load_config
    from ard.pipeline import run

    existing = 2
    _, bank, spy, plan = _resume_rig(
        tmp_path, monkeypatch, existing=existing, overwrite=True, tag="fresh"
    )

    run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    assert spy.requested == [_TARGET_COUNT], (
        "overwrite starts from an empty bank, so the full plan is requested"
    )
    records = _records(bank)
    assert len(records) == _TARGET_COUNT
    # The plan double is positional, so clearing the bank regenerates the plan
    # from its first entry — the ids are a function of the plan, not of the run.
    assert [r["id"] for r in records] == [f"resumed-{index}" for index in range(_TARGET_COUNT)]


# ── 4. the interrupted-run fragment must not defeat the resume ──────────────


def test_trailing_fragment_is_ignored_instead_of_crashing_the_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A half-written last line is skipped, not fatal (regression).

    ``append_anchor`` flushes after every record, so an interrupted run leaves
    complete records plus at most one fragment.  Before the fix the counter
    counted the fragment (so the shortfall was computed against N+1) and
    ``read_anchor_bank`` raised on it — the resume path failed on exactly the
    file it exists to rescue.
    """
    from ard.config import load_config
    from ard.pipeline import run

    existing = 2
    _, bank, spy, plan = _resume_rig(tmp_path, monkeypatch, existing=existing)
    with bank.open("a", encoding="utf-8") as f:
        f.write('{"id": "interrupted", "messages": [{"role": "user"')  # no newline

    with caplog.at_level("WARNING"):
        run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    assert any("unreadable line" in record.getMessage() for record in caplog.records), (
        "the discarded fragment must be announced, not silently swallowed"
    )
    assert spy.requested == [_TARGET_COUNT - existing], (
        "the fragment is not a record, so it must not shrink the missing batch"
    )
    records = _records(bank)
    assert [r["id"] for r in records[:existing]] == [
        f"resumed-{index}" for index in range(existing)
    ]
    assert [r["id"] for r in records[existing:]] == [
        f"resumed-{index}" for index in range(existing, _TARGET_COUNT)
    ], "the records appended after the fragment must survive as separate lines"
    assert len(records) == _TARGET_COUNT, (
        "the fragment itself stays on disk and is simply never counted as an anchor"
    )


# ── 5. a bank that is still short after a resume stays append-only ──────────


def test_failed_resume_grows_the_bank_without_rewriting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even when the generator delivers less than asked, nothing is destroyed.

    The batch shrinks; it never becomes an overwrite.  The next run simply sees
    a smaller record count and asks again — this is the loop §3.1 同效退路
    describes (retry changes the cost, not the result), and it is why a resumed
    run that falls short must leave the bank intact.
    """
    from ard import pipeline
    from ard.config import load_config

    existing = 2
    output_dir, bank, spy, plan = _resume_rig(tmp_path, monkeypatch, existing=existing)
    before = bank.read_text(encoding="utf-8")

    # A generator that delivers one anchor no matter how many were requested.
    def _half_delivery(**kwargs: object) -> list[GeneratedAnchor]:
        specs = kwargs["specs"]
        stats = kwargs["stats"]
        output_path = kwargs["output_path"]
        assert isinstance(specs, list)
        assert isinstance(stats, AnchorGenerationStats)
        assert isinstance(output_path, Path)
        spy.requested.append(len(specs))
        anchor = _anchor("only-one")
        append_anchor(anchor, output_path)
        stats.requested = len(specs)
        stats.written = 1
        stats.succeeded = 1
        return [anchor]

    monkeypatch.setattr(pipeline, "generate_text_anchors", _half_delivery)
    pipeline.run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    assert spy.requested == [_TARGET_COUNT - existing]
    assert bank.read_text(encoding="utf-8").startswith(before)
    assert len(_records(bank)) == existing + 1


# ── 6. resume is keyed by coordinate identity, not by record count (F1) ─────


def _distinct_plan() -> list[AnchorSpec]:
    """A small plan whose entries are pairwise distinct *coordinates*.

    The readout test shrinks the construction rule's expected counts to this
    plan's size (see :func:`test_readout_reconciles_with_the_bank`), so the
    entries also have to be distinct for ``duplicate_coordinates`` to be 0.
    """
    return [
        AnchorSpec(
            id=f"resumed-{index}",
            anchor_meta={"language": "English", "knowledge_domain": f"domain-{index}"},
            turns=[TurnSpec(turn_index=0, role="user", is_final=True)],
        )
        for index in range(_PLAN_SIZE)
    ]


def test_resume_asks_for_a_middle_anchor_the_previous_run_abandoned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1: the shortfall is ``plan - bank`` by id, not ``plan[record_count:]``.

    Run 1 abandons the plan's *middle* entry.  A count-based resume sees 4
    records and asks for ``plan[4:]`` — the last entry, already on disk — so the
    abandoned coordinate is never retried.  An identity-based resume asks for
    exactly the coordinate the bank is missing.
    """
    from ard.config import load_config
    from ard.pipeline import run

    output_dir, bank, spy, plan = _resume_rig(
        tmp_path, monkeypatch, existing=0, abandoned={"resumed-2"}
    )
    run(load_config(tmp_path / "config.toml"), generate_specs=plan)
    assert spy.requested_ids == [["resumed-0", "resumed-1", "resumed-2", "resumed-3", "resumed-4"]]
    assert [r["id"] for r in _records(bank)] == [
        "resumed-0",
        "resumed-1",
        "resumed-3",
        "resumed-4",
    ], "the middle entry was abandoned, so the bank is NOT a prefix of the plan"

    resumed = _RunSpy("resumed")
    monkeypatch.setattr("ard.pipeline.generate_text_anchors", resumed)
    run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    assert resumed.requested_ids == [["resumed-2"]], (
        "the missing middle coordinate must be pending again; a count-based resume "
        "would ask for plan[4:] = resumed-4, which is already in the bank"
    )
    records = _records(bank)
    assert sorted(r["id"] for r in records) == sorted(
        f"resumed-{index}" for index in range(_PLAN_SIZE)
    ), "every planned coordinate must end up in the bank"
    assert records[-1]["id"] == "resumed-2", (
        "the recovered coordinate is appended at the end — resume is append-only, "
        "so the bank is not re-sorted into plan order"
    )
    assert (output_dir / "results" / "coverage.json").is_file()


def test_readout_reconciles_with_the_bank_and_names_the_missing_coordinate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1: ``coverage.json`` must never read green while the bank is short.

    The construction rule's expected counts are shrunk to the plan double's size
    so that ``within_rule`` would otherwise be *true* — the test then measures
    the bank reconciliation, not "the double is deliberately small".  Run 1
    abandons a middle entry; the readout has to report ``within_rule=false`` and
    name ``resumed-2``.  Run 2 fills it and the readout turns truthful again.
    """
    from ard.config import load_config
    from ard.core import sampling as sampling_module
    from ard.pipeline import run

    monkeypatch.setattr(sampling_module, "EXPECTED_TOTAL", _PLAN_SIZE)
    monkeypatch.setattr(sampling_module, "EXPECTED_TEXT_BLOCKS", 0)
    monkeypatch.setattr(sampling_module, "EXPECTED_IMAGE_BLOCKS", 0)
    monkeypatch.setattr(sampling_module, "EXPECTED_KNOWLEDGE_DOMAINS", _PLAN_SIZE)
    monkeypatch.setattr(sampling_module, "EXPECTED_VISUAL_DOMAINS", 0)

    def plan(_config: object) -> list[AnchorSpec]:
        return _distinct_plan()

    output_dir, bank, _spy, _ = _resume_rig(
        tmp_path, monkeypatch, existing=0, abandoned={"resumed-2"}
    )
    run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    coverage = json.loads((output_dir / "results" / "coverage.json").read_text(encoding="utf-8"))
    assert coverage["structure"]["within_rule"] is False, (
        "a bank short a planned coordinate must not be reported as within the rule"
    )
    assert any(
        "missing 1 planned coordinate" in warning and "resumed-2" in warning
        for warning in coverage["warnings"]
    ), "the readout must name the missing coordinate, not only flip the flag"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_anchors"] == _PLAN_SIZE - 1

    monkeypatch.setattr("ard.pipeline.generate_text_anchors", _RunSpy("resumed"))
    run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    coverage = json.loads((output_dir / "results" / "coverage.json").read_text(encoding="utf-8"))
    assert coverage["structure"]["within_rule"] is True
    assert not any("planned coordinate" in warning for warning in coverage["warnings"])
    assert len(_records(bank)) == _PLAN_SIZE


# ── 7. a resume that changes the *plan* is refused, not merged ──────────────
#
# B5 originally guarded on ``[generation] seed``.  S22 showed that key is wrong:
# a run directory's recorded seed is rewritten by every later invocation (even
# ones that generate nothing), so the recorded seed does not name the banked
# plan; and two runs can share a seed while sampling different plans.  The guard
# now compares the **plan identity** recorded in ``manifest.json``.  The tests
# below are the three cases: different plan ⇒ refused, different seed + same
# plan ⇒ allowed, and a legacy bank (no recorded identity) ⇒ structurally
# verified instead of assumed.


def _plan_tagged(tag: str) -> list[AnchorSpec]:
    """A plan double whose ids carry *tag*, so two plans cannot be confused."""
    return [
        AnchorSpec(
            id=f"{tag}-{index}",
            anchor_meta={"language": "English", "knowledge_domain": "math"},
            turns=[TurnSpec(turn_index=0, role="user", is_final=True)],
        )
        for index in range(_PLAN_SIZE)
    ]


def test_a_run_records_its_plan_identity_in_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run artifact names its plan: digest, algorithm, version and size."""
    from ard.config import load_config
    from ard.core.sampling import PlanIdentity
    from ard.pipeline import run

    output_dir, _bank, _spy, plan = _resume_rig(tmp_path, monkeypatch, existing=0)

    run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["plan_identity"] == PlanIdentity.of(_plan()).as_dict()


def test_resume_with_a_different_seed_but_the_same_plan_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEGATIVE CONTROL: a seed-comparing guard would refuse this resume.

    The plan double ignores the config, so editing the seed changes no
    coordinate: the recorded identity still matches and the resume must proceed.
    """
    from ard.config import load_config
    from ard.pipeline import run

    output_dir, bank, _spy, plan = _resume_rig(tmp_path, monkeypatch, existing=0)
    run(load_config(tmp_path / "config.toml"), generate_specs=plan)
    before = bank.read_text(encoding="utf-8")

    # The user edits the seed and resumes the same output directory.
    _write_config(tmp_path / "config.toml", output_dir, overwrite=False, seed=9)
    run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    assert bank.read_text(encoding="utf-8") == before
    assert len(_records(bank)) == _TARGET_COUNT


def test_resume_with_a_different_plan_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different plan is a different dataset: refusing beats silently mixing.

    The seed is deliberately left unchanged, so this refusal is only reachable
    through the plan identity — the old seed key cannot see the difference.
    """
    from ard.config import ConfigError, load_config
    from ard.pipeline import run

    output_dir, bank, _spy, plan = _resume_rig(tmp_path, monkeypatch, existing=0)
    run(load_config(tmp_path / "config.toml"), generate_specs=plan)
    before = bank.read_text(encoding="utf-8")
    snapshot = (output_dir / "config.toml").read_text(encoding="utf-8")

    with pytest.raises(ConfigError, match="holds a different plan"):
        run(
            load_config(tmp_path / "config.toml"),
            generate_specs=lambda config: _plan_tagged("other"),
        )

    assert bank.read_text(encoding="utf-8") == before, (
        "a refused resume must not append anything to the bank"
    )
    assert (output_dir / "config.toml").read_text(encoding="utf-8") == snapshot, (
        "a refused resume must not rewrite the previous run's config snapshot"
    )


def test_resume_without_a_recorded_identity_refuses_a_foreign_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-identity bank is verified structurally, never assumed compatible."""
    from ard.config import ConfigError, load_config
    from ard.pipeline import run

    output_dir, bank, _spy, plan = _resume_rig(tmp_path, monkeypatch, existing=0)
    append_anchor(_anchor("foreign-0"), bank)

    with pytest.raises(ConfigError, match="not part of this run's plan"):
        run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    assert len(_records(bank)) == 1


def test_resume_without_a_recorded_identity_allows_a_subset_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A pre-identity bank whose ids are all in the plan is provably one plan."""
    from ard.config import load_config
    from ard.pipeline import run

    _output_dir, bank, _spy, plan = _resume_rig(tmp_path, monkeypatch, existing=2)

    with caplog.at_level("WARNING"):
        run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    assert len(_records(bank)) == _TARGET_COUNT
    assert any("no recorded plan_identity" in record.message for record in caplog.records)


def test_resume_with_the_same_plan_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal is specific to a *changed* plan, not to resuming at all."""
    from ard.config import load_config
    from ard.pipeline import run

    _output_dir, bank, _spy, plan = _resume_rig(tmp_path, monkeypatch, existing=2)

    run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    assert len(_records(bank)) == _TARGET_COUNT
