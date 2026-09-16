"""Checkpoint/resume contract of ``pipeline.run`` and the bank it resumes from (R5).

User requirement under test, verbatim:

    "不需要跨 run 覆盖，每个 run 独立，不过之前支持断点继续，如果 output
     目录下面是未完成的应该支持继续。"

Translated into the three behaviours frozen here:

1. an output directory holding fewer records than ``generation.target_count``
   resumes: only the missing ``target_count - existing`` anchors are asked for,
   they are **appended** to the bank, and the records already on disk survive
   byte-for-byte;
2. an output directory that already satisfies ``target_count`` generates
   nothing at all and returns without touching the bank (idempotent re-run);
3. ``output.overwrite`` is the explicit opt-in that clears the bank (§3.3
   预授权退路) — without it, a second run never destroys the first one's records.

Plus the failure mode that made (1) untrustworthy: an append interrupted mid-line
left a fragment without its trailing newline, after which the record *count* and
the record *reader* disagreed and every following record was concatenated onto the
fragment.  Sections 5–7 pin that down at the bank level.

Scope note: the pipeline half is deliberately *not* a second sampler test.  ``run``
is exercised with the ontology loader and the spec sampler replaced by doubles
that return ids taken straight from the requested ``target_count``, so what the
assertions read is the resume arithmetic and the bank on disk — not the FPS
sampler (covered by ``tests/test_fps_sampler.py``) and not the turn generator
(covered by ``tests/domain/test_text_anchor_backpressure.py``).  No network call
is reachable: the generator double never touches the API clients the pipeline
constructs, and ``api_base`` points at a closed local port.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ard.core.types import AnchorSpec, GeneratedAnchor, TurnSpec
from ard.domain.bank import (
    append_anchor,
    count_existing_anchors,
    count_unique_anchor_ids,
    read_anchor_bank,
)

#: Target used by the "incomplete bank" cases: small enough to read at a glance,
#: large enough that "2 existing + 3 missing" is not a coincidence of the bank.
_TARGET_COUNT = 5


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


def _specs_for(target_count: int, tag: str) -> list[AnchorSpec]:
    """``target_count`` distinct specs — the pipeline will hand them to the generator."""
    assert target_count > 0, "the pipeline must not ask for a non-positive batch"
    return [
        AnchorSpec(
            id=f"{tag}-{index}",
            anchor_meta={"language": "English", "knowledge_domain": "math"},
            turns=[TurnSpec(turn_index=0, role="user", is_final=True)],
        )
        for index in range(target_count)
    ]


def _write_config(path: Path, output_dir: Path, target_count: int, *, overwrite: bool) -> None:
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
        f"target_count = {target_count}",
        "seed = 7",
        "concurrency = 1",
        "max_turns = 1",
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

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.requested: list[int] = []

    def __call__(self, **kwargs: object) -> list[GeneratedAnchor]:
        specs = kwargs["specs"]
        output_path = kwargs["output_path"]
        stats = kwargs["stats"]
        assert isinstance(specs, list)
        assert isinstance(output_path, Path)
        self.requested.append(len(specs))
        written: list[GeneratedAnchor] = []
        for spec in specs:
            assert isinstance(spec, AnchorSpec)
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
    target_count: int = _TARGET_COUNT,
    overwrite: bool = False,
    tag: str = "resumed",
) -> tuple[Path, Path, _RunSpy]:
    """Prepare an output dir with *existing* records and return the run rig."""
    from ard import pipeline

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)
    bank = output_dir / "anchor_bank.jsonl"
    for index in range(existing):
        append_anchor(_anchor(f"existing-{index}"), bank)

    config_path = tmp_path / "config.toml"
    _write_config(config_path, output_dir, target_count, overwrite=overwrite)

    spy = _RunSpy(tag)
    monkeypatch.setattr(pipeline, "load_ontology", lambda path: {"stub": True})
    monkeypatch.setattr(
        pipeline,
        "sample_anchors",
        lambda ontology, config, rng: _specs_for(config.target_count, tag),
    )
    monkeypatch.setattr(pipeline, "generate_text_anchors", spy)
    return output_dir, bank, spy


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
    """N records on disk + ``target_count=M`` → exactly M-N new records, appended."""
    from ard.config import load_config
    from ard.pipeline import run

    existing = 2
    output_dir, bank, spy = _resume_rig(tmp_path, monkeypatch, existing=existing)
    before = bank.read_text(encoding="utf-8")

    run(load_config(tmp_path / "config.toml"))

    assert spy.requested == [_TARGET_COUNT - existing], (
        "the run must ask for the shortfall, not for the full target again"
    )
    records = _records(bank)
    assert len(records) == _TARGET_COUNT
    assert [r["id"] for r in records[:existing]] == [
        f"existing-{index}" for index in range(existing)
    ], "the pre-existing records must not be reordered or rewritten"
    assert [r["id"] for r in records[existing:]] == [
        f"resumed-{index}" for index in range(_TARGET_COUNT - existing)
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

    _, bank, spy = _resume_rig(tmp_path, monkeypatch, existing=3)

    def _forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("the resume path must not reach the API clients")

    monkeypatch.setattr("ard.backends.api_client.ChatAPIClient.chat", _forbidden)
    run(load_config(tmp_path / "config.toml"))
    assert spy.requested == [_TARGET_COUNT - 3]
    assert count_existing_anchors(bank) == _TARGET_COUNT


# ── 2. finished run → nothing to do ─────────────────────────────────────────


def test_complete_bank_is_left_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N ≥ M → no generation, no rewrite, same bytes (a re-run is idempotent)."""
    from ard.config import load_config
    from ard.pipeline import run

    output_dir, bank, spy = _resume_rig(tmp_path, monkeypatch, existing=_TARGET_COUNT)
    before = bank.read_text(encoding="utf-8")

    result_dir = run(load_config(tmp_path / "config.toml"))

    assert spy.requested == [], "a satisfied bank must not trigger generation"
    assert bank.read_text(encoding="utf-8") == before
    assert count_existing_anchors(bank) == _TARGET_COUNT
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_anchors"] == _TARGET_COUNT


def test_overfilled_bank_is_left_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """More records than requested is still "nothing to do", not an error."""
    from ard.config import load_config
    from ard.pipeline import run

    _, bank, spy = _resume_rig(tmp_path, monkeypatch, existing=_TARGET_COUNT + 2)
    before = bank.read_text(encoding="utf-8")

    run(load_config(tmp_path / "config.toml"))

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
    _, bank, spy = _resume_rig(
        tmp_path, monkeypatch, existing=existing, overwrite=True, tag="fresh"
    )

    run(load_config(tmp_path / "config.toml"))

    assert spy.requested == [_TARGET_COUNT], (
        "overwrite starts from an empty bank, so the full target is requested"
    )
    records = _records(bank)
    assert len(records) == _TARGET_COUNT
    assert [r["id"] for r in records] == [f"fresh-{index}" for index in range(_TARGET_COUNT)]


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
    _, bank, spy = _resume_rig(tmp_path, monkeypatch, existing=existing)
    with bank.open("a", encoding="utf-8") as f:
        f.write('{"id": "interrupted", "messages": [{"role": "user"')  # no newline

    with caplog.at_level("WARNING"):
        run(load_config(tmp_path / "config.toml"))

    assert any(
        "unreadable line" in record.getMessage() for record in caplog.records
    ), "the discarded fragment must be announced, not silently swallowed"
    assert spy.requested == [_TARGET_COUNT - existing], (
        "the fragment is not a record, so it must not shrink the missing batch"
    )
    records = _records(bank)
    assert [r["id"] for r in records[:existing]] == [
        f"existing-{index}" for index in range(existing)
    ]
    assert [r["id"] for r in records[existing:]] == [
        f"resumed-{index}" for index in range(_TARGET_COUNT - existing)
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
    output_dir, bank, spy = _resume_rig(tmp_path, monkeypatch, existing=existing)
    before = bank.read_text(encoding="utf-8")

    # A generator that delivers one anchor no matter how many were requested.
    def _half_delivery(**kwargs: object) -> list[GeneratedAnchor]:
        specs = kwargs["specs"]
        stats = kwargs["stats"]
        assert isinstance(specs, list)
        spy.requested.append(len(specs))
        anchor = _anchor("only-one")
        append_anchor(anchor, kwargs["output_path"])
        stats.requested = len(specs)
        stats.written = 1
        stats.succeeded = 1
        return [anchor]

    monkeypatch.setattr(pipeline, "generate_text_anchors", _half_delivery)
    pipeline.run(load_config(tmp_path / "config.toml"))

    assert spy.requested == [_TARGET_COUNT - existing]
    assert bank.read_text(encoding="utf-8").startswith(before)
    assert len(_records(bank)) == existing + 1
