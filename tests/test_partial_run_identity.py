"""A run that did not finish still leaves a bindable plan identity (WP-S25).

User requirement under test, translated:

    an artifact directory — *including one produced by a run that was
    interrupted, killed or still generating* — must always be bindable to the
    plan that produced it, and the finished artifact must be the authoritative
    declaration, not a restatement of an unfinished one.

The failure this module guards against was observed on the real pipeline: the
1,826-anchor path is routinely interrupted and resumed (about a quarter of the
coordinates are abandoned and asked for again), and a run that stops before its
final ``manifest.json`` used to leave a bank with **no plan name at all**.

What is frozen here:

* ``plan_identity.in_progress.json`` exists from before the first endpoint call,
  carries ``status: "in_progress"`` and the plan's own digest — recomputed
  independently in the test from the construction rule's plan, not read back from
  the file;
* a finished run writes ``manifest.json`` with ``status: "complete"`` and the
  *same* digest, and drops the intermediate record — the two never disagree;
* a resume keeps the recorded identity, and refuses a bank recorded under a
  different plan (the S22 guard, now also covering the interrupted-run case);
* the no-op path (nothing left to generate) does not restate a completed run's
  manifest, and does not leave a stale ``in_progress`` record behind.

Scope note: ``run`` is exercised with the spec plan replaced by a small
deterministic double (the ``generate_specs`` seam of ``pipeline.run``) and the
generator replaced by an in-test writer, so no network call is reachable and the
assertions read the artifact on disk.  The v4 construction rule itself is covered
by ``tests/core/test_sampling.py``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from ard.core.sampling import PlanIdentity
from ard.core.types import AnchorSpec, GeneratedAnchor, TurnSpec
from ard.domain.bank import append_anchor, read_anchor_bank
from ard.domain.text_anchor import AnchorGenerationStats

#: The interrupt record's own file name — asserted literally, because "what a
#: reader finds in an unfinished run directory" is part of the contract.
_PROGRESS_RECORD = "plan_identity.in_progress.json"

_PLAN_SIZE = 5


def _anchor(anchor_id: str, *, answer: str | None = None) -> GeneratedAnchor:
    """A minimal anchor that satisfies the bank's exit gates."""
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
    """The deterministic plan double — the same shape the resume tests use."""
    return [
        AnchorSpec(
            id=f"partial-{index}",
            anchor_meta={"language": "English", "knowledge_domain": "math"},
            turns=[TurnSpec(turn_index=0, role="user", is_final=True)],
        )
        for index in range(_PLAN_SIZE)
    ]


class _InterruptingGenerator:
    """Appends one record per spec and raises after *interrupt_after* of them.

    Raises instead of returning so the interruption has a single, unambiguous
    cause: the run stops *inside* generation, exactly like a killed process —
    every step the pipeline had already taken (config snapshot, progress record,
    the records written so far) stays on disk.
    """

    def __init__(self, *, interrupt_after: int | None = None) -> None:
        self.interrupt_after = interrupt_after
        self.requested: list[int] = []
        self.appended = 0

    def __call__(self, **kwargs: object) -> list[GeneratedAnchor]:
        specs = kwargs["specs"]
        output_path = kwargs["output_path"]
        stats = kwargs["stats"]
        assert isinstance(specs, list)
        assert isinstance(output_path, Path)
        assert isinstance(stats, AnchorGenerationStats)
        self.requested.append(len(specs))
        written: list[GeneratedAnchor] = []
        for index, spec in enumerate(specs):
            if self.interrupt_after is not None and index >= self.interrupt_after:
                raise RuntimeError("simulated interruption inside generation")
            assert isinstance(spec, AnchorSpec)
            anchor = _anchor(spec.id)
            outcome = append_anchor(anchor, output_path)
            assert outcome.value == "appended", f"bank refused a planned anchor: {outcome}"
            written.append(anchor)
            self.appended += 1
        stats.requested = len(specs)
        stats.written = len(written)
        stats.succeeded = len(written)
        return written


def _write_config(path: Path, output_dir: Path) -> None:
    """Minimal valid config; the endpoints are dummies that are never reached."""
    path.write_text(
        "\n".join(
            [
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
                "seed = 7",
                "concurrency = 1",
                "",
                "[ontology]",
                'path = "unused.json"',
                "",
                "[output]",
                f'directory = "{output_dir}"',
                "overwrite = false",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _rig(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    existing: int = 0,
    interrupt_after: int | None = None,
) -> tuple[Path, Path, _InterruptingGenerator, Callable[[object], list[AnchorSpec]]]:
    """Prepare an output directory, the config and the generator double.

    *existing* anchors of the plan are put in the bank first (the state a resume
    finds); a directory that already exists is reused, so the same rig serves the
    "interrupt, then prepare the resume" sequence across two ``run`` calls.
    """
    from ard import pipeline

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    bank = output_dir / "anchor_bank.jsonl"
    for spec in _plan()[:existing]:
        append_anchor(_anchor(spec.id), bank)

    config_path = tmp_path / "config.toml"
    _write_config(config_path, output_dir)

    generator = _InterruptingGenerator(interrupt_after=interrupt_after)
    # The ontology is never read: the plan double replaces the v4 sampler.
    monkeypatch.setattr(
        pipeline, "sample_specs", lambda config: pytest.fail("the plan double must be used")
    )
    monkeypatch.setattr(pipeline, "generate_text_anchors", generator)
    return output_dir, bank, generator, (lambda config: _plan())


def _progress_record(output_dir: Path) -> dict:
    """Read the intermediate record — failing loudly when it is not there."""
    path = output_dir / _PROGRESS_RECORD
    assert path.is_file(), (
        f"an interrupted run left no {_PROGRESS_RECORD}: the directory cannot be "
        f"bound to a plan (contents: {sorted(p.name for p in output_dir.iterdir())})"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _identity_from_bank(bank: Path) -> dict:
    """Recompute the plan identity the recorded coordinates *mean*.

    The plan double is read back through its ids (the bank stores each coordinate's
    stable id), then the identity is recomputed by ``PlanIdentity.of`` — the same
    recipe the pipeline uses, exercised here on the plan the artifact actually
    holds rather than on the value the file reports.
    """
    by_id = {spec.id: spec for spec in _plan()}
    return PlanIdentity.of([by_id[record["id"]] for record in read_anchor_bank(bank)]).as_dict()


def _manifest(output_dir: Path) -> dict:
    return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))


# ── (a) interrupted run → the directory is still bound to its plan ──────────


def test_interrupted_run_leaves_a_progress_record_with_the_plan_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interruption is exactly the case that used to leave no plan name."""
    from ard.config import load_config
    from ard.pipeline import run

    output_dir, bank, _, plan = _rig(tmp_path, monkeypatch, interrupt_after=1)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    record = _progress_record(output_dir)
    expected = PlanIdentity.of(_plan()).as_dict()

    assert record["ard_progress_record"] == "plan_identity/v1"
    assert record["status"] == "in_progress", "the file must say it is not the manifest"
    assert "NOT the final declaration" in record["note"]
    assert record["plan_identity"] == expected, (
        "the recorded digest must be the plan's own digest, recomputed independently"
    )
    assert record["plan_identity"]["plan_size"] == _PLAN_SIZE
    assert record["counters"] == {"existing": 0, "new": _PLAN_SIZE, "written": _PLAN_SIZE}
    assert record["output_dir"] == str(output_dir)
    # The record is self-describing: its status is readable, not inferred.
    time.strptime(record["started_at"], "%Y-%m-%d %H:%M:%S")

    # It must not pretend to be the finished artifact.
    assert not (output_dir / "manifest.json").exists()
    assert "total_anchors" not in record

    # The bank itself is bound by the same recipe: the partial bank recomputes to
    # the partial plan's digest (which is *not* the full plan's — that is why the
    # counters and the status, not the bank, are what the record adds).
    subset = _identity_from_bank(bank)
    assert subset["plan_size"] == 1, "one record survived the interruption"
    assert subset == PlanIdentity.of(_plan()[:1]).as_dict()
    assert subset["digest"] != expected["digest"]


# ── (b) finished run → one authoritative declaration, same digest ───────────


def test_finished_run_manifest_and_record_agree_on_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``manifest.json`` is complete and authoritative; no stale record survives."""
    from ard.config import load_config
    from ard.pipeline import run

    output_dir, bank, generator, plan = _rig(tmp_path, monkeypatch)
    run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    manifest = _manifest(output_dir)
    expected = PlanIdentity.of(_plan()).as_dict()
    assert manifest["status"] == "complete"
    assert manifest["plan_identity"] == expected
    assert manifest["total_anchors"] == _PLAN_SIZE
    assert manifest["generation"]["counters"]["written"] == _PLAN_SIZE
    assert generator.requested == [_PLAN_SIZE]
    assert not (output_dir / _PROGRESS_RECORD).exists(), (
        "a finished run carries exactly one declaration"
    )
    assert _identity_from_bank(bank) == expected


# ── (c) resume → the identity does not move, and different plans are refused ─


def test_resume_after_interruption_keeps_the_recorded_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted directory resumes as one plan and ends with a manifest."""
    from ard.config import load_config
    from ard.pipeline import run

    output_dir, bank, _, plan = _rig(tmp_path, monkeypatch, interrupt_after=1)
    config = load_config(tmp_path / "config.toml")
    with pytest.raises(RuntimeError):
        run(config, generate_specs=plan)
    recorded = _progress_record(output_dir)["plan_identity"]

    _, _, resume_generator, plan_again = _rig(tmp_path, monkeypatch, existing=1)
    run(load_config(tmp_path / "config.toml"), generate_specs=plan_again)

    assert resume_generator.requested == [_PLAN_SIZE - 1], "only the missing part is asked for"
    manifest = _manifest(output_dir)
    assert manifest["plan_identity"] == recorded, "the plan identity must not move on resume"
    assert manifest["plan_identity"] == PlanIdentity.of(_plan()).as_dict()
    assert manifest["status"] == "complete"
    assert not (output_dir / _PROGRESS_RECORD).exists()
    assert _identity_from_bank(bank) == manifest["plan_identity"]


def test_resume_refuses_a_directory_recorded_under_another_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interrupted-record guard is the S22 guard, not a weaker one."""
    from ard.config import ConfigError, load_config
    from ard.pipeline import run

    output_dir, bank, _, plan = _rig(tmp_path, monkeypatch, existing=1)
    before = bank.read_text(encoding="utf-8")
    foreign = PlanIdentity.of(_plan()[:-1]).as_dict()  # a genuinely different plan
    assert foreign["digest"] != PlanIdentity.of(_plan()).digest
    (output_dir / _PROGRESS_RECORD).write_text(
        json.dumps({"status": "in_progress", "plan_identity": foreign}), encoding="utf-8"
    )

    with pytest.raises(ConfigError, match="different plan"):
        run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    assert bank.read_text(encoding="utf-8") == before
    assert not (output_dir / "manifest.json").exists()
    assert not (output_dir / "config.toml").exists(), "the refusal is before any side effect"


# ── (e) a no-op invocation never restates a finished run ────────────────────


def test_noop_rerun_leaves_a_completed_manifest_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to generate → the finished run's health and status stay put."""
    from ard.config import load_config
    from ard.pipeline import run

    output_dir, bank, generator, plan = _rig(tmp_path, monkeypatch)
    run(load_config(tmp_path / "config.toml"), generate_specs=plan)
    manifest_path = output_dir / "manifest.json"
    before = manifest_path.read_text(encoding="utf-8")
    mtime_before = manifest_path.stat().st_mtime_ns
    assert json.loads(before)["status"] == "complete"
    bank_before = bank.read_text(encoding="utf-8")

    _, _, noop_generator, plan_again = _rig(tmp_path, monkeypatch, existing=_PLAN_SIZE)
    run(load_config(tmp_path / "config.toml"), generate_specs=plan_again)

    assert noop_generator.requested == [], "a satisfied bank must generate nothing"
    assert manifest_path.read_text(encoding="utf-8") == before, (
        "the no-op path must not restate a completed run (status, counters and all)"
    )
    assert manifest_path.stat().st_mtime_ns == mtime_before, "the file was not rewritten"
    assert not (output_dir / _PROGRESS_RECORD).exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "complete"
    assert bank.read_text(encoding="utf-8") == bank_before, "the bank is untouched too"
    assert generator.appended == _PLAN_SIZE


def test_noop_after_interrupted_run_still_writes_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The preserve rule is about *completed* manifests only.

    An interrupted directory has no manifest; a no-op invocation there must write
    one (``status: "complete"``, same identity) rather than preserve a progress
    record as if it were authoritative.
    """
    from ard.config import load_config
    from ard.pipeline import run

    output_dir, _, _, plan = _rig(tmp_path, monkeypatch, interrupt_after=1)
    with pytest.raises(RuntimeError):
        run(load_config(tmp_path / "config.toml"), generate_specs=plan)
    assert not (output_dir / "manifest.json").exists()

    _, _, noop_generator, plan_again = _rig(tmp_path, monkeypatch)
    run(load_config(tmp_path / "config.toml"), generate_specs=plan_again)

    assert noop_generator.requested == [_PLAN_SIZE - 1]
    manifest = _manifest(output_dir)
    assert manifest["status"] == "complete"
    assert manifest["plan_identity"] == PlanIdentity.of(_plan()).as_dict()
    assert not (output_dir / _PROGRESS_RECORD).exists()
