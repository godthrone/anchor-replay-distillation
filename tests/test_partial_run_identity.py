"""Run identity, prefix-compatible resume, and the no-coordinate-dedup rule (WP-4).

User requirement under test, translated:

    a run directory — *including one produced by an interrupted run* — is always
    bindable to the plan that produced it; raising ``[generation] count`` and
    re-running the same directory appends; a bank that does not line up with the
    new plan is refused; and two samples that happen to share a coordinate are
    both kept, because a coordinate is content, not identity.

What is frozen here (v5):

* an anchor id is the plan *position* (``run_key + cycle + position``) and does
  not depend on N, so the first N ids of a larger plan are exactly the smaller
  plan's — the property that makes "raise N, re-run the same directory" append
  instead of rewrite;
* an id is therefore **not** a content fingerprint: the resume guard compares
  the recorded coordinate *at each id* against the plan's coordinate there, and
  refuses a foreign id or a coordinate mismatch.  The pre-registered negative
  control is the smoke plan, whose ids are a subset of the full plan's while its
  coordinates are not;
* a repeated coordinate is never a duplicate.  Two records may carry the same
  ``anchor_meta`` under different ids, and both survive; the coverage readout
  counts distinct coordinates but never gates records on them;
* an interrupted run leaves ``plan_identity.in_progress.json`` from before the
  first endpoint call, a finished run replaces it with ``manifest.json``
  (``status: complete``, same identity), and a no-op re-run does not restate a
  completed run.

Scope note: the pipeline half uses ``run``'s ``generate_specs`` seam for the
deterministic cases (no ontology load, no network) and the real ontology for the
cases that need real prefix-stable ids and the smoke/full shape difference.  The
acceptance phase is disabled in these configs: it is a separate concern with its
own module (``tests/test_acceptance_wiring.py``), so the resume tests do not
depend on it.  The generator is replaced by an in-test writer, so no endpoint is
reachable.  The construction rule itself is covered by
``tests/core/test_sampling.py``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from ard.core.sampling import PlanIdentity
from ard.core.types import AnchorSpec, GeneratedAnchor, TurnSpec
from ard.domain.bank import anchor_to_dict, append_anchor, read_anchor_bank
from ard.domain.text_anchor import AnchorGenerationStats

#: The interrupt record's own file name — asserted literally, because "what a
#: reader finds in an unfinished run directory" is part of the contract.
_PROGRESS_RECORD = "plan_identity.in_progress.json"

#: The injected plan's length.  Small enough to read, large enough that
#: "2 existing + 3 missing" is not a coincidence of the bank.
_PLAN_SIZE = 5

#: The seed the test configs pin; it enters the run key of a real-ontology id.
_SEED = 7

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ONTOLOGY = str(_REPO_ROOT / "ontology" / "anchor_ontology.v4.json")


def _spec(anchor_id: str, meta: dict[str, str]) -> AnchorSpec:
    """One minimally valid spec at *anchor_id* carrying *meta*."""
    return AnchorSpec(
        id=anchor_id,
        anchor_meta=dict(meta),
        turns=[TurnSpec(turn_index=0, role="user", is_final=True)],
    )


def _plan() -> list[AnchorSpec]:
    """The deterministic plan double — the same shape the resume tests use."""
    meta = {"language": "English", "knowledge_domain": "math"}
    return [_spec(f"partial-{index}", meta) for index in range(_PLAN_SIZE)]


def _prefix_plan(length: int) -> list[AnchorSpec]:
    """A plan double whose first *k* entries are stable as *length* grows.

    Every position carries the same coordinate, so ``_prefix_plan(5)`` and
    ``_prefix_plan(8)`` agree on their shared ids **and** coordinates — the
    "user raised N and re-ran the directory" case, in its purest form.
    """
    meta = {"language": "English", "knowledge_domain": "math"}
    return [_spec(f"prefix-{index}", meta) for index in range(length)]


def _recurring_coordinate_plan() -> list[AnchorSpec]:
    """Two plan positions carrying an *identical* coordinate under distinct ids.

    This is the cross-cycle case in miniature: cycle 1 can re-ask a coordinate
    cycle 0 already asked.  Both are valid samples (the generator is
    stochastic), so the pipeline must keep both.
    """
    meta = {"language": "English", "knowledge_domain": "math"}
    return [_spec("recurring-c00000p00000", meta), _spec("recurring-c00001p00000", meta)]


def _identity(specs: list[AnchorSpec], *, plan_size: int = _PLAN_SIZE) -> dict:
    """The v2 identity of *specs* as the injected-plan pipeline computes it.

    An injected plan has no ontology, so the fingerprint is the explicit unknown
    ``""`` and the plan is described as one cycle of ``plan_size`` units — the
    same convention ``pipeline._injected_plan_context`` uses.
    """
    return PlanIdentity.of(
        list(specs),
        ontology_sha256="",
        seed=_SEED,
        count=plan_size,
        unit_total=plan_size,
    ).as_dict()


def _anchor_for(spec: AnchorSpec) -> GeneratedAnchor:
    """A bank-valid anchor whose ``anchor_meta`` is exactly the spec's coordinate."""
    return GeneratedAnchor(
        id=spec.id,
        messages=[{"role": "user", "content": f"question {spec.id}"}],
        target_answer=f"answer {spec.id}",
        target_model="target-model",
        input_generator_model="input-model",
        anchor_meta=dict(spec.anchor_meta),
        reasoning=None,
    )


class _WritingGenerator:
    """Appends one record per spec; can abandon ids or raise after *interrupt_after*.

    Raising instead of returning makes the interruption have a single,
    unambiguous cause: the run stops *inside* generation, exactly like a killed
    process — every step the pipeline had already taken (config snapshot,
    progress record, the records written so far) stays on disk.
    """

    def __init__(
        self,
        *,
        interrupt_after: int | None = None,
        abandoned: set[str] | None = None,
        bulk: bool = False,
    ) -> None:
        self.interrupt_after = interrupt_after
        self.abandoned = abandoned or set()
        #: Write the whole bank in one pass instead of one ``append_anchor`` per
        #: record.  ``append_anchor``'s id cache is invalidated by every append,
        #: so N appends re-read the file N times (O(N²)); at N = 2U that costs
        #: ~30 s in bookkeeping the bank's own tests cover.  The large-count test
        #: only needs the pipeline to *plan and request* every position and the
        #: bank to end up with all of them, so it writes in bulk and says so.
        self.bulk = bulk
        self.requested: list[int] = []
        self.requested_ids: list[list[str]] = []
        self.appended = 0

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
        for index, spec in enumerate(specs):
            if self.interrupt_after is not None and index >= self.interrupt_after:
                raise RuntimeError("simulated interruption inside generation")
            if self.bulk:
                continue  # handled in one pass below
            assert isinstance(spec, AnchorSpec)
            if spec.id in self.abandoned:
                continue  # the exhausted-retries failure this test reproduces
            anchor = _anchor_for(spec)
            outcome = append_anchor(anchor, output_path)
            assert outcome.value == "appended", f"bank refused a planned anchor: {outcome}"
            written.append(anchor)
            self.appended += 1
        if self.bulk:
            assert self.interrupt_after is None and not self.abandoned
            written = [_anchor_for(spec) for spec in specs if isinstance(spec, AnchorSpec)]
            output_path.write_text(
                "".join(
                    json.dumps(anchor_to_dict(anchor), ensure_ascii=False) + "\n"
                    for anchor in written
                ),
                encoding="utf-8",
            )
            self.appended = len(written)
        stats.requested = len(specs)
        stats.written = len(written)
        stats.succeeded = len(written)
        return written


def _write_config(
    path: Path,
    output_dir: Path,
    *,
    seed: int = _SEED,
    count: int | None = None,
    ontology: str = "unused.json",
) -> None:
    """Minimal valid config; the endpoints are dummies that are never called.

    ``[coverage] enabled = false`` keeps the resume tests independent of the
    acceptance phase (it has its own module and its own tests); everything the
    assertions read is written by the pipeline itself.
    """
    generation = ["[generation]", f"seed = {seed}"]
    if count is not None:
        generation.append(f"count = {count}")
    generation.append("concurrency = 1")
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
        *generation,
        "",
        "[ontology]",
        f'path = "{ontology}"',
        "",
        "[output]",
        f'directory = "{output_dir}"',
        "overwrite = false",
        "",
        "[coverage]",
        "enabled = false",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _rig(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    existing: int = 0,
    interrupt_after: int | None = None,
    plan: Callable[[], list[AnchorSpec]] = _plan,
) -> tuple[Path, Path, _WritingGenerator, Callable[[object], list[AnchorSpec]]]:
    """Prepare an output directory, the config and the generator double.

    *existing* anchors of the plan are put in the bank first (the state a resume
    finds); a directory that already exists is reused, so the same rig serves the
    "interrupt, then prepare the resume" sequence across two ``run`` calls.
    """
    from ard import pipeline

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    bank = output_dir / "anchor_bank.jsonl"
    for spec in plan()[:existing]:
        append_anchor(_anchor_for(spec), bank)

    config_path = tmp_path / "config.toml"
    _write_config(config_path, output_dir)

    generator = _WritingGenerator(interrupt_after=interrupt_after)
    # The ontology is never read: the plan double replaces the v4 sampler.
    monkeypatch.setattr(
        pipeline, "sample_specs", lambda config: pytest.fail("the plan double must be used")
    )
    monkeypatch.setattr(pipeline, "generate_text_anchors", generator)
    return output_dir, bank, generator, (lambda config: plan())


def _progress_record(output_dir: Path) -> dict:
    """Read the intermediate record — failing loudly when it is not there."""
    path = output_dir / _PROGRESS_RECORD
    assert path.is_file(), (
        f"an interrupted run left no {_PROGRESS_RECORD}: the directory cannot be "
        f"bound to a plan (contents: {sorted(p.name for p in output_dir.iterdir())})"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(output_dir: Path) -> dict:
    return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))


def _records(bank: Path) -> list[dict]:
    return read_anchor_bank(bank)


def _ids(bank: Path) -> list[str]:
    return [record["id"] for record in _records(bank)]


def _identity_from_bank(bank: Path) -> dict:
    """Recompute the identity the recorded coordinates *mean*.

    The plan double is read back through its ids (the bank stores each record's
    id), then the identity is recomputed by ``PlanIdentity.of`` — the same recipe
    the pipeline uses, exercised here on the plan the artifact actually holds
    rather than on the value the file reports.
    """
    by_id = {spec.id: spec for spec in _plan()}
    return _identity([by_id[record["id"]] for record in _records(bank)])


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
    expected = _identity(_plan())

    assert record["ard_progress_record"] == "plan_identity/v1"
    assert record["status"] == "in_progress", "the file must say it is not the manifest"
    assert "NOT the final declaration" in record["note"]
    assert record["plan_identity"] == expected, (
        "the recorded digest must be the plan's own digest, recomputed independently"
    )
    assert record["plan_identity"]["version"] == 2
    assert record["plan_identity"]["plan_size"] == _PLAN_SIZE
    assert record["counters"] == {"existing": 0, "new": _PLAN_SIZE, "written": _PLAN_SIZE}
    assert record["output_dir"] == str(output_dir)
    # The record is self-describing: its status is readable, not inferred.
    time.strptime(record["started_at"], "%Y-%m-%d %H:%M:%S")

    # It must not pretend to be the finished artifact.
    assert not (output_dir / "manifest.json").exists()
    assert "total_anchors" not in record

    # The bank itself is bound by the same recipe: the partial bank recomputes to
    # a different digest than the full plan (fewer rows), which is why the record
    # exists at all.
    subset = _identity_from_bank(bank)
    assert subset != expected
    assert len(_records(bank)) == 1, "one record survived the interruption"


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
    expected = _identity(_plan())
    assert manifest["status"] == "complete"
    assert manifest["plan_identity"] == expected
    assert manifest["total_anchors"] == _PLAN_SIZE
    assert manifest["generation"]["counters"]["written"] == _PLAN_SIZE
    assert generator.requested == [_PLAN_SIZE]
    assert not (output_dir / _PROGRESS_RECORD).exists(), (
        "a finished run carries exactly one declaration"
    )
    assert _identity_from_bank(bank) == expected


# ── (c) resume → the identity does not move ─────────────────────────────────


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
    assert manifest["plan_identity"] == _identity(_plan())
    assert manifest["status"] == "complete"
    assert not (output_dir / _PROGRESS_RECORD).exists()
    assert _identity_from_bank(bank) == manifest["plan_identity"]


# ── (d) the guard refuses a bank that is not at its own plan positions ──────


def test_resume_refuses_a_foreign_anchor_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An id outside the plan is a foreign record, whatever its coordinate."""
    from ard.config import ConfigError, load_config
    from ard.pipeline import run

    output_dir, bank, _, plan = _rig(tmp_path, monkeypatch, existing=1)
    append_anchor(_anchor_for(_spec("foreign-9", {"language": "English"})), bank)
    before = bank.read_text(encoding="utf-8")

    with pytest.raises(ConfigError, match="not part of this run's plan"):
        run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    assert bank.read_text(encoding="utf-8") == before, "a refused resume appends nothing"
    assert not (output_dir / "config.toml").exists(), "the refusal is before any side effect"


def test_resume_refuses_a_coordinate_that_differs_at_the_same_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The id is a position serial number, so a shared id is not the same sample.

    This is the smoke-vs-full shape difference in the injected-plan seam: an id
    that exists in both plans can still name a different coordinate.
    """
    from ard.config import ConfigError, load_config
    from ard.pipeline import run

    output_dir, bank, _, plan = _rig(tmp_path, monkeypatch)
    append_anchor(
        _anchor_for(_spec("partial-0", {"language": "English", "knowledge_domain": "physics"})),
        bank,
    )
    before = bank.read_text(encoding="utf-8")

    with pytest.raises(ConfigError, match="coordinate differs"):
        run(load_config(tmp_path / "config.toml"), generate_specs=plan)

    assert bank.read_text(encoding="utf-8") == before
    assert not (output_dir / "config.toml").exists(), "the refusal is before any side effect"


def test_a_stale_recorded_identity_alone_does_not_block_a_matching_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recorded identity is an audit trail; the coordinate comparison decides.

    A recorded identity that no longer matches this run's (a hand-edited record,
    or a smaller plan that the bank's ids and coordinates still fit) must not, by
    itself, refuse a resume whose records all sit at their own plan positions —
    that is the property that lets a raised N append.
    """
    from ard.config import load_config
    from ard.pipeline import run

    output_dir, bank, _, plan = _rig(tmp_path, monkeypatch, existing=1)
    (output_dir / _PROGRESS_RECORD).write_text(
        json.dumps({"status": "in_progress", "plan_identity": _identity(_plan()[:-1])}),
        encoding="utf-8",
    )

    _, _, generator, plan_again = _rig(tmp_path, monkeypatch, existing=1)
    run(load_config(tmp_path / "config.toml"), generate_specs=plan_again)

    assert generator.requested == [_PLAN_SIZE - 1]
    assert _manifest(output_dir)["plan_identity"] == _identity(_plan())


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
    assert manifest["plan_identity"] == _identity(_plan())
    assert not (output_dir / _PROGRESS_RECORD).exists()


# ── (f) ★ prefix-compatible resume: raising N appends, never rewrites ───────


def test_raising_the_plan_length_appends_and_leaves_existing_records_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The load-bearing property: plan(5) is a prefix of plan(8) in ids *and* coords."""
    from ard.config import load_config
    from ard.pipeline import run

    output_dir, bank, _, small = _rig(tmp_path, monkeypatch, plan=lambda: _prefix_plan(5))
    run(load_config(tmp_path / "config.toml"), generate_specs=small)
    before = bank.read_text(encoding="utf-8")
    assert len(_records(bank)) == 5

    _, _, bigger, large = _rig(tmp_path, monkeypatch, existing=5, plan=lambda: _prefix_plan(8))
    run(load_config(tmp_path / "config.toml"), generate_specs=large)

    after = bank.read_text(encoding="utf-8")
    assert after.startswith(before), "the existing records must be byte-identical, not rewritten"
    assert bigger.requested == [3], "only the three new positions are generated"
    assert len(_records(bank)) == 8
    assert len(set(_ids(bank))) == 8
    assert _manifest(output_dir)["status"] == "complete"


# ── (g) ★ no coordinate de-duplication: a repeated coordinate is kept ───────


def test_a_coordinate_repeated_in_a_later_cycle_is_kept_as_a_second_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POSITIVE assertion: same coordinate, two ids → two records survive."""
    from ard.config import load_config
    from ard.pipeline import run

    plan = _recurring_coordinate_plan()
    output_dir, bank, generator, sampler = _rig(tmp_path, monkeypatch, plan=lambda: plan)
    run(load_config(tmp_path / "config.toml"), generate_specs=sampler)

    records = _records(bank)
    assert generator.requested == [2], "both positions are asked for"
    assert sorted(record["id"] for record in records) == sorted(spec.id for spec in plan)
    assert len(records) == 2, "a repeated coordinate must not be dropped as a duplicate"
    coordinates = [record["anchor_meta"] for record in records]
    assert coordinates[0] == coordinates[1], "the fixture really does repeat the coordinate"

    readout = _manifest(output_dir)["plan"]
    assert readout["planned_anchors"] == 2
    assert readout["written_anchors"] == 2
    assert readout["distinct_coordinates"] == 1, (
        "the coverage readout counts the coordinate once, but the records are not gated on it"
    )


# ── (h) real-ontology resume: raise [generation] count on the same directory ─


def _real_ontology_rig(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_dir: Path,
    *,
    seed: int = _SEED,
    count: int | None = None,
    bulk: bool = False,
) -> tuple[Path, _WritingGenerator]:
    """Config + generator double for a run that builds its plan from the ontology."""
    from ard import pipeline

    config_path = tmp_path / "config.toml"
    _write_config(config_path, output_dir, seed=seed, count=count, ontology=_ONTOLOGY)
    generator = _WritingGenerator(bulk=bulk)
    monkeypatch.setattr(pipeline, "generate_text_anchors", generator)
    return config_path, generator


def test_raising_config_count_on_the_same_directory_appends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`count = 6` then `count = 12` on one directory: append within one plan."""
    from ard.config import load_config
    from ard.pipeline import run

    output_dir = tmp_path / "out"
    config_path, first = _real_ontology_rig(tmp_path, monkeypatch, output_dir, count=6)
    run(load_config(config_path))
    before = (output_dir / "anchor_bank.jsonl").read_text(encoding="utf-8")
    assert len(_records(output_dir / "anchor_bank.jsonl")) == 6
    assert first.requested == [6]

    config_path, second = _real_ontology_rig(tmp_path, monkeypatch, output_dir, count=12)
    run(load_config(config_path))

    after = (output_dir / "anchor_bank.jsonl").read_text(encoding="utf-8")
    assert after.startswith(before), "raising N must not rewrite the first six records"
    assert second.requested == [6], "only the six new positions are generated"
    ids = _ids(output_dir / "anchor_bank.jsonl")
    assert len(ids) == 12
    assert len(set(ids)) == 12, "ids are pairwise distinct"
    manifest = _manifest(output_dir)
    assert manifest["plan"]["count"] == 12
    assert manifest["plan"]["planned_anchors"] == 12
    assert manifest["plan"]["written_anchors"] == 12
    assert manifest["plan"]["full_cycles"] == 0
    assert manifest["plan"]["last_cycle_size"] == 12
    assert manifest["plan"]["ontology_sha256"], "a real run records its ontology fingerprint"


def test_a_changed_seed_refuses_the_resume_with_an_actionable_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different seed mints different run keys, so every old id is foreign."""
    from ard.config import ConfigError, load_config
    from ard.pipeline import run

    output_dir = tmp_path / "out"
    config_path, _ = _real_ontology_rig(tmp_path, monkeypatch, output_dir, count=6)
    run(load_config(config_path))
    bank = output_dir / "anchor_bank.jsonl"
    before = bank.read_text(encoding="utf-8")
    snapshot = (output_dir / "config.toml").read_text(encoding="utf-8")

    config_path, _ = _real_ontology_rig(tmp_path, monkeypatch, output_dir, seed=99, count=6)
    with pytest.raises(ConfigError) as refused:
        run(load_config(config_path))

    message = str(refused.value)
    assert "refusing to resume" in message
    assert "own output.directory" in message, (
        "the refusal must tell the user what to do: give this batch a new directory"
    )
    assert bank.read_text(encoding="utf-8") == before
    assert (output_dir / "config.toml").read_text(encoding="utf-8") == snapshot, (
        "a refused resume must not rewrite the previous run's config snapshot"
    )


def _full_plan_ids(*, count: int) -> set[str]:
    """The id set a real full plan of *count* positions mints for this ontology."""
    from ard.backends.ontology_loader import load_ontology_v4
    from ard.core.sampling import build_specs, sample_coordinates

    ontology = load_ontology_v4(_ONTOLOGY)
    return {
        spec.id
        for spec in build_specs(
            sample_coordinates(ontology, _SEED, count=count), ontology=ontology, seed=_SEED
        )
    }


def test_smoke_ids_subset_the_full_plan_but_the_guard_still_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """★ Pre-registered negative control: id subset ≠ same coordinate.

    The smoke plan is the first ``k`` units of *each modality* from cycle 0,
    while the full plan is the first ``N`` units of the single cycle-0 order.
    Their ids coincide position for position — same ``run_key``, same cycle,
    same positions — but the coordinates behind those ids differ.  An id-subset
    guard would accept this resume and mix a smoke artifact into a full-plan
    directory; the coordinate comparison must refuse it.
    """
    from ard.config import ConfigError, load_config
    from ard.pipeline import run

    smoke_dir = tmp_path / "out"
    smoke_config, _ = _real_ontology_rig(tmp_path, monkeypatch, smoke_dir, count=None)
    produced = run(load_config(smoke_config), smoke=True)
    smoke_bank = produced / "anchor_bank.jsonl"
    smoke_ids = set(_ids(smoke_bank))
    assert len(smoke_ids) == 8, "smoke plans 4 text + 4 image positions"
    smoke_records = {record["id"]: record for record in _records(smoke_bank)}

    full_config, full_generator = _real_ontology_rig(tmp_path, monkeypatch, produced, count=8)
    assert smoke_ids <= _full_plan_ids(count=8), (
        "the premise of this control: the smoke ids ARE a subset of the full plan's"
    )

    with pytest.raises(ConfigError) as refused:
        run(load_config(full_config))

    message = str(refused.value)
    assert "coordinate differs" in message, (
        "the refusal must name the coordinate mismatch, not just an unknown id"
    )
    assert "own output.directory" in message
    assert full_generator.requested == [], "nothing was generated"
    assert {record["id"]: record for record in _records(smoke_bank)} == smoke_records, (
        "the smoke bank is untouched"
    )


def test_a_large_count_writes_every_position_with_distinct_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2U positions: nothing is dropped and no id collides (v5 ruling)."""
    from ard.backends.ontology_loader import load_ontology_v4
    from ard.config import load_config
    from ard.core.sampling import unit_total
    from ard.pipeline import run

    total = unit_total(load_ontology_v4(_ONTOLOGY))
    output_dir = tmp_path / "out"
    config_path, generator = _real_ontology_rig(
        tmp_path, monkeypatch, output_dir, count=2 * total, bulk=True
    )
    run(load_config(config_path))

    ids = _ids(output_dir / "anchor_bank.jsonl")
    assert generator.requested == [2 * total]
    assert len(ids) == 2 * total, "no record may be dropped"
    assert len(set(ids)) == 2 * total, "ids are pairwise distinct"
    requested_ids = generator.requested_ids[0]
    assert len(requested_ids) == 2 * total
    assert len(set(requested_ids)) == 2 * total, (
        "the pipeline plans 2U distinct positions and drops none"
    )
    readout = _manifest(output_dir)["plan"]
    assert readout["unit_total"] == total
    assert readout["full_cycles"] == 2
    assert readout["last_cycle_size"] == 0
    assert readout["density"] == pytest.approx(2.0)
    assert readout["written_anchors"] == 2 * total
    # Coverage is the saturated coordinate-set readout: 2U distinct coordinates
    # over U saturates at 1.0 (density, the sample ratio, stays 2.0).
    assert readout["distinct_coordinates"] == 2 * total
    assert readout["coverage_ratio"] == pytest.approx(1.0)


# ── (i) static check: no coordinate-uniqueness gate exists in the pipeline ──


def test_pipeline_has_no_coordinate_uniqueness_path() -> None:
    """The uniqueness decisions are id-keyed; coordinates only describe coverage."""
    import inspect

    from ard import pipeline

    missing = inspect.getsource(pipeline._missing_plan_coordinates)
    assert 'record.get("id")' in missing and "spec.id" in missing, (
        "the pending set is a plan-minus-bank difference on ids"
    )

    guard = inspect.getsource(pipeline._refuse_foreign_records_on_resume)
    assert 'record["id"]' in guard and "spec.id" in guard
    assert "seen" not in guard, "the guard keys on the id map, never on a set of coordinates"

    # The only coordinate-set helper is the coverage readout, and it is called
    # from exactly one place — nothing filters records through it.
    source = inspect.getsource(pipeline)
    assert source.count("_distinct_coordinate_count") == 2, (
        "defined once, used once (the manifest readout)"
    )
    for line in source.splitlines():
        if "anchor_meta" in line and "set(" in line:
            raise AssertionError(f"a coordinate set appears in a uniqueness-looking line: {line}")
