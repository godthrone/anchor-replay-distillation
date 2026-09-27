"""Per-``visual_domain`` image addressing in ``pipeline.run`` (WP-S6d, WP-13).

The behaviour frozen here is the user's ruling, translated into four contracts:

1. an image-modality anchor's picture is read from
   ``<image_dir>/<visual_domain>/`` **when that directory holds one** — the
   domain-matched picture is always preferred;
2. a domain whose own directory holds no usable image **reuses** one from the
   tree-wide pool instead of losing its anchors: a user who supplied few
   pictures (even a single one) still gets a complete run, and the reuse is
   declared per ``(round, domain)`` and in aggregate in ``manifest.json``;
3. only a tree with **no usable image at all** is refused by default, before
   the output directory exists, naming the domains left without a picture and
   the directories they would be read from;
4. setting ``[images] skip_missing_images = true`` (the §3.3 预授权退路)
   applies to that same no-image-anywhere state only: those anchors are skipped
   instead — one WARNING per dropped anchor and a machine-readable declaration
   in ``manifest.json``, never a silent shortfall.

No network is reachable: the generator is replaced by a spy that appends a
minimal valid record per spec, and the API clients the pipeline builds point at
a closed local port but are never called.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ard.config import ARDConfig, ConfigError, load_config
from ard.core.types import AnchorSpec, DataSource, GeneratedAnchor, StringList, TurnSpec
from ard.domain.bank import append_anchor
from ard.domain.text_anchor import AnchorGenerationStats

#: The production ontology.  The acceptance phase is enabled in these configs
#: and its structure readout is defined against the run's own sampling space, so
#: the config must point at a real ontology even though the plan is injected.
_ONTOLOGY = str(Path(__file__).resolve().parents[1] / "ontology" / "anchor_ontology.v4.json")


def _spec(anchor_id: str, visual_domain: str | None) -> AnchorSpec:
    """One minimally valid spec; ``None`` means a text-only coordinate."""
    meta: dict[str, str] = {"language": "English", "knowledge_domain": "math"}
    if visual_domain is not None:
        meta["modality"] = "image"
        meta["visual_domain"] = visual_domain
    return AnchorSpec(
        id=anchor_id,
        anchor_meta=meta,
        turns=[TurnSpec(turn_index=0, role="user", is_final=True)],
    )


class _GeneratorSpy:
    """Appends one records-valid anchor per spec; records the plans it saw."""

    def __init__(self) -> None:
        self.requested: list[StringList] = []
        self.image_paths: list[str | None] = []

    def __call__(self, **kwargs: object) -> list[GeneratedAnchor]:
        specs = kwargs["specs"]
        output_path = kwargs["output_path"]
        stats = kwargs["stats"]
        assert isinstance(specs, list)
        assert isinstance(output_path, Path)
        assert isinstance(stats, AnchorGenerationStats)
        self.requested.append([spec.id for spec in specs])
        written: list[GeneratedAnchor] = []
        for spec in specs:
            assert isinstance(spec, AnchorSpec)
            self.image_paths.append(spec.turns[0].image_path)
            anchor = GeneratedAnchor(
                id=spec.id,
                messages=[{"role": "user", "content": f"question {spec.id}"}],
                target_answer=f"answer {spec.id}",
                target_model="target-model",
                input_generator_model="input-model",
                anchor_meta=dict(spec.anchor_meta),
                data_source=(
                    DataSource.ARD_MULTI if spec.turns[0].image_path else DataSource.ARD_TEXT
                ),
            )
            assert append_anchor(anchor, output_path).value == "appended"
            written.append(anchor)
        stats.requested = len(specs)
        stats.written = len(written)
        stats.succeeded = len(written)
        return written


def _write_config(
    path: Path,
    output_dir: Path,
    *,
    seed: int = 7,
    skip_missing_images: bool = False,
    convert: bool = True,
) -> None:
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
        f"seed = {seed}",
        "concurrency = 1",
        "",
        "[ontology]",
        f'path = "{_ONTOLOGY}"',
        "",
        "[output]",
        f'directory = "{output_dir}"',
        "overwrite = false",
        "",
        "[images]",
        f"skip_missing_images = {'true' if skip_missing_images else 'false'}",
        f"convert = {'true' if convert else 'false'}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _rig(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    specs: list[AnchorSpec],
    *,
    skip_missing_images: bool = False,
    seed: int = 7,
    convert: bool = True,
) -> tuple[Path, _GeneratorSpy, Callable[[object], list[AnchorSpec]]]:
    """Config + spy + plan double; the ontology is never loaded.

    The double returns a fresh copy on every call: the pipeline mutates the
    plan (image assignment), and a sampler is supposed to hand out new specs.
    """
    from ard import pipeline

    output_dir = tmp_path / "out"
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        output_dir,
        seed=seed,
        skip_missing_images=skip_missing_images,
        convert=convert,
    )
    spy = _GeneratorSpy()
    monkeypatch.setattr(
        pipeline,
        "sample_specs",
        lambda config: pytest.fail("the plan double must replace the v4 sampler"),
    )
    monkeypatch.setattr(pipeline, "generate_text_anchors", spy)
    return output_dir, spy, (lambda config: copy.deepcopy(specs))


def _image_dir(tmp_path: Path, domains: dict[str, int]) -> Path:
    """``tmp_path/images`` with *count* tiny PNGs under each named domain."""
    from PIL import Image

    root = tmp_path / "images"
    for domain, count in domains.items():
        directory = root / domain
        directory.mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (4, 4), color=index * 20).save(directory / f"img_{index}.png", "PNG")
    return root


def _manifest(output_dir: Path) -> dict:
    return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))


# ── 1. per-domain addressing ──────────────────────────────────────────────────


def test_each_image_anchor_gets_a_picture_from_its_own_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ard.pipeline import run

    images = _image_dir(tmp_path, {"animals": 2, "plants": 1})
    specs = [_spec("t1", None), _spec("a1", "animals"), _spec("p1", "plants")]
    output_dir, spy, plan = _rig(tmp_path, monkeypatch, specs)

    run(load_config(tmp_path / "config.toml"), image_dir=str(images), generate_specs=plan)

    assert spy.requested == [["t1", "a1", "p1"]]
    text_path, animal_path, plant_path = spy.image_paths
    assert text_path is None, "a text-only coordinate must never receive an image"
    assert animal_path is not None and "/images/animals/" in animal_path
    assert plant_path is not None and "/images/plants/" in plant_path
    assert (output_dir / "images" / "animals").is_dir()
    assert (output_dir / "images" / "plants").is_dir()


def test_images_rotate_by_round_through_the_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v5: one domain's picture advances by round, and the rounds are resolved apart.

    Reaching round 1 through the real rule needs more than ``U`` anchors, so the
    plan context is forced to ``U = 1``: two positions then span two rounds.  The
    code under test is the real wiring — group the pending specs by
    ``cycle_of``, call ``resolve_domain_images`` once per round, copy each
    ``(round, domain)`` source, and assign every spec its own round's file.
    """
    from ard import pipeline
    from ard.pipeline import run

    images = _image_dir(tmp_path, {"animals": 2})
    specs = [_spec("r0", "animals"), _spec("r1", "animals")]
    output_dir, spy, plan = _rig(tmp_path, monkeypatch, specs)

    original_context = pipeline._injected_plan_context

    def forced_two_rounds(injected: list[AnchorSpec], *, ontology: object = None) -> object:
        context = original_context(injected, ontology=ontology)
        return replace(context, unit_total=1, rounds=(len(context.specs), 0))

    monkeypatch.setattr(pipeline, "_injected_plan_context", forced_two_rounds)

    run(load_config(tmp_path / "config.toml"), image_dir=str(images), generate_specs=plan)

    first, second = spy.image_paths
    assert first is not None and second is not None
    assert first != second, (
        "each round must use its own file: grouping by domain alone pins both "
        "rounds to the same picture"
    )
    manifest = _manifest(output_dir)
    assert manifest["images"]["domain_candidate_counts"] == {"animals": 2}
    assert manifest["images"]["pool_candidate_count"] == 2
    assert manifest["images"]["fallback_visual_domains"] == []
    assert manifest["images"]["fallback_anchor_count"] == 0
    assert [
        (row["cycle"], row["visual_domain"]) for row in manifest["images"]["resolved_images"]
    ] == [(0, "animals"), (1, "animals")]
    assert all(row["fallback"] is False for row in manifest["images"]["resolved_images"])
    files = {Path(first).name, Path(second).name}
    assert len(files) == 2
    assert (output_dir / "images" / "animals").is_dir()


def test_the_output_records_reference_the_domain_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ard.pipeline import run

    images = _image_dir(tmp_path, {"animals": 1})
    output_dir, _spy, plan = _rig(tmp_path, monkeypatch, [_spec("a1", "animals")])

    run(load_config(tmp_path / "config.toml"), image_dir=str(images), generate_specs=plan)

    record = json.loads((output_dir / "anchor_bank.jsonl").read_text(encoding="utf-8").strip())
    assert record["data_source"] == "ard_multi"

    manifest = _manifest(output_dir)
    assert manifest["images"]["addressing"] == "<image_dir>/<visual_domain>/<image file>"
    assert manifest["images"]["skipped_anchor_count"] == 0
    assert manifest["images"]["resolved_visual_domains"] == ["animals"]


def test_same_seed_is_byte_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same image tree + same seed ⇒ same picture, same bytes in the artifact."""
    from ard.pipeline import run

    images = _image_dir(tmp_path, {"animals": 4, "plants": 3})
    specs = [_spec("a1", "animals"), _spec("p1", "plants")]

    digests: list[str] = []
    for attempt in range(2):
        root = tmp_path / f"attempt_{attempt}"
        root.mkdir()
        output_dir, spy, plan = _rig(root, monkeypatch, specs, seed=1234)
        run(load_config(root / "config.toml"), image_dir=str(images), generate_specs=plan)
        selected = [
            Path(path).relative_to(output_dir.parent).as_posix() for path in spy.image_paths if path
        ]
        copied = sorted(
            f"{p.relative_to(output_dir).as_posix()}:{hashlib.sha256(p.read_bytes()).hexdigest()}"
            for p in output_dir.rglob("*")
            if p.is_file() and p.suffix == ".png"
        )
        digests.append(hashlib.sha256("\n".join([*selected, *copied]).encode()).hexdigest())

    assert digests[0] == digests[1]


# ── 2. missing images: refuse by default, before any side effect ──────────────


def test_an_unstocked_domain_reuses_the_pool_and_declares_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A domain with no directory of its own is served from the pool, not dropped.

    The old rule made this plan impossible (``vehicles`` had no directory, so its
    two anchors were refused or skipped).  The ruling replaces it with reuse: the
    anchors are generated, shown a picture the user did supply, and the
    substitution is named both per ``(round, domain)`` and in aggregate.
    """
    from ard.pipeline import run

    images = _image_dir(tmp_path, {"animals": 1})
    specs = [_spec("a1", "animals"), _spec("v1", "vehicles"), _spec("v2", "vehicles")]
    output_dir, spy, plan = _rig(tmp_path, monkeypatch, specs)

    run(load_config(tmp_path / "config.toml"), image_dir=str(images), generate_specs=plan)

    assert spy.requested == [["a1", "v1", "v2"]], "no anchor is dropped for want of a picture"
    animal_path, vehicle_one, vehicle_two = spy.image_paths
    assert animal_path is not None and "/images/animals/" in animal_path
    # One source file is placed once and shared by every (round, domain) that
    # selected it, so the reused picks reference the placed copy rather than a
    # second `images/vehicles/` duplicate — the manifest, not the path, is what
    # declares which anchors are served by reuse.
    assert Path(vehicle_one).name == "img_0.png"
    assert Path(vehicle_one).is_file() and Path(vehicle_two).is_file()

    manifest = _manifest(output_dir)
    images_section = manifest["images"]
    assert images_section["skipped_anchor_count"] == 0
    assert images_section["fallback_visual_domains"] == ["vehicles"]
    assert images_section["fallback_anchor_count"] == 2
    assert images_section["pool_candidate_count"] == 1
    assert images_section["domain_candidate_counts"] == {"animals": 1, "vehicles": 0}
    rows = {(row["cycle"], row["visual_domain"]): row for row in images_section["resolved_images"]}
    assert rows[(0, "animals")]["fallback"] is False
    assert rows[(0, "vehicles")]["fallback"] is True


def test_a_missing_image_tree_is_refused_with_domains_paths_and_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No usable image anywhere: the refusal names the domains and the count."""
    from ard.pipeline import run

    root = tmp_path / "images"
    (root / "animals").mkdir(parents=True)
    (root / "animals" / "notes.md").write_text("not an image", encoding="utf-8")
    specs = [_spec("a1", "animals"), _spec("v1", "vehicles"), _spec("v2", "vehicles")]
    output_dir, spy, plan = _rig(tmp_path, monkeypatch, specs)

    with pytest.raises(ConfigError) as excinfo:
        run(load_config(tmp_path / "config.toml"), image_dir=str(root), generate_specs=plan)

    message = str(excinfo.value)
    assert "vehicles" in message, "the domain left without a picture must be named"
    assert str(root / "vehicles") in message, "the directory it would be read from must be named"
    assert "2 anchor(s)" in message, "the affected anchor count must be named"
    assert "Total affected anchors: 3 of 3 planned sample(s)" in message
    assert "no image at all" in message, "the refusal must say why reuse cannot help"
    assert "skip_missing_images" in message, "the way out must be pointed at"
    assert spy.requested == [], "nothing may be generated after a refusal"
    assert not output_dir.exists(), "the refusal must precede the output directory (§2.3)"


def test_a_missing_image_directory_is_refused_before_the_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ard.pipeline import run

    output_dir, _spy, plan = _rig(tmp_path, monkeypatch, [_spec("a1", "animals")])

    with pytest.raises(ConfigError, match="image directory not found"):
        run(
            load_config(tmp_path / "config.toml"),
            image_dir=str(tmp_path / "nowhere"),
            generate_specs=plan,
        )

    assert not output_dir.exists()


def test_a_three_domain_plan_runs_on_a_one_picture_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """★ 主席点名的场景, end to end: one picture, three domains, no anchor lost.

    Two of the three domains have no directory at all, so the old rule would
    have refused the plan (or dropped most of it).  The run must complete, every
    anchor must end up referencing a file, and the manifest must say that two
    domains were served by reuse.
    """
    from ard.pipeline import run

    images = _image_dir(tmp_path, {"animals": 1})
    specs = [_spec("a1", "animals"), _spec("p1", "plants"), _spec("v1", "vehicles")]
    output_dir, spy, plan = _rig(tmp_path, monkeypatch, specs)

    run(load_config(tmp_path / "config.toml"), image_dir=str(images), generate_specs=plan)

    assert spy.requested == [["a1", "p1", "v1"]], "no anchor may be lost on a one-picture tree"
    assert all(path is not None for path in spy.image_paths)
    # The single source file is placed once and shared by all three domains; the
    # manifest (not a per-domain duplicate) carries the reuse declaration.
    placed = sorted(p.relative_to(output_dir).as_posix() for p in output_dir.rglob("*.png"))
    assert placed == ["images/animals/img_0.png"], "one source, one placed copy"
    for path in spy.image_paths:
        assert path is not None and Path(path).is_file(), "no reference may dangle"

    images_section = _manifest(output_dir)["images"]
    assert images_section["pool_candidate_count"] == 1
    assert images_section["fallback_visual_domains"] == ["plants", "vehicles"]
    assert images_section["fallback_anchor_count"] == 2
    assert images_section["skipped_anchor_count"] == 0


# ── 3. the opt-in skip switch ─────────────────────────────────────────────────


def test_skip_switch_defaults_to_false() -> None:
    assert ARDConfig.model_validate({}).images.skip_missing_images is False


def test_skip_switch_is_declared_in_both_configs() -> None:
    for path in ("configs/config.toml", "configs/config.override.sample.toml"):
        text = Path(path).read_text(encoding="utf-8")
        assert "[images]" in text
        assert "skip_missing_images" in text


def test_skip_missing_images_warns_and_declares_each_dropped_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The skip switch governs the one state where reuse cannot help: no image anywhere."""
    from ard.pipeline import run

    root = tmp_path / "images"
    (root / "animals").mkdir(parents=True)
    (root / "animals" / "notes.md").write_text("not an image", encoding="utf-8")
    specs = [_spec("t1", None), _spec("a1", "animals"), _spec("v1", "vehicles")]
    output_dir, spy, plan = _rig(tmp_path, monkeypatch, specs, skip_missing_images=True)

    with caplog.at_level("WARNING"):
        run(load_config(tmp_path / "config.toml"), image_dir=str(root), generate_specs=plan)

    assert spy.requested == [["t1"]], "every unsupported anchor is not generated"
    skipped = [
        record.getMessage() for record in caplog.records if "Skipping anchor" in record.getMessage()
    ]
    assert len(skipped) == 2, "one WARNING per dropped anchor"
    assert any("a1" in message and "animals" in message for message in skipped)
    assert any("v1" in message and "vehicles" in message for message in skipped)

    manifest = _manifest(output_dir)
    assert manifest["images"]["skip_missing_images"] is True
    assert manifest["images"]["skipped_anchor_count"] == 2
    assert manifest["images"]["skipped_visual_domains"] == ["animals", "vehicles"]
    records = (output_dir / "anchor_bank.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 1, "the artifact is short and the manifest says why"


def test_resuming_a_skip_run_stays_declared_and_writes_no_new_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The second run has one anchor left — the skipped one — and must not loop."""
    from ard.pipeline import run

    root = tmp_path / "images"
    root.mkdir()
    specs = [_spec("a1", "animals"), _spec("v1", "vehicles")]
    output_dir, spy, plan = _rig(tmp_path, monkeypatch, specs, skip_missing_images=True)
    config = load_config(tmp_path / "config.toml")

    with caplog.at_level("WARNING"):
        run(config, image_dir=str(root), generate_specs=plan)
        bank = output_dir / "anchor_bank.jsonl"
        before = bank.read_text(encoding="utf-8") if bank.exists() else None
        run(config, image_dir=str(root), generate_specs=plan)

    assert spy.requested == [], "there is no image anywhere, so nothing is generated"
    bank = output_dir / "anchor_bank.jsonl"
    assert (bank.read_text(encoding="utf-8") if bank.exists() else None) == before
    assert any(
        "Every remaining anchor (2) was skipped" in record.getMessage() for record in caplog.records
    )
    assert _manifest(output_dir)["images"]["skipped_visual_domains"] == ["animals", "vehicles"]


# ── 4. an abandoned anchor leaves no picture behind ─────────────────────────


class _AbandoningGeneratorSpy(_GeneratorSpy):
    """Writes every spec except *abandoned*, then declares those dropped.

    This is the shape a real failed run produces: the abandoned anchors never
    reach ``anchor_bank.jsonl``, but their pictures were already copied into
    ``output/images/`` before generation started.
    """

    def __init__(self, abandoned: set[str]) -> None:
        super().__init__()
        self.abandoned = set(abandoned)
        self.seen_plan: StringList = []

    def __call__(self, **kwargs: object) -> list[GeneratedAnchor]:
        specs = kwargs["specs"]
        stats = kwargs["stats"]
        assert isinstance(specs, list)
        assert isinstance(stats, AnchorGenerationStats)
        self.seen_plan = [spec.id for spec in specs if isinstance(spec, AnchorSpec)]
        kept = [
            spec for spec in specs if isinstance(spec, AnchorSpec) and spec.id not in self.abandoned
        ]
        written = super().__call__(**{**kwargs, "specs": kept})
        stats.abandoned_total = len(self.abandoned)
        stats.abandoned_by_reason = {"transport_error": len(self.abandoned)}
        return written


def _bank_image_reference(abs_path: str) -> str:
    """The bank's persisted form of *abs_path*: ``images/<domain>/<file>``."""
    index = abs_path.rfind("/images/")
    return abs_path[index + 1 :]


def test_convert_false_resume_reuses_the_placed_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``convert = false`` resume reuses the placed image instead of copying `_1` (B3).

    ``copy_images_to_output`` had no ``force=False`` reuse branch, so a resume
    that still had one anchor to generate copied the domain image again as
    ``img_1.png``/``img_0_1.png`` — one duplicate per run, with the bank left
    referencing two identical files.  The ``convert = true`` path already reused.
    """
    from ard.pipeline import run

    images = _image_dir(tmp_path, {"animals": 1})
    specs = [_spec("a1", "animals"), _spec("a2", "animals")]
    output_dir, _spy, plan = _rig(tmp_path, monkeypatch, specs, convert=False)
    config = load_config(tmp_path / "config.toml")

    # run 1 abandons a2, as if its generation had failed and the run stopped.
    monkeypatch.setattr("ard.pipeline.generate_text_anchors", _AbandoningGeneratorSpy({"a2"}))
    run(config, image_dir=str(images), generate_specs=plan)
    placed = sorted(p.name for p in (output_dir / "images" / "animals").iterdir())
    assert placed == ["img_0.png"], "a1 survives, so the shared picture must survive too"

    # run 2 resumes a2 in the same domain: reuse the file, do not copy a second.
    monkeypatch.setattr("ard.pipeline.generate_text_anchors", _GeneratorSpy())
    run(config, image_dir=str(images), generate_specs=plan)

    assert sorted(p.name for p in (output_dir / "images" / "animals").iterdir()) == ["img_0.png"], (
        "a resume must not place a suffixed duplicate of an image already in the run dir"
    )
    records = (output_dir / "anchor_bank.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 2


class _ImagePartGeneratorSpy(_GeneratorSpy):
    """Writes records that reference their picture the way the bank really does.

    ``_GeneratorSpy`` stores plain-string content, which cannot express "this
    record references ``images/<domain>/<file>``" — the exact input the prune
    ownership rule reads.  Content here mirrors
    ``text_anchor._convert_images_to_paths``: the persisted part is
    ``{"type": "image", "image": "<output-relative path>"}``.
    """

    def __init__(self, abandoned: set[str] | None = None) -> None:
        super().__init__()
        self.abandoned = set(abandoned or ())
        self.seen_plan: StringList = []

    def __call__(self, **kwargs: object) -> list[GeneratedAnchor]:
        specs = kwargs["specs"]
        output_path = kwargs["output_path"]
        stats = kwargs["stats"]
        assert isinstance(specs, list)
        assert isinstance(output_path, Path)
        assert isinstance(stats, AnchorGenerationStats)
        self.seen_plan = [spec.id for spec in specs if isinstance(spec, AnchorSpec)]
        kept = [
            spec for spec in specs if isinstance(spec, AnchorSpec) and spec.id not in self.abandoned
        ]
        written: list[GeneratedAnchor] = []
        for spec in kept:
            rel = next(
                (_bank_image_reference(turn.image_path) for turn in spec.turns if turn.image_path),
                None,
            )
            content: list[dict[str, Any]] = [{"type": "text", "text": f"question {spec.id}"}]
            if rel is not None:
                content.append({"type": "image", "image": rel})
            anchor = GeneratedAnchor(
                id=spec.id,
                messages=[{"role": "user", "content": content}],
                target_answer=f"answer {spec.id}",
                target_model="target-model",
                input_generator_model="input-model",
                anchor_meta=dict(spec.anchor_meta),
                data_source=DataSource.ARD_MULTI if rel else DataSource.ARD_TEXT,
            )
            assert append_anchor(anchor, output_path).value == "appended"
            written.append(anchor)
        stats.abandoned_total = len(self.abandoned)
        stats.abandoned_by_reason = (
            {"transport_error": len(self.abandoned)} if self.abandoned else {}
        )
        stats.requested = len(specs)
        stats.written = len(written)
        stats.succeeded = len(written)
        return written


def test_an_abandoned_anchor_has_its_image_removed_but_shared_files_survive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A dropped anchor leaves no picture; a picture shared by survivors stays."""
    from ard import pipeline
    from ard.pipeline import run

    images = _image_dir(tmp_path, {"animals": 1, "plants": 1})
    # The one `animals` picture is shared by a1 and a2; `plants` has a single
    # anchor and that anchor is the one the run abandons.
    specs = [_spec("a1", "animals"), _spec("a2", "animals"), _spec("p1", "plants")]
    output_dir, _spy, plan = _rig(tmp_path, monkeypatch, specs)
    spy = _AbandoningGeneratorSpy({"p1"})
    monkeypatch.setattr(pipeline, "generate_text_anchors", spy)

    with caplog.at_level("INFO"):
        run(load_config(tmp_path / "config.toml"), image_dir=str(images), generate_specs=plan)

    assert spy.seen_plan == ["a1", "a2", "p1"]

    survivors = sorted(p.relative_to(output_dir).as_posix() for p in output_dir.rglob("*.png"))
    assert survivors == ["images/animals/img_0.png"], (
        "only the picture still referenced by a surviving anchor may remain, "
        f"and it must not be the abandoned one's: {survivors}"
    )
    assert not list((output_dir / "images" / "plants").glob("*")), (
        "the abandoned anchor's picture must be removed from the artifact"
    )
    assert "Removed image images/plants/img_0.png" in caplog.text

    written = {
        json.loads(line)["id"]
        for line in (output_dir / "anchor_bank.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert written == {"a1", "a2"}, "the artifact itself is unaffected by the cleanup"


def test_a_failed_image_cleanup_is_a_warning_not_a_run_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Housekeeping must never take the run down (§3.2: announce, do not crash)."""
    from ard import pipeline
    from ard.pipeline import run

    images = _image_dir(tmp_path, {"animals": 1, "plants": 1})
    specs = [_spec("a1", "animals"), _spec("p1", "plants")]
    output_dir, _spy, plan = _rig(tmp_path, monkeypatch, specs)
    monkeypatch.setattr(pipeline, "generate_text_anchors", _AbandoningGeneratorSpy({"p1"}))

    real_unlink = Path.unlink

    def _refuse_plants(self: Path, missing_ok: bool = False) -> None:
        if "plants" in self.as_posix():
            raise OSError("simulated read-only artifact")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", _refuse_plants)

    with caplog.at_level("WARNING"):
        returned = run(
            load_config(tmp_path / "config.toml"), image_dir=str(images), generate_specs=plan
        )

    assert returned == output_dir, "a failed unlink must not fail the run"
    assert (output_dir / "images" / "plants" / "img_0.png").exists()
    assert "Could not remove image images/plants/img_0.png" in caplog.text
    assert (output_dir / "anchor_bank.jsonl").exists()


# ── 5. resume: a picture a pre-existing record references is not prunable ────


def test_resuming_reuses_a_picture_that_stays_referenced_by_the_old_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Run 2 resumes: the pending anchor reuses the file and is then dropped.

    Run 1 writes ``a1`` and its picture.  Run 2 has only ``p1`` pending, in the
    same domain (so the same file); the copy is skipped because the file
    already exists and ``p1`` is abandoned.  ``a1``'s record still references
    the file, so deleting it leaves a dangling path in ``anchor_bank.jsonl``.
    """
    from ard import pipeline
    from ard.pipeline import run

    images = _image_dir(tmp_path, {"animals": 1})
    specs = [_spec("a1", "animals"), _spec("p1", "animals")]
    output_dir, _spy, _plan = _rig(tmp_path, monkeypatch, specs)
    config = load_config(tmp_path / "config.toml")

    # Both runs hand the pipeline the *same* plan: since S22 the resume guard is
    # keyed on the plan identity, so a shorter run-1 plan would be a different
    # plan and would be refused before the pruning logic under test runs.  Run 1
    # abandons ``p1``, so only ``a1`` reaches the bank while the identity is the
    # full plan's — exactly the state a real interrupted resume leaves behind.
    monkeypatch.setattr(pipeline, "generate_text_anchors", _ImagePartGeneratorSpy({"p1"}))
    run(config, image_dir=str(images), generate_specs=lambda _config: copy.deepcopy(specs))
    picture = output_dir / "images" / "animals" / "img_0.png"
    assert picture.exists(), "precondition: run 1 leaves the picture behind"

    monkeypatch.setattr(pipeline, "generate_text_anchors", _ImagePartGeneratorSpy({"p1"}))
    with caplog.at_level("INFO"):
        run(config, image_dir=str(images), generate_specs=lambda _config: copy.deepcopy(specs))

    assert "Skipping img_0.png (already exists)" in caplog.text, (
        "precondition: the resume must reuse the existing file, not rewrite it"
    )
    assert picture.exists(), "the picture a1 still references must survive the resume run"
    assert "Removed image images/animals/img_0.png" not in caplog.text

    records = [
        json.loads(line)
        for line in (output_dir / "anchor_bank.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["id"] for record in records] == ["a1"]
    references = [
        part["image"]
        for message in records[0]["messages"]
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image"
    ]
    assert references == ["images/animals/img_0.png"]
    assert (output_dir / references[0]).exists(), "no reference in the bank may dangle"


def test_a_picture_no_record_references_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The rule must stay a rule: nothing references the file → it is removed."""
    from ard import pipeline
    from ard.pipeline import run

    images = _image_dir(tmp_path, {"animals": 1, "plants": 1})
    specs = [_spec("a1", "animals"), _spec("p1", "plants")]
    output_dir, _spy, plan = _rig(tmp_path, monkeypatch, specs)
    monkeypatch.setattr(pipeline, "generate_text_anchors", _ImagePartGeneratorSpy({"p1"}))

    with caplog.at_level("INFO"):
        run(load_config(tmp_path / "config.toml"), image_dir=str(images), generate_specs=plan)

    assert not list((output_dir / "images" / "plants").glob("*")), (
        "a picture whose only owner was abandoned, and which no record references, must be removed"
    )
    assert "Removed image images/plants/img_0.png" in caplog.text
    assert (output_dir / "images" / "animals" / "img_0.png").exists()


def test_a_picture_shared_by_a_surviving_record_is_not_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Shared protection across a resume: p1 drops, p2 and the old a1 keep X."""
    from ard import pipeline
    from ard.pipeline import run

    images = _image_dir(tmp_path, {"animals": 1})
    specs = [_spec("a1", "animals"), _spec("p1", "animals"), _spec("p2", "animals")]
    output_dir, _spy, _plan = _rig(tmp_path, monkeypatch, specs)
    config = load_config(tmp_path / "config.toml")

    # Same plan in both runs (the S22 identity guard forbids a plan change);
    # run 1 abandons both pending anchors, so only ``a1`` lands in the bank.
    monkeypatch.setattr(pipeline, "generate_text_anchors", _ImagePartGeneratorSpy({"p1", "p2"}))
    run(config, image_dir=str(images), generate_specs=lambda _config: copy.deepcopy(specs))

    monkeypatch.setattr(pipeline, "generate_text_anchors", _ImagePartGeneratorSpy({"p1"}))
    with caplog.at_level("INFO"):
        run(config, image_dir=str(images), generate_specs=lambda _config: copy.deepcopy(specs))

    picture = output_dir / "images" / "animals" / "img_0.png"
    assert picture.exists(), "one file shared by several anchors must never be pruned"
    assert "Removed image" not in caplog.text
    ids = [
        json.loads(line)["id"]
        for line in (output_dir / "anchor_bank.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert ids == ["a1", "p2"]
