"""Per-``visual_domain`` image addressing in ``pipeline.run`` (WP-S6d).

The behaviour frozen here is the user's ruling, translated into three contracts:

1. an image-modality anchor's picture is read from
   ``<image_dir>/<visual_domain>/`` — never from a cross-domain pool, so the
   image always matches the coordinate it is attached to;
2. a required visual domain with no usable image is **refused by default**,
   before the output directory exists, with the missing domains, their expected
   paths and the affected anchor count in the message;
3. setting ``[images] skip_missing_images = true`` (the §3.3 预授权退路) skips
   those anchors instead — one WARNING per dropped anchor and a machine-readable
   declaration in ``manifest.json``, never a silent shortfall.

No network is reachable: the generator is replaced by a spy that appends a
minimal valid record per spec, and the API clients the pipeline builds point at
a closed local port but are never called.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from ard.config import ARDConfig, ConfigError, load_config
from ard.core.types import AnchorSpec, DataSource, GeneratedAnchor, TurnSpec
from ard.domain.bank import append_anchor
from ard.domain.text_anchor import AnchorGenerationStats


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
        self.requested: list[list[str]] = []
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
        'path = "unused.json"',
        "",
        "[output]",
        f'directory = "{output_dir}"',
        "overwrite = false",
        "",
        "[images]",
        f"skip_missing_images = {'true' if skip_missing_images else 'false'}",
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
) -> tuple[Path, _GeneratorSpy, Callable[[object], list[AnchorSpec]]]:
    """Config + spy + plan double; the ontology is never loaded.

    The double returns a fresh copy on every call: the pipeline mutates the
    plan (image assignment), and a sampler is supposed to hand out new specs.
    """
    from ard import pipeline

    output_dir = tmp_path / "out"
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path, output_dir, seed=seed, skip_missing_images=skip_missing_images
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
            Image.new("RGB", (4, 4), color=index * 20).save(
                directory / f"img_{index}.png", "PNG"
            )
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


def test_the_output_records_reference_the_domain_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ard.pipeline import run

    images = _image_dir(tmp_path, {"animals": 1})
    output_dir, _spy, plan = _rig(tmp_path, monkeypatch, [_spec("a1", "animals")])

    run(load_config(tmp_path / "config.toml"), image_dir=str(images), generate_specs=plan)

    record = json.loads(
        (output_dir / "anchor_bank.jsonl").read_text(encoding="utf-8").strip()
    )
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
            Path(path).relative_to(output_dir.parent).as_posix()
            for path in spy.image_paths
            if path
        ]
        copied = sorted(
            f"{p.relative_to(output_dir).as_posix()}:"
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}"
            for p in output_dir.rglob("*")
            if p.is_file() and p.suffix == ".png"
        )
        digests.append(
            hashlib.sha256("\n".join([*selected, *copied]).encode()).hexdigest()
        )

    assert digests[0] == digests[1]


# ── 2. missing images: refuse by default, before any side effect ──────────────


def test_a_missing_domain_is_refused_with_domains_paths_and_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ard.pipeline import run

    images = _image_dir(tmp_path, {"animals": 1})
    specs = [_spec("a1", "animals"), _spec("v1", "vehicles"), _spec("v2", "vehicles")]
    output_dir, spy, plan = _rig(tmp_path, monkeypatch, specs)

    with pytest.raises(ConfigError) as excinfo:
        run(load_config(tmp_path / "config.toml"), image_dir=str(images), generate_specs=plan)

    message = str(excinfo.value)
    assert "vehicles" in message, "the missing visual_domain must be named"
    assert str(images / "vehicles") in message, "its expected path must be named"
    assert "2 anchor(s)" in message, "the affected anchor count must be named"
    assert "Total affected anchors: 2 of 3 planned sample(s)" in message
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


def test_an_empty_or_unrelated_domain_directory_counts_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ard.pipeline import run

    root = tmp_path / "images"
    (root / "animals").mkdir(parents=True)
    (root / "animals" / "notes.md").write_text("not an image", encoding="utf-8")
    output_dir, _spy, plan = _rig(tmp_path, monkeypatch, [_spec("a1", "animals")])

    with pytest.raises(ConfigError, match="no usable image"):
        run(load_config(tmp_path / "config.toml"), image_dir=str(root), generate_specs=plan)

    assert not output_dir.exists()


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
    from ard.pipeline import run

    images = _image_dir(tmp_path, {"animals": 1})
    specs = [_spec("t1", None), _spec("a1", "animals"), _spec("v1", "vehicles")]
    output_dir, spy, plan = _rig(
        tmp_path, monkeypatch, specs, skip_missing_images=True
    )

    with caplog.at_level("WARNING"):
        run(load_config(tmp_path / "config.toml"), image_dir=str(images), generate_specs=plan)

    assert spy.requested == [["t1", "a1"]], "the unsupported anchor is not generated"
    skipped = [
        record.getMessage()
        for record in caplog.records
        if "Skipping anchor" in record.getMessage()
    ]
    assert len(skipped) == 1, "one WARNING per dropped anchor"
    assert "v1" in skipped[0] and "vehicles" in skipped[0]

    manifest = _manifest(output_dir)
    assert manifest["images"]["skip_missing_images"] is True
    assert manifest["images"]["skipped_anchor_count"] == 1
    assert manifest["images"]["skipped_visual_domains"] == ["vehicles"]
    records = (output_dir / "anchor_bank.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 2, "the artifact is short and the manifest says why"


def test_resuming_a_skip_run_stays_declared_and_writes_no_new_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The second run has one anchor left — the skipped one — and must not loop."""
    from ard.pipeline import run

    images = _image_dir(tmp_path, {"animals": 1})
    specs = [_spec("a1", "animals"), _spec("v1", "vehicles")]
    output_dir, spy, plan = _rig(tmp_path, monkeypatch, specs, skip_missing_images=True)
    config = load_config(tmp_path / "config.toml")

    with caplog.at_level("WARNING"):
        run(config, image_dir=str(images), generate_specs=plan)
        before = (output_dir / "anchor_bank.jsonl").read_text(encoding="utf-8")
        run(config, image_dir=str(images), generate_specs=plan)

    assert spy.requested == [["a1"]], "the second run generates nothing new"
    assert (output_dir / "anchor_bank.jsonl").read_text(encoding="utf-8") == before
    assert any(
        "Every remaining anchor (1) was skipped" in record.getMessage()
        for record in caplog.records
    )
    assert _manifest(output_dir)["images"]["skipped_visual_domains"] == ["vehicles"]
