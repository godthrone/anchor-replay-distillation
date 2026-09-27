"""Direct tests for :mod:`ard.domain.image_store` (T-4).

These are *direct* unit tests mirroring ``src/ard/domain/image_store.py`` —
the module currently has zero direct test coverage.  The copy / conversion
legs are exercised on **real** image files (Pillow-synthesised, saved to
``tmp_path``) so the true encode/decode path runs: PNG/JPEG direct copy is
byte-identical; BMP/TIFF/GIF are really re-encoded to JPEG by Pillow.

The only mocked boundary is the camera-RAW decoder (:mod:`rawpy`) — a genuine
``.CR2`` capture cannot be synthesised in a test suite.  Per T-4's boundary
rule, ``rawpy.imread`` is replaced with a fake that yields a **real** RGB
numpy array, and the array→JPEG leg (Pillow real encode) runs unmocked, so
this is a *boundary* mock (§6.1), not a mock-identity shortcut (task-C §九
"假绿" lesson).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ard.core.types import AnchorSpec, TurnSpec
from ard.domain.image_store import (
    VISUAL_DOMAIN_LAYOUT,
    convert_and_copy_images,
    convert_image,
    copy_images_to_output,
    domain_directory,
    list_domain_images,
    resolve_domain_images,
    sample_images,
    scan_images,
    select_domain_image,
)


def _make_image(path: Path, fmt: str, color: int = 120) -> Path:
    Image.new("RGB", (4, 4), color=color).save(path, fmt)
    return path


def _make_png(path: Path, color: int = 120) -> Path:
    return _make_image(path, "PNG", color)


# ── scan_images ──────────────────────────────────────────────────────────────


def test_scan_images_recursive_finds_nested_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "b.png").write_bytes(b"junk")
    (tmp_path / "a.jpg").write_bytes(b"junk")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.webp").write_bytes(b"junk")
    (sub / "note.txt").write_bytes(b"not an image")

    found = scan_images(tmp_path, recursive=True)

    assert found == [
        (tmp_path / "a.jpg").resolve(),
        (tmp_path / "b.png").resolve(),
        (sub / "c.webp").resolve(),
    ]


def test_scan_images_non_recursive_excludes_subdir(tmp_path: Path) -> None:
    (tmp_path / "top.png").write_bytes(b"junk")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.png").write_bytes(b"junk")

    found = scan_images(tmp_path, recursive=False)

    assert found == [(tmp_path / "top.png").resolve()]


def test_scan_images_respects_extensions_and_case(tmp_path: Path) -> None:
    (tmp_path / "a.PNG").write_bytes(b"junk")
    (tmp_path / "b.webp").write_bytes(b"junk")
    (tmp_path / "c.txt").write_bytes(b"junk")

    found = scan_images(tmp_path, recursive=False, extensions={".webp"})

    assert found == [(tmp_path / "b.webp").resolve()]


def test_scan_images_missing_dir_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(NotADirectoryError):
        scan_images(missing)


# ── sample_images ─────────────────────────────────────────────────────────────


def test_sample_images_is_deterministic_with_seed(tmp_path: Path) -> None:
    paths = [tmp_path / f"img_{i}.png" for i in range(10)]
    for p in paths:
        p.touch()

    first = sample_images(paths, count=4, seed=42)
    second = sample_images(paths, count=4, seed=42)

    assert first == second
    assert len(first) == 4
    assert set(first) <= set(paths)


def test_sample_images_count_at_least_length_returns_all(tmp_path: Path) -> None:
    paths = [tmp_path / f"img_{i}.png" for i in range(3)]
    for p in paths:
        p.touch()

    assert sample_images(paths, count=3, seed=1) == paths
    assert sample_images(paths, count=99, seed=1) == paths


# ── copy_images_to_output ─────────────────────────────────────────────────────


def test_copy_images_to_output_copies_and_handles_collision(tmp_path: Path) -> None:
    _make_image(tmp_path / "photo.png", "PNG")
    _make_image(tmp_path / "photo.png", "PNG", color=10)  # duplicate name

    rel = copy_images_to_output(
        [tmp_path / "photo.png", tmp_path / "photo.png"], tmp_path / "out", force=True
    )

    assert rel == ["images/photo.png", "images/photo_1.png"]
    assert (tmp_path / "out" / "images" / "photo.png").read_bytes() == (
        tmp_path / "photo.png"
    ).read_bytes()
    assert (tmp_path / "out" / "images" / "photo_1.png").is_file()


def test_copy_images_to_output_resume_reuses_instead_of_duplicating(tmp_path: Path) -> None:
    """``force=False`` (the default) is the resume branch: reuse, do not re-copy.

    The ``convert = false`` path used to have no reuse branch, so every resume
    copied the same source again as ``photo_1.png`` — one duplicate per run, with
    the bank then referencing two identical files.
    """
    _make_image(tmp_path / "photo.png", "PNG")
    out = tmp_path / "out"

    first = copy_images_to_output([tmp_path / "photo.png"], out, subdir="animals")
    assert first == ["images/animals/photo.png"]
    before = (out / "images" / "animals" / "photo.png").stat().st_mtime_ns

    resumed = copy_images_to_output([tmp_path / "photo.png"], out, subdir="animals")
    assert resumed == first, "the resume must report the file that is already there"
    assert sorted(p.name for p in (out / "images" / "animals").iterdir()) == ["photo.png"], (
        "a resume must not add a second copy under a suffixed name"
    )
    assert (out / "images" / "animals" / "photo.png").stat().st_mtime_ns == before


# ── convert_image (real format paths) ──────────────────────────────────────────


def test_convert_image_png_is_byte_identical_direct_copy(tmp_path: Path) -> None:
    src = _make_png(tmp_path / "src.png")
    before = src.read_bytes()
    dst = tmp_path / "dst.png"

    assert convert_image(src, dst) is True

    assert dst.read_bytes() == before  # PNG → PNG direct copy (§3.1 same-effect)


def test_convert_image_bmp_to_jpeg_real_encode(tmp_path: Path) -> None:
    src = _make_image(tmp_path / "src.bmp", "BMP", color=70)
    dst = tmp_path / "conv.jpg"

    assert convert_image(src, dst) is True

    out = dst.with_suffix(".jpg")
    assert out.is_file()
    with Image.open(out) as im:
        assert im.format == "JPEG"
        assert im.size == (4, 4)


def test_convert_image_tiff_to_jpeg_real_encode(tmp_path: Path) -> None:
    src = _make_image(tmp_path / "src.tiff", "TIFF", color=90)

    assert convert_image(src, tmp_path / "conv.jpg") is True

    out = tmp_path / "conv.jpg"
    assert out.is_file()
    with Image.open(out) as im:
        assert im.format == "JPEG"


def test_convert_image_unsupported_format_returns_false(tmp_path: Path) -> None:
    src = tmp_path / "src.xyz"
    src.write_bytes(b"not an image")

    assert convert_image(src, tmp_path / "conv.jpg") is False


def test_convert_image_rawpy_boundary_mock_then_real_jpeg_encode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RAW decode is mocked at the ``rawpy`` boundary; the array→JPEG is real.

    A genuine RAW capture cannot be synthesised, so ``rawpy.imread`` is mocked
    to return a fake raw object whose ``postprocess()`` yields a *real* RGB
    numpy array.  The Pillow JPEG encode then runs unmocked (real mechanism,
    not a mock-identity shortcut).

    ``rawpy`` lives in the optional ``.[raw]`` extra, so in a default
    installation there is no decoder to mock: the test skips instead of turning
    that absence into a failure.
    """
    rawpy = pytest.importorskip("rawpy")

    def fake_imread(_path: str) -> _FakeRaw:
        return _FakeRaw()

    class _FakeRaw:
        def __enter__(self) -> _FakeRaw:
            return self

        def __exit__(self, *exc: object) -> None:
            # ``None``/``False`` both mean "do not suppress the exception"; the
            # ``None`` annotation is the one mypy accepts for a context manager
            # that never swallows one.
            return None

        def postprocess(self) -> np.ndarray:
            return np.full((4, 4, 3), 33, dtype=np.uint8)

    monkeypatch.setattr(rawpy, "imread", fake_imread)

    src = tmp_path / "camera.cr2"
    src.touch()
    ok = convert_image(src, tmp_path / "conv.jpg")

    assert ok is True
    out = tmp_path / "conv.jpg"
    assert out.is_file()
    with Image.open(out) as im:
        assert im.format == "JPEG"
        assert im.size == (4, 4)


def test_convert_image_raw_without_rawpy_warns_and_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A default install (no ``.[raw]``) must degrade loudly, not fatally.

    ``sys.modules["rawpy"] = None`` makes ``import rawpy`` raise ``ImportError``
    even where the extra *is* installed, so this exercises the default-install
    branch from either environment.
    """
    monkeypatch.setitem(sys.modules, "rawpy", None)
    src = tmp_path / "camera.cr2"
    src.write_bytes(b"not a genuine RAW capture")

    with caplog.at_level(logging.WARNING, logger="ard.domain.image_store"):
        assert convert_image(src, tmp_path / "conv.jpg") is False

    assert not (tmp_path / "conv.jpg").exists()
    assert "rawpy not installed" in caplog.text
    assert str(src) in caplog.text


# ── convert_and_copy_images (real files, resume / force) ───────────────────────


def test_convert_and_copy_images_resume_reuses_skips_force_retranscodes(
    tmp_path: Path,
) -> None:
    png = _make_png(tmp_path / "src.png", color=20)
    bmp = _make_image(tmp_path / "src.bmp", "BMP", color=80)
    out = tmp_path / "out"
    sources = [png, bmp]
    png_bytes = png.read_bytes()

    # Force pass on a clean output: PNG is direct-copied losslessly (keeps its
    # basename), BMP is really transcoded to JPEG.
    first = convert_and_copy_images(sources, out, force=True)

    assert first == ["images/src.png", "images/src.jpg"]
    assert (out / "images" / "src.png").is_file()
    assert (out / "images" / "src.png").read_bytes() == png_bytes
    assert (out / "images" / "src.jpg").is_file()
    with Image.open(out / "images" / "src.jpg") as im:
        assert im.format == "JPEG"
        assert im.size == (4, 4)

    # Resume pass (force=False): both destinations already exist → same
    # relative paths reported, nothing re-converted (mtime stable).
    pre_mtime = (out / "images" / "src.jpg").stat().st_mtime_ns
    resumed = convert_and_copy_images(sources, out, force=False)
    assert resumed == ["images/src.png", "images/src.jpg"]
    assert (out / "images" / "src.jpg").stat().st_mtime_ns == pre_mtime

    # Force pass again: re-converts; the collision handler suffixes _1.
    forced = convert_and_copy_images(sources, out, force=True)
    assert forced == ["images/src_1.png", "images/src_1.jpg"]
    assert (out / "images" / "src_1.png").is_file()
    assert (out / "images" / "src_1.jpg").is_file()
    with Image.open(out / "images" / "src_1.jpg") as im:
        assert im.format == "JPEG"


# ── per-visual_domain addressing ───────────────────────────────────────────────


def _image_spec(anchor_id: str, visual_domain: str | None) -> AnchorSpec:
    """One minimally valid spec; ``None`` means a text-only coordinate."""
    meta: dict[str, str] = {"language": "English"}
    if visual_domain is not None:
        meta["modality"] = "image"
        meta["visual_domain"] = visual_domain
    else:
        meta["modality"] = "text_only"
    return AnchorSpec(
        id=anchor_id,
        anchor_meta=meta,
        turns=[TurnSpec(turn_index=0, role="user", is_final=True)],
    )


def test_the_layout_constant_names_the_convention() -> None:
    assert VISUAL_DOMAIN_LAYOUT == "<image_dir>/<visual_domain>/<image file>"


def test_domain_directory_is_the_subdirectory_named_by_the_leaf(tmp_path: Path) -> None:
    assert domain_directory(tmp_path, "animals") == tmp_path / "animals"


def test_list_domain_images_is_sorted_and_ignores_unrelated_files(tmp_path: Path) -> None:
    domain = tmp_path / "animals"
    domain.mkdir()
    _make_png(domain / "b.png")
    _make_png(domain / "a.png")
    (domain / "notes.txt").write_text("not an image", encoding="utf-8")
    nested = domain / "more"
    nested.mkdir()
    _make_png(nested / "c.png")

    found = list_domain_images(tmp_path, "animals")

    assert [p.name for p in found] == ["a.png", "b.png"]
    assert not any("c.png" in p.name for p in found), "nested files are not this domain's"


def test_list_domain_images_empty_or_missing_is_empty(tmp_path: Path) -> None:
    (tmp_path / "plants").mkdir()
    (tmp_path / "plants" / "readme.md").write_text("x", encoding="utf-8")

    assert list_domain_images(tmp_path, "plants") == []
    assert list_domain_images(tmp_path, "does_not_exist") == []


def test_select_domain_image_is_reproducible_and_seed_dependent(tmp_path: Path) -> None:
    candidates = []
    domain = tmp_path / "vehicles"
    domain.mkdir()
    for index in range(6):
        candidates.append(_make_png(domain / f"img_{index}.png", color=index * 10))

    same = select_domain_image(candidates, "vehicles", 42)
    assert select_domain_image(candidates, "vehicles", 42) == same
    assert same in candidates
    picks = {select_domain_image(candidates, "vehicles", seed) for seed in range(32)}
    assert len(picks) > 1, "the seed must actually steer the pick"


def test_select_domain_image_refuses_an_empty_candidate_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no candidate image"):
        select_domain_image([], "animals", 42)


# ── cycle rotation (v5: 1 domain, many images, one per round) ──────────────────


def _five_image_domain(tmp_path: Path, domain: str = "vehicles") -> list[Path]:
    directory = tmp_path / domain
    directory.mkdir()
    return [_make_png(directory / f"img_{index}.png", color=index * 10) for index in range(5)]


def test_select_domain_image_rotates_one_step_per_cycle_and_wraps(tmp_path: Path) -> None:
    """Every image of the domain participates; the sequence wraps at K."""
    candidates = _five_image_domain(tmp_path)

    picks = [select_domain_image(candidates, "vehicles", 0, cycle=cycle) for cycle in range(7)]

    assert picks[5] == picks[0], "cycle K must wrap back to the first pick"
    assert picks[6] == picks[1], "the rotation continues after the wrap"
    assert len(set(picks[:5])) == 5, "all five images must be shown over five rounds"
    assert set(picks[:5]) == set(candidates), "the rotation must cover the whole directory"


def test_select_domain_image_is_deterministic_per_cycle_and_cycle_zero_is_pre_v5(
    tmp_path: Path,
) -> None:
    """Same ``(seed, domain, cycle)`` ⇒ same file; ``cycle=0`` is the old pick.

    The ``cycle=0`` expectation is a golden record of the pre-v5 behaviour
    (``sha256(f"{seed}:{visual_domain}")`` seeds ``random.Random``, whose first
    ``randrange`` indexes the name-sorted candidates).  Freezing it here is what
    keeps an existing directory / sample reading the picture it already
    documents — and it is also what makes the negative control bite: a
    "rotation removed, always the first file" implementation fails this line.
    """
    candidates = _five_image_domain(tmp_path)

    assert select_domain_image(candidates, "vehicles", 0, cycle=0) == candidates[2]
    for cycle in range(4):
        first = select_domain_image(candidates, "vehicles", 0, cycle=cycle)
        assert select_domain_image(candidates, "vehicles", 0, cycle=cycle) == first
    assert select_domain_image(candidates, "vehicles", 0, cycle=1) == candidates[3]
    assert select_domain_image(candidates, "vehicles", 0, cycle=2) == candidates[4]


def test_select_domain_image_reuses_a_lone_candidate_in_every_cycle(tmp_path: Path) -> None:
    """A one-image domain legitimately reuses that image; the path says so."""
    directory = tmp_path / "animals"
    directory.mkdir()
    only = _make_png(directory / "only.png")

    picks = {select_domain_image([only], "animals", 7, cycle=cycle) for cycle in range(4)}

    assert picks == {only}, "one candidate cannot rotate; it is reused, not invented"


def test_select_domain_image_refuses_a_negative_cycle(tmp_path: Path) -> None:
    """The round index is 0-based (§2.3): a negative cycle is a caller bug."""
    candidates = _five_image_domain(tmp_path)

    with pytest.raises(ValueError, match="cycle must be >= 0"):
        select_domain_image(candidates, "vehicles", 0, cycle=-1)


def test_select_domain_image_case_mixed_names_rotate_in_sorted_order(tmp_path: Path) -> None:
    """Name sorting is the canonical order, whatever the case of name/extension."""
    directory = tmp_path / "animals"
    directory.mkdir()
    names = ["b.PNG", "A.jpeg", "c.WebP"]
    for name in names:
        (directory / name).write_bytes(b"junk")
    (directory / "notes.txt").write_bytes(b"not an image")
    candidates = list_domain_images(tmp_path, "animals")

    picks = [select_domain_image(candidates, "animals", 3, cycle=cycle).name for cycle in range(3)]

    assert picks[0] == "A.jpeg"
    assert set(picks) == set(names), "each round takes the next name-sorted file"


def test_resolve_domain_images_rotates_by_cycle_and_reports_the_readout(
    tmp_path: Path,
) -> None:
    """The per-round resolution, plus the counts that separate real from fake variety."""
    candidates = _five_image_domain(tmp_path)
    specs = [_image_spec(f"a{index}", "vehicles") for index in range(3)]
    specs.append(_image_spec("t1", None))

    raised = resolve_domain_images(tmp_path, specs, seed=0, cycle=0)
    next_round = resolve_domain_images(tmp_path, specs, seed=0, cycle=1)

    assert raised.cycle == 0
    assert next_round.cycle == 1
    assert raised.candidate_counts == {"vehicles": 5}
    assert set(raised.selected) == {"vehicles"}
    assert raised.selected["vehicles"] in candidates
    assert next_round.selected["vehicles"] != raised.selected["vehicles"]
    assert raised.selected["vehicles"] == select_domain_image(candidates, "vehicles", 0, cycle=0), (
        "the resolution must report the file it actually selected"
    )


def test_resolve_domain_images_counts_candidates_for_a_missing_domain(tmp_path: Path) -> None:
    """A missing domain still reports its (zero) candidate count, next to ``missing``."""
    _five_image_domain(tmp_path)
    specs = [_image_spec("a1", "vehicles"), _image_spec("p1", "plants")]

    resolution = resolve_domain_images(tmp_path, specs, seed=0, cycle=2)

    assert set(resolution.selected) == {"vehicles"}
    assert resolution.missing == {"plants": ["p1"]}
    assert resolution.candidate_counts == {"vehicles": 5, "plants": 0}


def test_list_domain_images_orders_mixed_case_names_and_extensions(tmp_path: Path) -> None:
    """Determinism does not depend on ``os.listdir`` order or extension case."""
    domain = tmp_path / "animals"
    domain.mkdir()
    for name in ("b.PNG", "A.jpeg", "c.WebP", "d.JPG"):
        (domain / name).write_bytes(b"junk")
    (domain / "notes.txt").write_bytes(b"not an image")

    names = [p.name for p in list_domain_images(tmp_path, "animals")]

    assert names == ["A.jpeg", "b.PNG", "c.WebP", "d.JPG"]


def test_resolve_domain_images_maps_each_leaf_to_its_own_directory(tmp_path: Path) -> None:
    for domain in ("animals", "plants"):
        directory = tmp_path / domain
        directory.mkdir()
        _make_png(directory / "only.png", color=40)

    specs = [_image_spec("a1", "animals"), _image_spec("t1", None), _image_spec("p1", "plants")]

    resolution = resolve_domain_images(tmp_path, specs, seed=7)

    assert set(resolution.missing) == set()
    assert resolution.selected["animals"] == tmp_path / "animals" / "only.png"
    assert resolution.selected["plants"] == tmp_path / "plants" / "only.png"


def test_resolve_domain_images_reports_the_affected_anchor_ids(tmp_path: Path) -> None:
    (tmp_path / "animals").mkdir()
    _make_png(tmp_path / "animals" / "a.png")
    specs = [
        _image_spec("a1", "animals"),
        _image_spec("v1", "vehicles"),
        _image_spec("v2", "vehicles"),
    ]

    resolution = resolve_domain_images(tmp_path, specs, seed=7)

    assert set(resolution.selected) == {"animals"}
    assert resolution.missing == {"vehicles": ["v1", "v2"]}


def test_convert_and_copy_images_places_files_under_the_subdir(tmp_path: Path) -> None:
    src = _make_png(tmp_path / "src.png")
    out = tmp_path / "out"

    rel = convert_and_copy_images([src], out, subdir="animals", force=True)

    assert rel == ["images/animals/src.png"]
    assert (out / "images" / "animals" / "src.png").is_file()


def test_copy_images_to_output_places_files_under_the_subdir(tmp_path: Path) -> None:
    _make_png(tmp_path / "photo.png")

    rel = copy_images_to_output([tmp_path / "photo.png"], tmp_path / "out", subdir="plants")

    assert rel == ["images/plants/photo.png"]
    assert (tmp_path / "out" / "images" / "plants" / "photo.png").is_file()
