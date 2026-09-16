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

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ard.domain.image_store import (
    convert_and_copy_images,
    convert_image,
    copy_images_to_output,
    sample_images,
    scan_images,
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
        [tmp_path / "photo.png", tmp_path / "photo.png"], tmp_path / "out"
    )

    assert rel == ["images/photo.png", "images/photo_1.png"]
    assert (tmp_path / "out" / "images" / "photo.png").read_bytes() == (
        tmp_path / "photo.png"
    ).read_bytes()
    assert (tmp_path / "out" / "images" / "photo_1.png").is_file()


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
    """
    import rawpy

    def fake_imread(_path: str) -> _FakeRaw:
        return _FakeRaw()

    class _FakeRaw:
        def __enter__(self) -> _FakeRaw:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

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
