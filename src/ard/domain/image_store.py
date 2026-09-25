"""Image resource management for multimodal anchors.

Responsibility: own the **image addressing convention** of a run — an
image-modality anchor's picture is looked up under
``<image_dir>/<visual_domain>/`` so the content actually matches the
``visual_domain`` coordinate the anchor is labelled with — plus the copying /
format-conversion of the selected files into ``<output_dir>/images/``.

The addressing convention is the only layout the pipeline uses.  The flat
helpers :func:`scan_images` / :func:`sample_images` remain as generic utilities
(and are exercised directly by the unit tests); a run never picks an image from
a cross-domain pool, because a pool cannot tell whether the picture matches the
label it is attached to.
"""

from __future__ import annotations

import hashlib
import logging
import random
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ard.core.types import AnchorSpec

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

RAW_EXTENSIONS = {
    ".cr2", ".nef", ".arw", ".dng", ".orf", ".rw2",
    ".raf", ".pef", ".srw", ".3fr", ".erf", ".mef",
    ".mrw", ".nrw", ".ptx", ".r3d", ".rwl", ".srf",
    ".x3f",
}

CONVERTABLE_EXTENSIONS = (
    SUPPORTED_EXTENSIONS
    | {".bmp", ".tiff", ".tif"}
    | RAW_EXTENSIONS
)


def scan_images(
    image_dir: str | Path,
    recursive: bool = True,
    *,
    extensions: set[str] | None = None,
) -> list[Path]:
    """Scan a directory for image files.

    Args:
        image_dir: Root directory to scan.
        recursive: If True, scan subdirectories recursively.
        extensions: Allowed file extensions. Defaults to
            :data:`SUPPORTED_EXTENSIONS` (PNG/JPEG/GIF/WEBP).

    Returns:
        Sorted list of absolute Paths to image files.
    """
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS

    root = Path(image_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Image directory not found: {root}")

    if recursive:
        paths = [
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        ]
    else:
        paths = [
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]

    return sorted(paths)


def sample_images(
    image_paths: list[Path],
    count: int,
    seed: int = 42,
) -> list[Path]:
    """Sample a subset of images randomly.

    Args:
        image_paths: All available image paths.
        count: Number of images to sample.
        seed: Random seed.

    Returns:
        Sampled image paths.
    """
    if count >= len(image_paths):
        return list(image_paths)

    rng = random.Random(seed)
    return rng.sample(image_paths, count)


# ── Per-visual_domain addressing (the run convention) ──────────────────────
#
# Every image-modality anchor carries exactly one ``visual_domain`` coordinate
# (see :mod:`ard.core.sampling`).  The image that anchor is shown must therefore
# be addressed *by that coordinate*, not drawn from a flat pool: a flat pool
# cannot guarantee that the picture matches the label it is attached to, and a
# mismatch silently mislabels the anchor.  One subdirectory per leaf:
#
#     <image_dir>/<visual_domain>/<image file>
#
# This section owns that convention end to end: which files a domain directory
# offers, which single file a run picks, and which required domains are missing.

#: The addressing convention, spelled out for error messages and docs.
VISUAL_DOMAIN_LAYOUT = "<image_dir>/<visual_domain>/<image file>"


def domain_directory(image_dir: str | Path, visual_domain: str) -> Path:
    """Return ``<image_dir>/<visual_domain>`` (the directory may not exist)."""
    return Path(image_dir) / visual_domain


def list_domain_images(
    image_dir: str | Path,
    visual_domain: str,
    *,
    extensions: set[str] | None = None,
) -> list[Path]:
    """List the usable image files of one ``visual_domain``, deterministically.

    Only *direct children* of ``<image_dir>/<visual_domain>/`` count: a nested
    directory is not an image, and accepting files from deeper levels would make
    "the images of this domain" ambiguous.  Files whose extension is not in
    *extensions* (default :data:`SUPPORTED_EXTENSIONS`) are ignored, so notes,
    sidecars and thumbnails never become anchors' content.  The result is sorted
    by file name, so it does not depend on filesystem enumeration order.

    Args:
        image_dir: Root of the image tree the user passed via ``--image-dir``.
        visual_domain: The leaf whose directory to list.
        extensions: Allowed extensions; defaults to ``SUPPORTED_EXTENSIONS``.

    Returns:
        Sorted paths; empty when the directory is missing or holds no image.
    """
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS
    directory = domain_directory(image_dir, visual_domain)
    if not directory.is_dir():
        return []
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions),
        key=lambda p: p.name,
    )


def select_domain_image(candidates: list[Path], visual_domain: str, seed: int) -> Path:
    """Pick one image from *candidates*, deterministically (§6 复现).

    The candidates are first put in a canonical order (file name), then indexed
    by a generator derived from ``sha256(f"{seed}:{visual_domain}")``.  Hashing
    instead of seeding ``random.Random(str)`` keeps the pick stable across
    Python versions and free of ``hash()``'s per-process salt, and mixing the
    domain in means two domains that happen to hold the same file names do not
    simply pick the same index.  Same ``(candidates, visual_domain, seed)`` ⇒
    same file, on any platform and any run; a different seed picks differently.

    Args:
        candidates: Non-empty list from :func:`list_domain_images`.
        visual_domain: The leaf, mixed into the seed.
        seed: The run's generation seed (recorded in ``<output_dir>/config.json``).

    Raises:
        ValueError: If *candidates* is empty — a caller must have refused the
            missing domain at the boundary instead of asking to select from it.
    """
    if not candidates:
        raise ValueError(f"visual_domain {visual_domain!r} has no candidate image")
    ordered = sorted(candidates, key=lambda p: p.name)
    digest = hashlib.sha256(f"{seed}:{visual_domain}".encode()).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    return ordered[rng.randrange(len(ordered))]


@dataclass(frozen=True, slots=True)
class DomainImageResolution:
    """The images a plan's visual domains resolved to, and what is missing.

    Attributes:
        selected: ``visual_domain -> source image`` for every required domain
            that has at least one usable file, in first-required order.
        missing: ``visual_domain -> anchor ids`` for every required domain that
            has none.  The ids are exactly the samples a skipping run drops, so
            the caller can count them and name them one by one.
    """

    selected: dict[str, Path]
    missing: dict[str, list[str]]


def resolve_domain_images(
    image_dir: str | Path,
    specs: Iterable[AnchorSpec],
    *,
    seed: int,
    extensions: set[str] | None = None,
) -> DomainImageResolution:
    """Resolve one source image per ``visual_domain`` the *specs* require.

    Only anchors whose coordinate names a ``visual_domain`` are considered:
    a text-only anchor has no visual coordinate, so attaching an image to it
    would be the same coordinate/content mismatch this addressing exists to
    prevent.

    Args:
        image_dir: Root of the image tree (``--image-dir``).
        specs: The anchors this run will generate.
        seed: The run seed, forwarded to :func:`select_domain_image`.
        extensions: Allowed extensions; defaults to ``SUPPORTED_EXTENSIONS``.

    Returns:
        A :class:`DomainImageResolution`; never raises for a missing domain —
        refusing (or skipping) is the caller's decision at the boundary.
    """
    required: dict[str, list[str]] = {}
    for spec in specs:
        domain = spec.anchor_meta.get("visual_domain")
        if not isinstance(domain, str):
            continue
        required.setdefault(domain, []).append(spec.id)

    selected: dict[str, Path] = {}
    missing: dict[str, list[str]] = {}
    for domain, anchor_ids in required.items():
        candidates = list_domain_images(image_dir, domain, extensions=extensions)
        if candidates:
            selected[domain] = select_domain_image(candidates, domain, seed)
        else:
            missing[domain] = anchor_ids
    return DomainImageResolution(selected=selected, missing=missing)


def _images_dir(output_dir: str | Path, subdir: str | None) -> Path:
    """Return (and create) ``<output_dir>/images[/<subdir>]``."""
    images_dir = Path(output_dir) / "images"
    if subdir is not None:
        images_dir = images_dir / subdir
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def _relative_image_path(name: str, subdir: str | None) -> str:
    """The JSONL path of an image placed at ``images[/<subdir>]/<name>``."""
    if subdir is None:
        return f"images/{name}"
    return f"images/{subdir}/{name}"


def copy_images_to_output(
    image_paths: list[Path],
    output_dir: str | Path,
    *,
    subdir: str | None = None,
) -> list[str]:
    """Copy images to output/images/ directory.

    Args:
        image_paths: Source image paths.
        output_dir: Output directory root.
        subdir: Optional subdirectory of ``images/`` (the run passes the
            ``visual_domain`` so the output tree mirrors the addressing
            convention).  ``None`` keeps the historical flat layout.

    Returns:
        List of relative paths (e.g., "images/photo.jpg") for JSONL.
    """
    images_dir = _images_dir(output_dir, subdir)

    relative_paths: list[str] = []
    for src in image_paths:
        dst = images_dir / src.name
        if dst.exists():
            stem = src.stem
            suffix = src.suffix
            counter = 1
            while dst.exists():
                dst = images_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        shutil.copy2(src, dst)
        relative_paths.append(_relative_image_path(dst.name, subdir))

    return relative_paths


# ── Image format conversion ────────────────────────────────────────────────


def convert_image(src: Path, dst: Path, quality: int = 95) -> bool:
    """Convert a single image to the target output format.

    * ``.png`` → direct copy (lossless, no re-encoding).
    * ``.jpg`` / ``.jpeg`` → direct copy (avoids double lossy compression).
    * RAW (``.cr2``, ``.nef``, ``.arw``, ``.dng``, …) → ``rawpy`` decode →
      Pillow encode as JPEG.
    * Other (``.bmp``, ``.tiff``, ``.gif``, ``.webp``) → Pillow open →
      encode as JPEG.

    Args:
        src: Source image path.
        dst: Destination path (extension determines output format).
        quality: JPEG quality (1–100). Only used when encoding to JPEG.

    Returns:
        ``True`` if conversion succeeded, ``False`` otherwise.
    """
    suffix = src.suffix.lower()

    # PNG / JPEG: direct copy — no re-encoding
    if suffix in {".png", ".jpg", ".jpeg"}:
        try:
            shutil.copy2(src, dst)
            return True
        except OSError as exc:
            logger.warning("Failed to copy %s to %s: %s", src, dst, exc)
            return False

    # RAW formats: rawpy → Pillow JPEG
    if suffix in RAW_EXTENSIONS:
        try:
            import rawpy  # noqa: PLC0415 — optional dependency
        except ImportError:
            logger.warning(
                "rawpy not installed, cannot convert RAW image: %s", src
            )
            return False
        try:
            with rawpy.imread(str(src)) as raw:
                rgb = raw.postprocess()
            from PIL import Image  # noqa: PLC0415

            dst_jpg = dst.with_suffix(".jpg")
            Image.fromarray(rgb).save(str(dst_jpg), "JPEG", quality=quality)
            return True
        except Exception as exc:
            logger.warning("Failed to convert RAW image %s: %s", src, exc)
            return False

    # Other formats: Pillow open → JPEG
    try:
        from PIL import Image  # noqa: PLC0415

        # ``Image.open`` is typed as returning ``ImageFile`` while ``convert``
        # returns ``Image``; annotating the wider base lets the reassignment
        # below (RGBA/P/LA → RGB) keep one variable, as the code reads.
        img: Image.Image = Image.open(str(src))
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        dst_jpg = dst.with_suffix(".jpg")
        img.save(str(dst_jpg), "JPEG", quality=quality)
        return True
    except Exception as exc:
        logger.warning("Failed to convert image %s: %s", src, exc)
        return False


def convert_and_copy_images(
    image_paths: list[Path],
    output_dir: str | Path,
    *,
    force: bool = False,
    quality: int = 95,
    subdir: str | None = None,
) -> list[str]:
    """Convert images to output-compatible formats and place in
    ``output_dir/images/``.

    * ``.png`` and ``.jpg`` / ``.jpeg`` keep their original extension.
    * All other formats are transcoded to ``.jpg``.

    When ``force=False`` (the default) an existing file at the destination
    is treated as already-processed and skipped — this is the **resume**
    scenario.  Set ``force=True`` to re-convert every image.

    Args:
        image_paths: Source image paths.
        output_dir: Output directory root.
        force: If ``True``, re-convert even when the destination already
            exists.
        quality: JPEG quality (1–100).
        subdir: Optional subdirectory of ``images/`` (the run passes the
            ``visual_domain`` so the output tree mirrors the addressing
            convention).  ``None`` keeps the historical flat layout.

    Returns:
        List of relative paths (e.g. ``"images/photo.jpg"``) for JSONL.
    """
    images_dir = _images_dir(output_dir, subdir)

    logger.info(
        "Converting %d images to %s (quality=%d, force=%s)…",
        len(image_paths), images_dir, quality, force,
    )

    relative_paths: list[str] = []
    for src in image_paths:
        suffix = src.suffix.lower()

        # Determine output extension
        if suffix in {".png", ".jpg", ".jpeg"}:
            out_suffix = suffix
        else:
            out_suffix = ".jpg"

        stem = src.stem
        dst = images_dir / f"{stem}{out_suffix}"

        # Resume: skip when destination already exists
        if dst.exists() and not force:
            logger.info("Skipping %s (already exists)", dst.name)
            relative_paths.append(_relative_image_path(dst.name, subdir))
            continue

        # Collision handling — same pattern as copy_images_to_output
        if dst.exists():
            counter = 1
            while dst.exists():
                dst = images_dir / f"{stem}_{counter}{out_suffix}"
                counter += 1

        if convert_image(src, dst, quality=quality):
            relative_paths.append(_relative_image_path(dst.name, subdir))
        # Failure is already logged by convert_image

    logger.info(
        "Conversion complete: %d/%d images succeeded",
        len(relative_paths), len(image_paths),
    )
    return relative_paths
