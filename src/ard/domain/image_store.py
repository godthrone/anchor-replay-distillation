"""Image resource management for multimodal anchors.

Scans user-provided image directories, performs random sampling,
and manages image files in the output directory — including optional
format conversion (RAW → JPG, BMP/TIFF/GIF/WEBP → JPG).
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

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


def copy_images_to_output(
    image_paths: list[Path],
    output_dir: str | Path,
) -> list[str]:
    """Copy images to output/images/ directory.

    Args:
        image_paths: Source image paths.
        output_dir: Output directory root.

    Returns:
        List of relative paths (e.g., "images/photo.jpg") for JSONL.
    """
    out = Path(output_dir)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

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
        relative_paths.append(f"images/{dst.name}")

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

        img = Image.open(str(src))
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

    Returns:
        List of relative paths (e.g. ``"images/photo.jpg"``) for JSONL.
    """
    out = Path(output_dir)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

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
            relative_paths.append(f"images/{dst.name}")
            continue

        # Collision handling — same pattern as copy_images_to_output
        if dst.exists():
            counter = 1
            while dst.exists():
                dst = images_dir / f"{stem}_{counter}{out_suffix}"
                counter += 1

        if convert_image(src, dst, quality=quality):
            relative_paths.append(f"images/{dst.name}")
        # Failure is already logged by convert_image

    logger.info(
        "Conversion complete: %d/%d images succeeded",
        len(relative_paths), len(image_paths),
    )
    return relative_paths