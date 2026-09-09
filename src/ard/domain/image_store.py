"""Image resource management for multimodal anchors.

Scans user-provided image directories, performs random sampling,
and manages image files in the output directory.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def scan_images(image_dir: str | Path, recursive: bool = True) -> list[Path]:
    """Scan a directory for image files.

    Args:
        image_dir: Root directory to scan.
        recursive: If True, scan subdirectories recursively.

    Returns:
        Sorted list of absolute Paths to image files.
    """
    root = Path(image_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Image directory not found: {root}")

    if recursive:
        paths = [
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    else:
        paths = [
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
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
