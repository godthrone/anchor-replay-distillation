"""Image resource management for multimodal anchors.

Responsibility: own the **image addressing convention** of a run — an
image-modality anchor's picture is looked up under
``<image_dir>/<visual_domain>/`` so the content actually matches the
``visual_domain`` coordinate the anchor is labelled with — plus which of that
domain's files a round shows (``cycle`` rotation, v5) and the copying /
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
from dataclasses import dataclass, field
from pathlib import Path

from ard.core.types import AnchorSpec, StringList

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

RAW_EXTENSIONS = {
    ".cr2",
    ".nef",
    ".arw",
    ".dng",
    ".orf",
    ".rw2",
    ".raf",
    ".pef",
    ".srw",
    ".3fr",
    ".erf",
    ".mef",
    ".mrw",
    ".nrw",
    ".ptx",
    ".r3d",
    ".rwl",
    ".srf",
    ".x3f",
}

CONVERTABLE_EXTENSIONS = SUPPORTED_EXTENSIONS | {".bmp", ".tiff", ".tif"} | RAW_EXTENSIONS


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
        paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
    else:
        paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in extensions]

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
# offers, which file a round (``cycle``) picks, and which required domains are
# missing.  v5 rotates the pick by round, so a domain with K images shows all K
# of them over K rounds instead of pinning the whole run to one file.

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


def select_domain_image(
    candidates: list[Path],
    visual_domain: str,
    seed: int,
    *,
    cycle: int = 0,
) -> Path:
    """Pick the image round *cycle* shows for *visual_domain*, deterministically (§6 复现).

    The candidates are first put in a canonical order (file name), then indexed
    by a generator derived from ``sha256(f"{seed}:{visual_domain}")``.  Hashing
    instead of seeding ``random.Random(str)`` keeps the pick stable across
    Python versions and free of ``hash()``'s per-process salt, and mixing the
    domain in means two domains that happen to hold the same file names do not
    simply pick the same index.

    **Rotation (v5).**  ``cycle`` advances the index by one per round:
    ``ordered[(offset + cycle) % len(ordered)]``.  Since the index wraps, a
    domain whose directory holds ``K`` usable images shows **all K of them over
    K rounds** — a run's image variety grows with its round count instead of
    being pinned to one file per domain.  ``cycle=0`` reproduces the pre-v5 pick
    exactly, so an existing directory or sample keeps the picture it already
    documents.  Same ``(candidates, visual_domain, seed, cycle)`` ⇒ same file,
    on any platform and any run.

    Every anchor of the domain shares that round's one image by design (the v5
    ruling rotates by round, not by anchor).  The returned path is the caller's
    evidence of *which* file that is, so "this domain really has several images"
    and "this domain has one image, necessarily reused" stay distinguishable.

    Args:
        candidates: Non-empty list from :func:`list_domain_images`.
        visual_domain: The leaf, mixed into the seed.
        seed: The run's generation seed (recorded in ``<output_dir>/config.toml``).
        cycle: The 0-based round index, with the same meaning as the sampling
            layer's "第 c 轮" (``c, _ = divmod(plan_index, U)``).  Default 0.

    Raises:
        ValueError: If *candidates* is empty — a caller must have refused the
            missing domain at the boundary instead of asking to select from it —
            or if *cycle* is negative (the round index is 0-based).
    """
    if not candidates:
        raise ValueError(f"visual_domain {visual_domain!r} has no candidate image")
    if cycle < 0:
        raise ValueError(f"cycle must be >= 0 (a 0-based round index), got {cycle}")
    ordered = sorted(candidates, key=lambda p: p.name)
    digest = hashlib.sha256(f"{seed}:{visual_domain}".encode()).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    offset = rng.randrange(len(ordered))
    return ordered[(offset + cycle) % len(ordered)]


@dataclass(frozen=True, slots=True)
class DomainImageResolution:
    """The images one round's visual domains resolved to, and what is missing.

    Attributes:
        selected: ``visual_domain -> source image`` for every required domain
            that has at least one usable file, in first-required order.
        missing: ``visual_domain -> anchor ids`` for every required domain that
            has none.  The ids are exactly the samples a skipping run drops, so
            the caller can count them and name them one by one.
        cycle: The 0-based round this resolution describes — the ``cycle`` it
            was asked for.
        candidate_counts: ``visual_domain -> number of usable files`` for every
            required domain (0 for a missing one).  This is the readout that
            separates a domain which can rotate (more than one candidate) from
            one that necessarily reuses its single image.
    """

    selected: dict[str, Path]
    missing: dict[str, StringList]
    cycle: int = 0
    candidate_counts: dict[str, int] = field(default_factory=dict)


def resolve_domain_images(
    image_dir: str | Path,
    specs: Iterable[AnchorSpec],
    *,
    seed: int,
    cycle: int = 0,
    extensions: set[str] | None = None,
) -> DomainImageResolution:
    """Resolve the source image each ``visual_domain`` uses in round *cycle*.

    The resolution is **per round**, because v5 rotates a domain's images: the
    same domain resolves to a different file in cycle 0, 1, 2, … .  A caller
    whose specs span several rounds must therefore call this once per round
    (group the specs by their cycle) and keep the results apart — passing a
    multi-round spec list to one call would silently pin every round to the same
    image.

    Only anchors whose coordinate names a ``visual_domain`` are considered:
    a text-only anchor has no visual coordinate, so attaching an image to it
    would be the same coordinate/content mismatch this addressing exists to
    prevent.

    A domain directory that is **missing, empty, or holds no usable image**
    keeps its pre-v5 behaviour: it is *not* an error here — the domain is
    reported through ``missing`` (with the anchor ids it affects) and the run
    decides at its boundary, either refusing the plan up front or, with
    ``[images] skip_missing_images = true``, dropping those anchors with one
    WARNING each and declaring them in ``manifest.json``.

    Args:
        image_dir: Root of the image tree (``--image-dir``).
        specs: The anchors this run will generate **in round** *cycle*.
        seed: The run seed, forwarded to :func:`select_domain_image`.
        cycle: The 0-based round index, forwarded to
            :func:`select_domain_image`.  Default 0.
        extensions: Allowed extensions; defaults to ``SUPPORTED_EXTENSIONS``.

    Returns:
        A :class:`DomainImageResolution`; never raises for a missing domain —
        refusing (or skipping) is the caller's decision at the boundary.
    """
    required: dict[str, StringList] = {}
    for spec in specs:
        domain = spec.anchor_meta.get("visual_domain")
        if not isinstance(domain, str):
            continue
        required.setdefault(domain, []).append(spec.id)

    selected: dict[str, Path] = {}
    missing: dict[str, StringList] = {}
    candidate_counts: dict[str, int] = {}
    for domain, anchor_ids in required.items():
        candidates = list_domain_images(image_dir, domain, extensions=extensions)
        candidate_counts[domain] = len(candidates)
        if candidates:
            chosen = select_domain_image(candidates, domain, seed, cycle=cycle)
            selected[domain] = chosen
            logger.info(
                "visual_domain %r: round %d uses %s (%d usable image(s) under %s)",
                domain,
                cycle,
                chosen.name,
                len(candidates),
                domain_directory(image_dir, domain),
            )
        else:
            missing[domain] = anchor_ids
    return DomainImageResolution(
        selected=selected,
        missing=missing,
        cycle=cycle,
        candidate_counts=candidate_counts,
    )


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
    force: bool = False,
    subdir: str | None = None,
) -> list[str]:
    """Copy images to output/images/ directory.

    When ``force=False`` (the default) an existing file at the destination is
    treated as already placed and skipped — this is the **resume** scenario, and
    it mirrors :func:`convert_and_copy_images`'s ``force=False`` branch so both
    ``[images] convert`` settings behave the same on a second run.  Without it a
    ``convert = false`` resume copied every domain image again as
    ``name_1.ext``, growing one duplicate per run.  Set ``force=True`` to copy
    unconditionally (the collision handler then suffixes ``_1``).

    Args:
        image_paths: Source image paths.
        output_dir: Output directory root.
        force: If ``True``, copy even when the destination already exists.
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
        # Resume: an existing destination is the file a previous run placed;
        # reuse it instead of writing a ``name_1`` duplicate.  One file may be
        # shared by several anchors, which is why the reference is returned
        # rather than the copy being skipped silently.
        if dst.exists() and not force:
            logger.info("Skipping %s (already exists)", dst.name)
            relative_paths.append(_relative_image_path(dst.name, subdir))
            continue
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
            logger.warning("rawpy not installed, cannot convert RAW image: %s", src)
            return False
        try:
            with rawpy.imread(str(src)) as raw:
                rgb = raw.postprocess()
            from PIL import Image  # noqa: PLC0415

            dst_jpg = dst.with_suffix(".jpg")
            Image.fromarray(rgb).save(str(dst_jpg), "JPEG", quality=quality)
            return True
        except Exception as exc:  # noqa: BLE001 — rawpy raises across its own
            # exception hierarchy per decoder; the conversion is best-effort and the
            # failure is reported per image at WARNING (§3.2 transparent).
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
    except Exception as exc:  # noqa: BLE001 — Pillow raises across its own
        # exception hierarchy per format; the conversion is best-effort and the
        # failure is reported per image at WARNING (§3.2 transparent).
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
        len(image_paths),
        images_dir,
        quality,
        force,
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
        len(relative_paths),
        len(image_paths),
    )
    return relative_paths
