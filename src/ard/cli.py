"""CLI entry point for ARD.

Provides a single ``ard generate`` command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ard.config import ConfigError, load_config
from ard.logging import get_logger
from ard.pipeline import run as run_pipeline

logger = get_logger(__name__)


def _find_project_root(start: Path) -> Path | None:
    """Return the root of the checkout that contains *start*, or ``None``.

    The project root is the nearest ancestor of *start* (inclusive) that holds
    a ``.git`` entry or a ``pyproject.toml`` file — the anchors a developer
    uses to recognise the top of a checkout.  ``None`` means *start* is not
    inside a checkout, so no ``.local/`` override can claim it.
    """
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _foreign_local_override(config_dir: Path) -> Path | None:
    """Return a ``.local`` override that belongs to a *different* config tree.

    Either the current directory's checkout or the config's own checkout may
    own ``.local/config.override.toml``.  When the file exists but belongs to
    the *other* tree it must not be loaded — yet the run still has to say so
    instead of silently ignoring it (§7.1 "not silent").
    """
    config_root = _find_project_root(config_dir)
    cwd_root = _find_project_root(Path.cwd())
    if cwd_root is None:
        return None
    candidate = cwd_root / ".local" / "config.override.toml"
    if not candidate.is_file():
        return None
    if config_root is not None and config_root.resolve() == cwd_root.resolve():
        return None  # tier 2 already owns this file
    return candidate.resolve()


def resolve_override(config_path: Path, explicit: str | None) -> Path | None:
    """Resolve the override file by the canonical three-tier priority (§7.1).

    Priority, first hit wins:

    1. an explicit ``--override PATH`` — a missing file is an error;
    2. ``<project_root>/.local/config.override.toml``, where *project_root* is
       the root of the checkout that contains *config_path*: its nearest
       ancestor holding a ``.git`` entry or a ``pyproject.toml`` file;
    3. ``<config_dir>/config.override.toml``, the file sitting next to
       ``--config`` (kept for backwards compatibility).

    Every decision is logged at INFO with the full path — which override was
    loaded is never silent, and never a secret.  A ``.local`` override that
    belongs to another checkout is reported and *not* loaded.

    Returns the selected path, or ``None`` when no override exists (the run
    then uses the base configuration only).

    Raises:
        FileNotFoundError: the explicit ``--override`` path does not exist.
    """
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"override config not found: {explicit}")
        logger.info("Override config (explicit --override): %s", candidate.resolve())
        return candidate

    config_dir = config_path.resolve().parent
    project_root = _find_project_root(config_dir)
    local_override = (
        project_root / ".local" / "config.override.toml" if project_root is not None else None
    )
    if local_override is not None and local_override.is_file():
        logger.info(
            "Override config (tier 2, .local/ of project root %s): %s",
            project_root,
            local_override.resolve(),
        )
        return local_override

    sibling = config_dir / "config.override.toml"
    if sibling.is_file():
        logger.info("Override config (tier 3, next to --config): %s", sibling)
        return sibling

    foreign = _foreign_local_override(config_dir)
    if foreign is not None:
        logger.info(
            "No override applied: %s belongs to another config tree than %s; "
            "using the base configuration only",
            foreign,
            config_path,
        )
    else:
        logger.info("No override config found; using the base configuration only")
    return None


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ard",
        description="Anchor Replay Distillation — multi-modal anchor data generation",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.toml",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Image directory for multimodal mode (optional)",
    )
    parser.add_argument(
        "--override",
        default=None,
        help="Path to a config.override.toml (optional). Default: auto-detect "
        ".local/config.override.toml under the project root of --config, then "
        "config.override.toml next to --config.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Smoke run: materialise the same construction rule at a reduced "
        "scale (8 of 1826 anchors) so a fresh checkout can see an artifact "
        "quickly. The artifact is deliberately incomplete — its run directory "
        "is suffixed _smoke, the log carries a WARNING, and manifest.json "
        "declares smoke: true. It is not a configuration field and does not "
        "change a run without --smoke.",
    )

    args = parser.parse_args()

    if args.config is None:
        parser.error("--config is required")

    config_path = Path(args.config)

    # Resolve the override: explicit --override > .local/ under the project
    # root > the file next to --config (backwards compatibility).
    try:
        override_path = resolve_override(config_path, args.override)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # Load config
    try:
        config = load_config(
            str(config_path),
            str(override_path) if override_path is not None else None,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # Run pipeline
    try:
        output_dir = run_pipeline(
            config,
            image_dir=args.image_dir,
            smoke=args.smoke,
        )
        logger.info("Done! Output: %s", output_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except ConfigError as exc:
        # A boundary rejection (§2.3), not a crash: print the field-level
        # message instead of a traceback.  Nothing was written — the guard runs
        # before the output directory is created.
        logger.error("%s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
