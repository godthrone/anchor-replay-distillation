"""CLI entry point for ARD.

Provides a single ``ard generate`` command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ard.config import load_config
from ard.pipeline import run as run_pipeline


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
        help="Path to config.override.toml (optional, default: auto-detect alongside --config)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum conversation turns (default: from config)",
    )

    args = parser.parse_args()

    if args.config is None:
        parser.error("--config is required")

    config_path = Path(args.config)

    # Auto-detect or use explicit override config
    if args.override:
        override_path: Path | None = Path(args.override)
        if not override_path.exists():
            print(f"Error: override config not found: {args.override}", file=sys.stderr)
            sys.exit(1)
    else:
        override_path = config_path.parent / "config.override.toml"
        if not override_path.exists():
            override_path = None

    # Load config
    try:
        config = load_config(
            str(config_path),
            str(override_path) if override_path is not None else None,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # CLI overrides
    if args.max_turns is not None:
        config.generation.max_turns = args.max_turns

    # Run pipeline
    try:
        output_dir = run_pipeline(config, image_dir=args.image_dir)
        print(f"\nDone! Output: {output_dir}")
    except FileExistsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
