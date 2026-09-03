"""ARD — Anchor Replay Distillation.

Multi-modal anchor data generation with log-prob export for OPD training.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    __version__ = "1.0.0.dev0"
