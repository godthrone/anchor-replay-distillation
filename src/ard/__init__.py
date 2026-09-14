"""ARD — Anchor Replay Distillation.

Multi-modal anchor data generation: prompts, teacher answers and teacher
reasoning traces, with the reasoning kept as its own field.

Version is derived by setuptools-scm from git tags — do not hardcode it here.
"""

from ard.config import ARDConfig, load_config
from ard.pipeline import run

__all__ = ["ARDConfig", "load_config", "run"]
