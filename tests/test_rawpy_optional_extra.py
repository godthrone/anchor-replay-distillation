"""Guard: ``rawpy`` stays an opt-in extra, never a default dependency.

Why
---
The ``rawpy`` wheel bundles LibRaw (LGPL-2.1 / CDDL-1.0), while every default
dependency is MIT- or Apache-licensed.  RAW camera support is therefore
installed explicitly with ``.[raw]`` (``uv sync --extra raw``), and a default
install must not pull the bundled LibRaw in.  This test checks the packaging
side of that decision in both ``pyproject.toml`` and the committed ``uv.lock``;
the runtime degradation without the extra is covered in
``tests/domain/test_image_store.py``.

``uv.lock`` carries both lists, so a lock regenerated without the extra — or a
hand edit that puts ``rawpy`` back at the root — fails here instead of shipping
silently.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_NAME = "anchor-replay-distillation"


def _toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _root_package(lock: dict[str, Any]) -> dict[str, Any]:
    for package in lock["package"]:
        if package["name"] == PROJECT_NAME:
            return package
    raise AssertionError(f"{PROJECT_NAME} missing from uv.lock")


def test_pyproject_keeps_rawpy_out_of_default_dependencies() -> None:
    project = _toml(REPO_ROOT / "pyproject.toml")["project"]

    assert not [req for req in project["dependencies"] if req.startswith("rawpy")]
    assert any(req.startswith("rawpy") for req in project["optional-dependencies"]["raw"])


def test_lock_keeps_rawpy_out_of_the_default_closure() -> None:
    root = _root_package(_toml(REPO_ROOT / "uv.lock"))

    assert not [dep for dep in root["dependencies"] if dep["name"] == "rawpy"]
    assert {dep["name"] for dep in root["optional-dependencies"]["raw"]} == {"rawpy"}
