"""Guard: ``rawpy`` (and with it numpy) stays an opt-in extra, never a default.

Why
---
The ``rawpy`` wheel bundles LibRaw (LGPL-2.1 / CDDL-1.0), while every default
dependency is MIT- or Apache-licensed.  RAW camera support is therefore
installed explicitly with ``.[raw]`` (``uv sync --extra raw``), and a default
install must not pull the bundled LibRaw in.  ``numpy`` rides in the same extra:
the RAW decode path hands a rawpy array to ``Image.fromarray``, so numpy is only
needed where rawpy is, and dropping it from the default closure is what makes a
plain install lighter.  This test checks the packaging side of that decision in
both ``pyproject.toml`` and the committed ``uv.lock``; the runtime degradation
without the extra is covered in ``tests/domain/test_image_store.py``.

``uv.lock`` carries both lists, so a lock regenerated without the extras — or a
hand edit that puts ``rawpy`` / ``numpy`` back at the root — fails here instead
of shipping silently.
"""

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


def test_pyproject_keeps_rawpy_and_numpy_out_of_default_dependencies() -> None:
    project = _toml(REPO_ROOT / "pyproject.toml")["project"]

    assert not [req for req in project["dependencies"] if req.startswith("rawpy")]
    assert not [req for req in project["dependencies"] if req.startswith("numpy")]
    assert any(req.startswith("rawpy") for req in project["optional-dependencies"]["raw"])
    assert any(req.startswith("numpy") for req in project["optional-dependencies"]["raw"])
    # The test suite imports numpy directly (tests/domain/test_image_store.py).
    assert any(req.startswith("numpy") for req in project["optional-dependencies"]["dev"])


def test_lock_keeps_rawpy_and_numpy_out_of_the_default_closure() -> None:
    root = _root_package(_toml(REPO_ROOT / "uv.lock"))

    assert not [dep for dep in root["dependencies"] if dep["name"] == "rawpy"]
    assert not [dep for dep in root["dependencies"] if dep["name"] == "numpy"]
    assert {dep["name"] for dep in root["optional-dependencies"]["raw"]} == {"numpy", "rawpy"}
    assert "numpy" in {dep["name"] for dep in root["optional-dependencies"]["dev"]}
