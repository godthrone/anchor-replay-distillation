"""Direct CLI tests for ``src/ard/cli.py`` and ``src/ard/__main__.py`` (T-4).

Covers the argument contract: ``--help`` exits 0, a missing ``--config`` is a
usage error, ``--no-convert`` is forwarded to the pipeline, and a missing
config file fails cleanly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import ard.cli as cli


def _run_main(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Call ``cli.main()`` with a synthetic argv (may raise ``SystemExit``)."""
    monkeypatch.setattr(sys, "argv", ["ard", *argv])
    cli.main()


def test_help_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run_main(["--help"], monkeypatch)
    assert excinfo.value.code == 0


def test_missing_config_is_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run_main([], monkeypatch)
    assert excinfo.value.code == 2  # argparse usage error


def test_no_convert_is_forwarded_to_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("# minimal config — every section has defaults\n", encoding="utf-8")

    calls: list[tuple] = []
    import ard.cli as cli_mod

    def fake_run(config: object, image_dir: str | None = None, no_convert: bool = False) -> str:
        calls.append((config, image_dir, no_convert))
        return "out"

    monkeypatch.setattr(cli_mod, "run_pipeline", fake_run)
    _run_main(["--config", str(config_path), "--no-convert"], monkeypatch)

    assert len(calls) == 1
    _, image_dir, no_convert = calls[0]
    assert no_convert is True
    assert image_dir is None


def test_missing_config_file_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "nope.toml"
    with pytest.raises(SystemExit) as excinfo:
        _run_main(["--config", str(missing)], monkeypatch)
    assert excinfo.value.code == 1


def test_python_dash_m_ard_help(tmp_path: Path) -> None:
    """``python -m ard --help`` wires ``__main__`` to the CLI and exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "ard", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Anchor Replay Distillation" in result.stdout
