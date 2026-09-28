"""Direct CLI tests for ``src/ard/cli.py`` and ``src/ard/__main__.py`` (T-4).

Covers the argument contract: ``--help`` exits 0, a missing ``--config`` is a
usage error, the removed image-conversion flag is rejected, ``--smoke`` is
forwarded to the pipeline, a missing config file fails cleanly, and the
three-tier override priority of §7.1.
"""

import logging
import subprocess
import sys
from pathlib import Path

import pytest

import ard.cli as cli


def _run_main(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Call ``cli.main()`` with a synthetic argv (may raise ``SystemExit``)."""
    monkeypatch.setattr(sys, "argv", ["ard", *argv])
    cli.main()


def _write_toml(path: Path, body: str = "# minimal config\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _pin_project_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Make project-root detection hermetic (never reach the real checkout)."""
    monkeypatch.setattr(cli, "_find_project_root", lambda _start: root)


def test_help_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run_main(["--help"], monkeypatch)
    assert excinfo.value.code == 0


def test_missing_config_is_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run_main([], monkeypatch)
    assert excinfo.value.code == 2  # argparse usage error


def test_no_convert_flag_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The old CLI flag is gone — image conversion lives in ``[images] convert``.

    §10.1 推论 2 (CLI parameters and config fields have zero intersection) does
    not allow the flag to survive as a compatibility alias: two sources for one
    parameter is a double source of truth (§1.4).
    """
    _pin_project_root(monkeypatch, tmp_path)
    config_path = tmp_path / "config.toml"
    config_path.write_text("# minimal config — every section has defaults\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _run_main(["--config", str(config_path), "--no-convert"], monkeypatch)

    assert excinfo.value.code == 2  # argparse: unrecognized argument


def test_convert_is_a_config_field() -> None:
    """The conversion switch has exactly one source: ``[images] convert``."""
    from ard.config import ARDConfig, ImageConfig

    assert "convert" in ImageConfig.model_fields
    assert ImageConfig().convert is True, "the default must match the old flag's absence"
    assert "no_convert" not in ARDConfig.model_fields


def test_pipeline_has_no_conversion_parameter() -> None:
    """``run()`` takes no conversion flag either — the config is the only source."""
    import inspect

    from ard.pipeline import run

    assert "no_convert" not in inspect.signature(run).parameters


def test_smoke_flag_is_forwarded_to_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--smoke`` reaches the pipeline as a run-boundary flag, not a config field."""
    _pin_project_root(monkeypatch, tmp_path)
    config_path = tmp_path / "config.toml"
    config_path.write_text("# minimal config — every section has defaults\n", encoding="utf-8")

    calls: list[tuple] = []
    import ard.cli as cli_mod

    def fake_run(
        config: object,
        image_dir: str | None = None,
        smoke: bool = False,
    ) -> str:
        calls.append((config, image_dir, smoke))
        return "out"

    monkeypatch.setattr(cli_mod, "run_pipeline", fake_run)
    _run_main(["--config", str(config_path), "--smoke"], monkeypatch)

    assert len(calls) == 1
    _, _, smoke = calls[0]
    assert smoke is True


def test_smoke_is_not_a_config_field() -> None:
    """§10.1 推论 2: CLI parameters and config fields have zero intersection."""
    from ard.config import ARDConfig

    assert "smoke" not in ARDConfig.model_fields


def test_missing_config_file_fails_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _pin_project_root(monkeypatch, tmp_path)
    missing = tmp_path / "nope.toml"
    with pytest.raises(SystemExit) as excinfo:
        _run_main(["--config", str(missing)], monkeypatch)
    assert excinfo.value.code == 1


def test_override_explicit_beats_local_and_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Tier 1: an explicit ``--override`` wins over both auto-detected files."""
    _pin_project_root(monkeypatch, tmp_path)
    config = _write_toml(tmp_path / "configs" / "config.toml")
    _write_toml(tmp_path / ".local" / "config.override.toml")
    _write_toml(tmp_path / "configs" / "config.override.toml")
    explicit = _write_toml(tmp_path / "explicit.toml")

    with caplog.at_level(logging.INFO):
        chosen = cli.resolve_override(config, str(explicit))

    assert chosen == explicit
    assert "explicit --override" in caplog.text
    assert str(explicit.resolve()) in caplog.text


def test_override_local_beats_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Tier 2: ``.local/config.override.toml`` wins over the adjacent file."""
    _pin_project_root(monkeypatch, tmp_path)
    config = _write_toml(tmp_path / "configs" / "config.toml")
    local = _write_toml(tmp_path / ".local" / "config.override.toml")
    _write_toml(tmp_path / "configs" / "config.override.toml")

    with caplog.at_level(logging.INFO):
        chosen = cli.resolve_override(config, None)

    assert chosen == local
    assert "tier 2" in caplog.text
    assert str(local.resolve()) in caplog.text
    assert "tier 3" not in caplog.text


def test_override_sibling_used_without_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Tier 3: the file next to ``--config`` is the backwards-compatible floor."""
    _pin_project_root(monkeypatch, tmp_path)
    config = _write_toml(tmp_path / "configs" / "config.toml")
    sibling = _write_toml(tmp_path / "configs" / "config.override.toml")

    with caplog.at_level(logging.INFO):
        chosen = cli.resolve_override(config, None)

    assert chosen == sibling
    assert "tier 3" in caplog.text
    assert str(sibling) in caplog.text


def test_no_override_logs_base_configuration_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No override anywhere: ``None`` and an explicit "base only" INFO line."""
    _pin_project_root(monkeypatch, tmp_path)
    config = _write_toml(tmp_path / "configs" / "config.toml")

    with caplog.at_level(logging.INFO):
        chosen = cli.resolve_override(config, None)

    assert chosen is None
    assert "No override config found" in caplog.text
    assert "using the base configuration only" in caplog.text


def test_explicit_override_missing_raises(tmp_path: Path) -> None:
    config = _write_toml(tmp_path / "configs" / "config.toml")
    with pytest.raises(FileNotFoundError):
        cli.resolve_override(config, str(tmp_path / "absent.toml"))


def test_foreign_local_override_is_reported_not_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A ``.local`` override outside the config's checkout is never loaded."""
    other = tmp_path / "other"
    other.mkdir()
    config = _write_toml(other / "config.toml")
    _write_toml(tmp_path / ".local" / "config.override.toml")
    monkeypatch.chdir(tmp_path)

    def fake_root(start: Path) -> Path:
        return other if Path(start) == other else tmp_path

    monkeypatch.setattr(cli, "_find_project_root", fake_root)

    with caplog.at_level(logging.INFO):
        chosen = cli.resolve_override(config, None)

    assert chosen is None
    assert "belongs to another config tree" in caplog.text
    assert "using the base configuration only" in caplog.text


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
