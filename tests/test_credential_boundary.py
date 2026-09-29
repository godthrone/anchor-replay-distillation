"""The credential boundary check — readable refusal *before* any side effect.

User-visible defect under test: an unset ``api_base`` / ``model_name`` used to
surface as a bare ``ValueError: api_base must not be None`` traceback from deep
inside the HTTP client, **after** the run had already created its output
directory and written a config snapshot.  The check now happens at the top of
``pipeline.run`` (§2.3 边界校验即防呆), so the failure names the missing field
and leaves nothing behind.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from ard.backends.api_client import ChatAPIConfig
from ard.cli import main as cli_main
from ard.config import ARDConfig, ConfigError, load_config
from ard.pipeline import run

_VALID_API_BASE = "http://127.0.0.1:9/v1"


def _write_config(
    tmp_path: Path,
    *,
    input_api_base: str = _VALID_API_BASE,
    input_model_name: str = "input-model",
    target_api_base: str = _VALID_API_BASE,
    target_model_name: str = "target-model",
    output_dir: Path | None = None,
) -> Path:
    """Write a minimal valid config with the given credential fields.

    The values are written verbatim into TOML: passing ``""`` reproduces the
    ``api_base = ""`` placeholder the shipped base config uses.
    """
    out = output_dir if output_dir is not None else tmp_path / "out"
    path = tmp_path / "config.toml"
    path.write_text(
        "\n".join(
            [
                "[input_generator]",
                f'api_base = "{input_api_base}"',
                f'model_name = "{input_model_name}"',
                'api_key = ""',
                "temperature = 0.8",
                "",
                "[target_model]",
                f'api_base = "{target_api_base}"',
                f'model_name = "{target_model_name}"',
                'api_key = ""',
                "temperature = 0.1",
                "",
                "[generation]",
                # No ``target_count``: the anchor count is derived from the
                # ontology by the v4 construction rule.
                "",
                "[output]",
                f'directory = "{out.as_posix()}"',
                "overwrite = false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_empty_api_base_is_refused_and_no_output_dir_is_created(tmp_path: Path) -> None:
    """The refusal must happen before the output directory exists (§2.3)."""
    out = tmp_path / "out"
    config = load_config(_write_config(tmp_path, input_api_base="", output_dir=out))

    # The TOML placeholder "" is normalized to None at the config boundary, so
    # the check sees the same value whether the field was omitted or left blank.
    assert config.input_generator.api_base is None

    with pytest.raises(ConfigError) as excinfo:
        run(config)

    message = str(excinfo.value)
    assert "input_generator" in message
    assert "api_base" in message
    assert not out.exists(), "a refused config must not leave an output directory behind"


def test_missing_model_name_names_the_field(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, target_model_name=""))

    with pytest.raises(ConfigError) as excinfo:
        run(config)

    message = str(excinfo.value)
    assert "target_model" in message
    assert "model_name" in message
    assert not (tmp_path / "out").exists()


def test_default_config_still_constructs(tmp_path: Path) -> None:
    """The boundary lives in ``run``, not in the model.

    ``ARDConfig()`` (all defaults, no credentials) stays constructible for
    callers that only inspect/merge configs; the refusal is about *running* an
    unusable endpoint.
    """
    config = ARDConfig()
    assert config.input_generator.api_base is None
    with pytest.raises(ConfigError):
        run(config)


def test_cli_reports_the_missing_field_without_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_path = _write_config(tmp_path, input_api_base="")
    monkeypatch.setattr(sys, "argv", ["ard", "--config", str(config_path)])
    # Hermetic: auto-detection must not reach a real developer
    # .local/config.override.toml inside this checkout.
    monkeypatch.setattr("ard.cli._find_project_root", lambda _start: None)

    with caplog.at_level(logging.ERROR, logger="ard.cli"):
        with pytest.raises(SystemExit) as excinfo:
            cli_main()

    assert excinfo.value.code == 1
    # ``ard.logging`` binds its StreamHandler to the stdout it saw at import
    # time, so assert on the log record rather than on the captured stream.
    assert "input_generator" in caplog.text
    assert "api_base" in caplog.text
    assert "Traceback" not in caplog.text


def test_blank_api_key_means_no_credential() -> None:
    """``api_key = ""`` must normalize to ``None``, never to an empty bearer."""
    endpoint = ChatAPIConfig(api_base=_VALID_API_BASE, model_name="m", api_key="")
    assert endpoint.api_key is None
