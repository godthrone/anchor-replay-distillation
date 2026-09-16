"""Output-snapshot secret redaction (R7).

The merged config carries API credentials, and the pipeline writes it into the
output directory (``config.json`` and the manifest's ``config`` section).
Output directories get shared and packed, so those credentials must never be
written verbatim — constitution §2.3 (boundary check before data lands on
disk).

Every credential used here is a fake literal; no real key appears anywhere.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from ard.config import load_config
from ard.pipeline import (
    NOT_SECRET_KEY_WORDS,
    REDACTED_KEY_WORDS,
    REDACTED_PLACEHOLDER,
    _is_secret_key,
    _redact_secrets,
    run,
)

FAKE_INPUT_KEY = "sk-test-DEADBEEF"
FAKE_TARGET_KEY = "sk-target-CAFEBABE"
# The substrings asserted absent from the produced artefacts — chosen so the
# assertion cannot be satisfied by a partial mask of the key name/value.
FORBIDDEN_MARKERS = ("DEADBEEF", "CAFEBABE")


def _write_config(tmp_path: Path, output_dir: Path, target_count: int = 5) -> Path:
    """Write a real config.toml carrying fake credentials.

    Built from the project's own ``configs/config.toml`` so the test exercises
    the same config shape the pipeline ships with.
    """
    repo_root = Path(__file__).resolve().parents[1]
    base = tomllib.loads((repo_root / "configs" / "config.toml").read_text(encoding="utf-8"))
    base["generation"]["target_count"] = target_count
    base["output"]["directory"] = str(output_dir)
    base["input_generator"]["api_key"] = FAKE_INPUT_KEY
    base["target_model"]["api_key"] = FAKE_TARGET_KEY
    for section in ("input_generator", "target_model"):
        base[section]["model_name"] = f"{section}-model"
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml_dump(base), encoding="utf-8")
    return config_path


def _toml_dump(data: dict) -> str:
    """Minimal nested-dict TOML writer (str/int/float/bool/list leaves)."""
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"[{key}]")
            for sub_key, sub_value in value.items():
                lines.append(f"{sub_key} = {_toml_value(sub_value)}")
            lines.append("")
    return "\n".join(lines)


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return str(value)


class TestSnapshotRedactionOnRealWritePath:
    """End-to-end: run the pipeline and inspect the artefacts on disk."""

    def test_config_snapshot_does_not_contain_the_api_keys(self, tmp_path: Path) -> None:
        """A real ``run()`` must not write the fake keys into ``config.json``.

        ``target_count = 0`` with an empty bank takes the early-return branch,
        which writes the snapshot and the manifest without generating anchors
        or contacting any endpoint.
        """
        output_dir = tmp_path / "out"
        config = load_config(str(_write_config(tmp_path, output_dir, target_count=0)))

        result_dir = run(config)

        snapshot_text = (result_dir / "config.json").read_text(encoding="utf-8")
        for marker in FORBIDDEN_MARKERS:
            assert marker not in snapshot_text, (
                f"{marker} leaked into config.json — the snapshot is not redacted"
            )
        assert REDACTED_PLACEHOLDER in snapshot_text, (
            "the masked placeholder is missing, so redaction cannot be confirmed "
            "from the artefact alone"
        )

    def test_config_snapshot_keeps_non_secret_fields_verbatim(self, tmp_path: Path) -> None:
        """Redaction must not disturb the snapshot's reproducibility purpose."""
        output_dir = tmp_path / "out"
        config = load_config(str(_write_config(tmp_path, output_dir, target_count=0)))

        result_dir = run(config)
        snapshot = json.loads((result_dir / "config.json").read_text(encoding="utf-8"))

        # File name, location and indentation are unchanged; other fields keep
        # their real values, so the snapshot still reproduces the run.
        assert snapshot["output"]["directory"] == str(output_dir)
        assert snapshot["input_generator"]["model_name"] == "input_generator-model"
        assert snapshot["generation"]["target_count"] == 0
        assert snapshot["input_generator"]["api_key"] == REDACTED_PLACEHOLDER
        assert snapshot["target_model"]["api_key"] == REDACTED_PLACEHOLDER

    def test_manifest_config_section_is_redacted_too(self, tmp_path: Path) -> None:
        """The manifest reuses the same dict, so it must be covered as well."""
        output_dir = tmp_path / "out"
        config = load_config(str(_write_config(tmp_path, output_dir, target_count=0)))

        result_dir = run(config)
        manifest_text = (result_dir / "manifest.json").read_text(encoding="utf-8")

        for marker in FORBIDDEN_MARKERS:
            assert marker not in manifest_text, (
                f"{marker} leaked into manifest.json's config section"
            )
        assert manifest_text.count(REDACTED_PLACEHOLDER) >= 2, (
            "both credentials should be masked in the manifest config section"
        )


class TestRedactSecretsRecursion:
    """Unit-level contract of ``_redact_secrets``."""

    def test_masks_nested_dicts_and_lists(self) -> None:
        source = {
            "input_generator": {"api_key": FAKE_INPUT_KEY, "model_name": "m"},
            "target_model": {"nested": {"deeper": {"api_key": FAKE_TARGET_KEY}}},
            "servers": [{"token": "tok-1"}, {"access_token": "tok-2"}],
            "keep": {"max_tokens": 128},
        }

        redacted = _redact_secrets(source)

        assert redacted["input_generator"]["api_key"] == REDACTED_PLACEHOLDER
        assert redacted["target_model"]["nested"]["deeper"]["api_key"] == REDACTED_PLACEHOLDER
        assert redacted["servers"][0]["token"] == REDACTED_PLACEHOLDER
        assert redacted["servers"][1]["access_token"] == REDACTED_PLACEHOLDER
        # A size field must not be collateral damage — see the test below.
        assert redacted["keep"]["max_tokens"] == 128

    def test_leaves_non_secret_values_untouched(self) -> None:
        """Sizes/durations must survive: 'token' matching must not be greedy."""
        source = {
            "generation": {"max_tokens": 128, "concurrency": 4},
            "input_generator": {"max_tokens": 2048, "first_token_timeout": 300.0},
            "output": {"max_tokens_with_image": 4096},
        }

        redacted = _redact_secrets(source)

        assert redacted == source

    def test_matching_is_case_insensitive_and_normalized(self) -> None:
        source = {
            "API_KEY": "a",
            "Access-Token": "b",
            "clientSecret": "c",
            "DB_PASSWORD": "d",
            "passwd": "e",
            "Authorization-Bearer": "f",
        }

        redacted = _redact_secrets(source)

        assert set(redacted.values()) == {REDACTED_PLACEHOLDER}

    def test_api_base_is_not_redacted(self) -> None:
        """An endpoint is a location, not a credential (documented tradeoff).

        The host is a documentation-reserved example, never a real endpoint
        (§15.1 — no internal addresses in tracked files).
        """
        source = {"input_generator": {"api_base": "http://api.example.invalid:8000/v1"}}

        redacted = _redact_secrets(source)

        assert redacted["input_generator"]["api_base"] == "http://api.example.invalid:8000/v1"

    def test_absent_credential_stays_none(self) -> None:
        """``None`` means "no key configured" and must not become a mask.

        Masking it would make an unconfigured run indistinguishable from a
        redacted one, inventing an empty-value sentinel (§2.2).
        """
        source = {"input_generator": {"api_key": None, "api_base": None}}

        redacted = _redact_secrets(source)

        assert redacted["input_generator"]["api_key"] is None
        assert redacted["input_generator"]["api_base"] is None

    def test_source_dict_is_not_mutated(self) -> None:
        """Redaction copies; the caller's config dict keeps its real values."""
        source = {"input_generator": {"api_key": FAKE_INPUT_KEY}}

        _redact_secrets(source)

        assert source["input_generator"]["api_key"] == FAKE_INPUT_KEY

    def test_every_declared_word_is_actually_matched(self) -> None:
        """Guard against a word being listed but never wired into matching."""
        source = {word: "value" for word in REDACTED_KEY_WORDS}

        redacted = _redact_secrets(source)

        assert set(redacted.values()) == {REDACTED_PLACEHOLDER}

    @pytest.mark.parametrize(
        "key",
        [
            "api_key",
            "API_KEY",
            "apiKey",
            "API-KEY",
            "apikey",
            "some_api_key",
            "service_key",
            "aws_access_key_id",
            "access_token",
            "accessToken",
            "hf_token",
            "client_secret",
            "clientSecret",
            "DB_PASSWORD",
            "passwd",
            "Authorization-Bearer",
            "user_auth",
        ],
    )
    def test_credential_key_spellings_are_matched(self, key: str) -> None:
        assert _is_secret_key(key), f"{key} should be treated as a credential"

    @pytest.mark.parametrize(
        "key",
        [
            "max_tokens",
            "max_tokens_with_image",
            "first_token_timeout",
            "inter_token_timeout",
            "api_base",
            "model_name",
            "temperature",
            "target_count",
            "concurrency",
            "languages",
            "seed",
            "resolved_seed",
            "connect_timeout",
            "max_retries",
            "retry_on_timeout",
            "backpressure_threshold",
            "backpressure_cooldown",
            "output",
            "directory",
            "embeddings_path",
            "tokenizer_path",
        ],
    )
    def test_non_secret_keys_are_never_masked(self, key: str) -> None:
        """Redaction must not corrupt the fields the snapshot exists to record."""
        assert not _is_secret_key(key), f"{key} must keep its real value"

    def test_every_not_secret_exemption_actually_exempts_something(self) -> None:
        """An exemption that exempts nothing is dead configuration."""
        assert all(not _is_secret_key(word) for word in NOT_SECRET_KEY_WORDS)
        # ...and each one would otherwise have matched, so the list earns its place.
        without_exemption = {
            "first_token_timeout": "token",
            "inter_token_timeout": "token",
        }
        for key, offending_word in without_exemption.items():
            assert offending_word in REDACTED_KEY_WORDS, (
                f"{key} is only exempt because {offending_word!r} is a credential word"
            )
