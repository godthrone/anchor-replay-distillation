# tests/test_config_count.py — `[generation] count` boundary contract tests.
# Responsibility: lock the anchor-count field's semantics — unset (None) means
# "one full round", an explicit integer must be >= 1, and a non-positive value
# is refused at config load with an actionable message (§1.4 single source of
# truth, §2.2 None is the only unset, §2.3 load-time boundary validation).
#
# N has NO upper bound: the plan rolls on across rounds and uniqueness comes
# from the anchor id, not from a rejection threshold. There is therefore no
# "too large" case in this file.


from pathlib import Path

import pytest

from ard.config import GenerationConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "config.toml"


def _write_config(tmp_path: Path, generation_body: str) -> Path:
    """Write a minimal config whose `[generation]` table is *generation_body*."""
    path = tmp_path / "config.toml"
    path.write_text(f"[generation]\n{generation_body}\n", encoding="utf-8")
    return path


# ── The default: unset count == one full round ──────────────────────────────


def test_shipped_config_leaves_count_unset() -> None:
    """The shipped base config does not pin N, so a round is the default."""
    config = load_config(BASE_CONFIG_PATH)
    assert config.generation.count is None


def test_generation_config_default_count_is_none() -> None:
    """The typed default is None, never 0 or -1 (§2.2)."""
    assert GenerationConfig().count is None


@pytest.mark.parametrize("text", ["count = 1", "count = 1826", "count = 1000000"])
def test_positive_counts_are_accepted(tmp_path: Path, text: str) -> None:
    """Any integer >= 1 is a valid N — there is no upper bound to trip over."""
    expected = int(text.split("=")[1])
    config = load_config(_write_config(tmp_path, text))
    assert config.generation.count == expected


# ── Boundary rejections (§2.3) ──────────────────────────────────────────────


@pytest.mark.parametrize("value", [0, -1, -100])
def test_non_positive_counts_are_refused_at_load(tmp_path: Path, value: int) -> None:
    """0 and negatives are refused at load, not read as 'unset' (§2.2)."""
    with pytest.raises(ValueError) as excinfo:
        load_config(_write_config(tmp_path, f"count = {value}"))
    message = str(excinfo.value)
    assert "generation.count" in message
    assert f"got {value}" in message
    assert "integer >= 1" in message


@pytest.mark.parametrize("text", ['count = "many"', "count = true", "count = 0.5"])
def test_non_integer_count_is_refused(tmp_path: Path, text: str) -> None:
    """A non-integer N is a boundary error, not coerced into a number.

    ``true`` is in the list because pydantic's lax mode would otherwise turn it
    into ``1`` and silently schedule a one-anchor run.
    """
    with pytest.raises(ValueError) as excinfo:
        load_config(_write_config(tmp_path, text))
    assert "generation.count" in str(excinfo.value)


def test_unknown_generation_key_is_still_forbidden(tmp_path: Path) -> None:
    """extra_forbid is retained on the generation section."""
    with pytest.raises(ValueError) as excinfo:
        load_config(_write_config(tmp_path, "count = 1\nanchor_count = 5"))
    assert "anchor_count" in str(excinfo.value)
