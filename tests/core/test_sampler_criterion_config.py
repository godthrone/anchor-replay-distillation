"""The greedy ``criterion`` as a production *config* option.

`core/cloud.py` has supported a selectable greedy rule since the
``selectable-criterion`` change, but it was only reachable from code:
``GenerationConfig`` had no such field, so ``[generation]`` in
``configs/config.toml`` could not ask for it.  These tests pin the config
surface that closes that gap:

* ``criterion`` has a default of ``"max"`` — the historical rule — so an unset
  config selects exactly what every previous release selected;
* ``"sum"`` reaches the FPS call through the whole chain
  ``GenerationConfig → AnchorGenerationConfig → sample_anchors → _sample_farthest``;
* an unknown value is refused **at config load** with the legal set in the
  message (never silently defaulted);
* the config file that ships with the repo loads with its own values.

The byte-identity evidence for the default path (SHA of the selection before and
after the change) lives in the work package's ``evidence/`` directory; the last
test here is the cheap in-repo guard that the two ways of asking for the
historical rule cannot drift apart.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from ard.config import GenerationConfig, load_config
from ard.core.cloud import CRITERION_MAX, CRITERION_SUM, FPS_CRITERIA
from ard.core.sampler import sample_anchors
from ard.core.types import AnchorGenerationConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_CONFIG = REPO_ROOT / "configs" / "config.toml"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_ontology() -> dict[str, Any]:
    """A small ontology with two domains so Layer 1 and Layer 2 both run.

    The system-prompt sections are omitted on purpose: an ontology without them
    degrades to the single ``("none", "none")`` pair, which keeps this fixture
    independent of the system-prompt vocabulary.
    """
    return {
        "languages": ["English", "简体中文"],
        "knowledge_domains": {
            "alpha": {"sub_categories": {"a1": {"leaf_topics": ["t1"]}}},
            "beta": {"sub_categories": {"b1": {"leaf_topics": ["t2"]}}},
        },
        "capabilities": {"reasoning": ["deduction", "induction"]},
        "conversation_types": {"single_turn": ["single_turn"]},
    }


def _make_embeddings(path: Path, dim: int = 8, seed: int = 7) -> Path:
    """Write an embeddings file the validator accepts for :func:`_make_ontology`."""
    import numpy as np

    rng = np.random.default_rng(seed)

    def unit() -> list[float]:
        vector = rng.normal(size=dim)
        return (vector / np.linalg.norm(vector)).tolist()

    data = {
        "model": "unit-test-embedder",
        "embedding_dimension": dim,
        "ontology_sha256": "0" * 64,
        "generated_at": "2026-09-18T00:00:00+00:00",
        "items": {
            "knowledge_domains": {"alpha": unit(), "beta": unit()},
            "capabilities": {"deduction": unit(), "induction": unit()},
            "languages": {"English": unit(), "简体中文": unit()},
            "conversation_types": {"single_turn": unit()},
            # This ontology declares no system-prompt sections, so the sampler
            # degrades to the single (``none``, ``none``) pair; ``none`` still
            # needs a vector because it is a real composition slot.
            "system_prompt": {"none": unit()},
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _dump_toml(data: dict[str, Any]) -> str:
    """Minimal nested TOML writer for str/int/float/bool/list leaves."""
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


# ---------------------------------------------------------------------------
# 1. the config model's contract
# ---------------------------------------------------------------------------


def test_default_criterion_is_the_historical_max_rule() -> None:
    """An unset field must mean "what every previous release did"."""
    assert GenerationConfig().criterion == CRITERION_MAX


def test_sum_is_accepted_as_the_alternative_rule() -> None:
    assert GenerationConfig(criterion=CRITERION_SUM).criterion == CRITERION_SUM


@pytest.mark.parametrize("bad", ["Mean", "MAX", "", " max", "max ", "total"])
def test_unknown_criterion_is_refused_with_the_legal_set(bad: str) -> None:
    """A typo must fail loudly and name the accepted values (§2.1 契约即防呆)."""
    with pytest.raises(ValueError) as excinfo:
        GenerationConfig(criterion=bad)
    message = str(excinfo.value)
    assert all(value in message for value in FPS_CRITERIA), message
    assert "criterion" in message


# ---------------------------------------------------------------------------
# 2. the TOML boundary
# ---------------------------------------------------------------------------


def test_shipped_config_loads_and_carries_max(tmp_path: Path) -> None:
    """``configs/config.toml`` must load and must ship the historical default."""
    config = load_config(SHIPPED_CONFIG)
    assert config.generation.criterion == CRITERION_MAX


def test_shipped_config_stays_valid_when_criterion_is_set_to_sum() -> None:
    """The base config carries the key, so an override can flip it to ``sum``.

    ``model_validate`` on the raw parsed file is enough here: it proves the
    shipped file is a valid base (the override is a deep merge of the same
    shape, and the field is a plain string).
    """
    import tomllib

    from ard.config import ARDConfig

    base = tomllib.loads(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    assert "criterion" in base["generation"], (
        "configs/config.toml must declare criterion so an override can set it"
    )
    base["generation"]["criterion"] = CRITERION_SUM
    assert ARDConfig.model_validate(base).generation.criterion == CRITERION_SUM


def test_override_file_can_flip_the_criterion(tmp_path: Path) -> None:
    """A real ``load_config(base, override)`` round-trip reaches ``sum``."""
    import tomllib

    base = tomllib.loads(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    base["generation"]["target_count"] = 3
    base_path = tmp_path / "config.toml"
    base_path.write_text(_dump_toml(base), encoding="utf-8")

    override_path = tmp_path / "override.toml"
    override_path.write_text('[generation]\ncriterion = "sum"\n', encoding="utf-8")

    config = load_config(base_path, override_path)
    assert config.generation.criterion == CRITERION_SUM


def test_override_file_with_a_typo_is_refused_at_load(tmp_path: Path) -> None:
    """The typo must die at the config boundary, not inside the sampler."""
    import tomllib

    base = tomllib.loads(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    base_path = tmp_path / "config.toml"
    base_path.write_text(_dump_toml(base), encoding="utf-8")
    override_path = tmp_path / "override.toml"
    override_path.write_text('[generation]\ncriterion = "maximum"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="criterion"):
        load_config(base_path, override_path)


# ---------------------------------------------------------------------------
# 3. the chain: GenerationConfig → AnchorGenerationConfig → fps
# ---------------------------------------------------------------------------


def test_criterion_reaches_the_fps_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``sample_anchors`` must forward ``config.criterion`` to the FPS layer.

    ``fps`` is spied on rather than reimplemented: the assertion is about the
    *argument that arrives*, which is exactly the link the config change adds.
    The real selection still runs, so the call must also succeed.
    """
    import ard.core._fps as fps_module

    embeddings_path = _make_embeddings(tmp_path / "embeddings.json")
    seen: list[str] = []
    real_fps = fps_module.fps

    def spy(cloud: Any, n: int, **kwargs: Any) -> Any:
        # Layer 1 (_farthest_domain_order) deliberately calls fps() without a
        # criterion; only Layer 2 and the remainder fill carry the configured
        # rule, which is what this test is about.
        if "criterion" in kwargs:
            seen.append(kwargs["criterion"])
        return real_fps(cloud, n, **kwargs)

    monkeypatch.setattr(fps_module, "fps", spy)

    config = AnchorGenerationConfig(
        target_count=4,
        seed=11,
        embeddings_path=str(embeddings_path),
        criterion=CRITERION_SUM,
    )
    specs = sample_anchors(_make_ontology(), config, random.Random(11))

    assert specs, "the spy must not change the result of the real selection"
    assert seen, "the FPS layer was never reached"
    assert set(seen) == {CRITERION_SUM}, (
        f"every FPS call must receive the configured rule, saw {sorted(set(seen))}"
    )


def test_unset_criterion_resolves_to_the_historical_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``criterion=None`` (the dataclass default) must resolve to ``max``."""
    import ard.core._fps as fps_module

    embeddings_path = _make_embeddings(tmp_path / "embeddings.json")
    seen: list[str] = []
    real_fps = fps_module.fps

    def spy(cloud: Any, n: int, **kwargs: Any) -> Any:
        # Layer 1 (_farthest_domain_order) deliberately calls fps() without a
        # criterion; only Layer 2 and the remainder fill carry the configured
        # rule, which is what this test is about.
        if "criterion" in kwargs:
            seen.append(kwargs["criterion"])
        return real_fps(cloud, n, **kwargs)

    monkeypatch.setattr(fps_module, "fps", spy)

    config = AnchorGenerationConfig(target_count=4, seed=11, embeddings_path=str(embeddings_path))
    assert config.criterion is None
    sample_anchors(_make_ontology(), config, random.Random(11))
    assert set(seen) == {CRITERION_MAX}


def test_explicit_sample_anchors_keyword_overrides_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function-level keyword still wins, so callers keep the old API."""
    import ard.core._fps as fps_module

    embeddings_path = _make_embeddings(tmp_path / "embeddings.json")
    seen: list[str] = []
    real_fps = fps_module.fps

    def spy(cloud: Any, n: int, **kwargs: Any) -> Any:
        # Layer 1 (_farthest_domain_order) deliberately calls fps() without a
        # criterion; only Layer 2 and the remainder fill carry the configured
        # rule, which is what this test is about.
        if "criterion" in kwargs:
            seen.append(kwargs["criterion"])
        return real_fps(cloud, n, **kwargs)

    monkeypatch.setattr(fps_module, "fps", spy)

    config = AnchorGenerationConfig(
        target_count=4,
        seed=11,
        embeddings_path=str(embeddings_path),
        criterion=CRITERION_SUM,
    )
    sample_anchors(_make_ontology(), config, random.Random(11), criterion=CRITERION_MAX)
    assert set(seen) == {CRITERION_MAX}


# ---------------------------------------------------------------------------
# 4. default-path equivalence
# ---------------------------------------------------------------------------


def test_default_and_explicit_max_select_the_same_anchors(tmp_path: Path) -> None:
    """The two ways of asking for "historical" must not drift apart.

    This is the in-repo guard; the byte-level before/after evidence for the
    change itself is in the work package's ``evidence/prod-fingerprint.*.json``.
    """
    embeddings_path = _make_embeddings(tmp_path / "embeddings.json")
    ontology = _make_ontology()
    kwargs = dict(target_count=5, seed=99, embeddings_path=str(embeddings_path))

    implicit = sample_anchors(ontology, AnchorGenerationConfig(**kwargs), random.Random(99))
    explicit = sample_anchors(
        ontology,
        AnchorGenerationConfig(**kwargs, criterion=CRITERION_MAX),
        random.Random(99),
    )

    def fingerprint(specs: list[Any]) -> str:
        return json.dumps(
            [
                {
                    "id": spec.id,
                    "meta": spec.anchor_meta,
                    "turns": [(t.turn_index, t.role, t.is_final) for t in spec.turns],
                }
                for spec in specs
            ],
            sort_keys=True,
            ensure_ascii=False,
        )

    assert fingerprint(implicit) == fingerprint(explicit)
