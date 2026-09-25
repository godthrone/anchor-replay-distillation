"""The greedy ``criterion`` of the legacy FPS sampler (WP-S2a note).

``core/cloud.py`` has supported a selectable greedy rule since the
``selectable-criterion`` change.  The *config* half of that change was removed in
WP-S2a together with the rest of the v3/FPS configuration surface (nothing in the
new v4 rule reads it), so what remains here is the part that still describes live
code: ``criterion`` reaches the FPS call through the chain
``AnchorGenerationConfig → sample_anchors → _sample_farthest``, and the two ways
of asking for the historical rule still cannot drift apart.

The byte-identity evidence for the default path (SHA of the selection before and
after the change) lives in the work package's ``evidence/`` directory.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, TypedDict

import pytest

from ard.core.cloud import CRITERION_MAX, CRITERION_SUM
from ard.core.sampler import sample_anchors
from ard.core.types import AnchorGenerationConfig


class _DefaultPathKwargs(TypedDict):
    """Fields shared by the two ``AnchorGenerationConfig``s compared in §4.

    The point of that test is that *only* ``criterion`` differs; spelling the
    shared fields once (and unpacking them into both constructions) keeps the
    two calls from drifting apart.
    """

    target_count: int
    seed: int
    embeddings_path: str


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
    # A TypedDict (not a bare ``dict``) so ``**kwargs`` still type-checks against
    # ``AnchorGenerationConfig``'s fields: an untyped dict literal widens every
    # value to ``object`` and mypy refuses the call outright.
    kwargs: _DefaultPathKwargs = {
        "target_count": 5,
        "seed": 99,
        "embeddings_path": str(embeddings_path),
    }

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
