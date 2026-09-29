# test_acceptance_wiring.py — The acceptance phase as the pipeline runs it.
# Responsibility: freeze that ``pipeline.run`` writes results/coverage.json and
# coverage.md after generation and before the manifest, that the structure
# readout costs zero model calls, that the phase writes nothing when disabled,
# that a leftover pre-removal ``[coverage]`` block still loads with a WARNING
# (§3.2), and that both shipped configs mirror each other's coverage fields and
# full field set (§7.1 mirror).
#
# No test dials a real endpoint: every test points ``httpx.Client`` at a stub
# that fails if it is ever constructed, and the generator is an in-process
# writer.

from __future__ import annotations

import json
import logging
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

import httpx
import pytest

from ard import pipeline
from ard.config import load_config
from ard.core.types import AnchorSpec, GeneratedAnchor, TurnSpec
from ard.domain.bank import append_anchor

_REPO_ROOT = Path(__file__).resolve().parents[1]
#: The production ontology.  The injected-plan seam still passes this to the
#: acceptance readout: the structure readout is defined against the run's own
#: sampling space, so it needs the real ontology even when the plan is injected.
_V4_ONTOLOGY = str(_REPO_ROOT / "ontology" / "anchor_ontology.v4.json")
_PLAN_SIZE = 3


# ── Rig ─────────────────────────────────────────────────────────────────────


def _plan() -> list[AnchorSpec]:
    """A small deterministic plan with distinct coordinates."""
    return [
        AnchorSpec(
            id=f"plumbing-{index}",
            anchor_meta={
                "modality": "text_only",
                "language": "English",
                "knowledge_domain": f"k{index}",
                "capability": f"c{index}",
            },
            turns=[TurnSpec(turn_index=0, role="user", is_final=True)],
        )
        for index in range(_PLAN_SIZE)
    ]


def _write_config(
    path: Path,
    output_dir: Path,
    coverage_block: str,
    *,
    overwrite: bool = False,
    ontology_path: str = _V4_ONTOLOGY,
) -> None:
    path.write_text(
        "\n".join(
            [
                "[input_generator]",
                'api_base = "http://127.0.0.1:1/v1"',
                'model_name = "input-model"',
                "",
                "[target_model]",
                'api_base = "http://127.0.0.1:1/v1"',
                'model_name = "target-model"',
                "",
                "[generation]",
                "seed = 7",
                "concurrency = 1",
                "",
                "[ontology]",
                f'path = "{ontology_path}"',
                "",
                "[output]",
                f'directory = "{output_dir}"',
                f"overwrite = {'true' if overwrite else 'false'}",
                "",
                coverage_block,
                "",
            ]
        ),
        encoding="utf-8",
    )


_STRUCTURE_ONLY = "[coverage]\nenabled = true\n"


def _install_offline_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the generator with an in-process writer — no model call, ever."""

    def spy(**kwargs: object) -> list[GeneratedAnchor]:
        specs = kwargs["specs"]
        output_path = kwargs["output_path"]
        stats = kwargs["stats"]
        assert isinstance(specs, list)
        assert isinstance(output_path, Path)
        written: list[GeneratedAnchor] = []
        for spec in specs:
            assert isinstance(spec, AnchorSpec)
            anchor = GeneratedAnchor(
                id=spec.id,
                messages=[{"role": "user", "content": f"question {spec.id}"}],
                target_answer=f"answer {spec.id}",
                target_model="target-model",
                input_generator_model="input-model",
                anchor_meta=spec.anchor_meta,
                reasoning=None,
            )
            append_anchor(anchor, output_path)
            written.append(anchor)
        stats.requested = len(specs)
        stats.written = len(written)
        stats.succeeded = len(written)
        return written

    monkeypatch.setattr(pipeline, "generate_text_anchors", spy)


def _rig(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the v4 sampler and the generator so ``run`` is offline and tiny."""
    monkeypatch.setattr(
        pipeline,
        "sample_specs",
        lambda config: pytest.fail("the plan double must replace the v4 sampler"),
    )
    _install_offline_generator(monkeypatch)


# ── Structure-only path ─────────────────────────────────────────────────────


def test_structure_only_run_writes_the_readout_without_any_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "out"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, output_dir, _STRUCTURE_ONLY)
    _rig(monkeypatch)
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda *args, **kwargs: pytest.fail("the structure readout must cost no model call"),
    )

    result_dir = pipeline.run(load_config(config_path), generate_specs=lambda cfg: _plan())

    report = json.loads((result_dir / "results" / "coverage.json").read_text(encoding="utf-8"))
    markdown = (result_dir / "results" / "coverage.md").read_text(encoding="utf-8")
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))

    assert report["report_schema"] == "ard-acceptance-4"
    assert "metrics" not in report
    assert report["warnings"] == []
    assert report["structure"]["plan_total"] == _PLAN_SIZE
    # The readout's expectations now come from the plan's own ``count`` (= its
    # length on the injected seam): three planned entries against a count of
    # three is exactly the rule's shape, so it reads green.  The old fixed
    # "= 1,826" expectation was removed.
    assert report["structure"]["within_rule"] is True
    assert "## Structure readout (zero model calls)" in markdown
    assert manifest["acceptance"]["coverage_json"] == "results/coverage.json"
    assert manifest["acceptance"]["coverage_md"] == "results/coverage.md"


def test_disabled_acceptance_phase_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "out"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, output_dir, "[coverage]\nenabled = false\n")
    _rig(monkeypatch)

    result_dir = pipeline.run(load_config(config_path), generate_specs=lambda cfg: _plan())

    assert not (result_dir / "results").exists()
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "acceptance" not in manifest


# ── Smoke run (--smoke): same rule, reduced scale, self-proving artifact ────


def test_smoke_run_is_marked_in_all_three_places_and_costs_no_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``--smoke`` plans 8/1826 through the real rule and says so three times over."""
    output_dir = tmp_path / "deliverable"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, output_dir, _STRUCTURE_ONLY, ontology_path=_V4_ONTOLOGY)
    _install_offline_generator(monkeypatch)
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda *args, **kwargs: pytest.fail("the structure readout must cost no model call"),
    )

    with caplog.at_level("WARNING"):
        result_dir = pipeline.run(load_config(config_path), smoke=True)

    # 1. the artifact's own run directory is tagged
    assert result_dir.name == "deliverable_smoke"
    # 2. the log carries the WARNING, with the exact scale
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("SMOKE RUN" in message for message in warnings)
    assert any("8 of 1826" in message for message in warnings)
    # 3. manifest.json declares the smoke run and its counts
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["smoke"] is True
    assert manifest["smoke_plan"]["run_name"] == "deliverable_smoke"
    assert manifest["smoke_plan"]["planned_anchors"] == 8
    assert manifest["smoke_plan"]["full_expected_anchors"] == 1826
    assert manifest["smoke_plan"]["text_blocks"] == 4
    assert manifest["smoke_plan"]["image_blocks"] == 4
    assert manifest["total_anchors"] == 8

    # the acceptance readout still exists, and it is honest about the short plan
    report = json.loads((result_dir / "results" / "coverage.json").read_text(encoding="utf-8"))
    assert "metrics" not in report
    assert report["structure"]["plan_total"] == 8
    # A smoke plan is measured against its OWN count (8), not against one full
    # cycle: the old expectation of 1,826 turned a normal smoke artifact into a
    # reported violation.
    assert report["structure"]["expected_total"] == 8
    assert report["structure"]["within_rule"] is True
    assert (result_dir / "results" / "coverage.md").is_file()

    # the non-smoke path is untouched: full plan, and no smoke config field
    from ard.config import ARDConfig

    assert "smoke" not in ARDConfig.model_fields
    assert len(pipeline.sample_specs(load_config(config_path))) == 1826


#: The two report artefacts of one run, in ``coverage.json`` / ``coverage.md`` order.
ReportBytes: TypeAlias = tuple[bytes, bytes]


def test_two_smoke_runs_with_the_same_seed_are_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same seed, two smoke runs → the readouts are byte-for-byte the same."""
    _install_offline_generator(monkeypatch)
    reports: list[ReportBytes] = []
    for index in (1, 2):
        output_dir = tmp_path / f"deliverable{index}"
        config_path = tmp_path / f"config{index}.toml"
        _write_config(config_path, output_dir, _STRUCTURE_ONLY, ontology_path=_V4_ONTOLOGY)
        result_dir = pipeline.run(load_config(config_path), smoke=True)
        reports.append(
            (
                (result_dir / "results" / "coverage.json").read_bytes(),
                (result_dir / "results" / "coverage.md").read_bytes(),
            )
        )

    assert reports[0] == reports[1]
    report = json.loads(reports[0][0])
    assert report["structure"]["plan_total"] == 8


# ── Removed-feature compatibility (§3.2) ────────────────────────────────────


def test_a_pre_removal_coverage_block_still_loads_with_one_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An old override / snapshot must not refuse the run — it warns instead.

    ``target_set_path`` and ``[coverage.embedding]`` are gone, but ``ARDConfig``
    forbids extras (§2.3) and a run's own ``config.toml`` snapshot invites reuse
    as ``--config``, so exactly those two leftovers are dropped loudly (§3.2).
    """
    legacy = tmp_path / "legacy.toml"
    _write_config(
        legacy,
        tmp_path / "out-legacy",
        "[coverage]\n"
        "enabled = true\n"
        'target_set_path = "targets.jsonl"\n'
        "\n"
        "[coverage.embedding]\n"
        'model = "old-embed"\n'
        "dimension = 8\n"
        "normalize = true\n",
    )
    with caplog.at_level(logging.WARNING):
        config = load_config(legacy)

    assert config.coverage.enabled is True
    dropped = [
        record.getMessage()
        for record in caplog.records
        if "ignoring [coverage] key" in record.getMessage()
    ]
    assert len(dropped) == 1, dropped
    assert "embedding" in dropped[0]
    assert "target_set_path" in dropped[0]


def test_a_coverage_section_without_leftovers_warns_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Only the two removed keys are announced; a clean table says nothing."""
    clean = tmp_path / "clean.toml"
    _write_config(clean, tmp_path / "out-clean", "[coverage]\nenabled = false\n")

    with caplog.at_level(logging.WARNING):
        config = load_config(clean)

    assert config.coverage.enabled is False
    assert not [r for r in caplog.records if "coverage" in r.getMessage()], caplog.records


def test_a_misspelled_coverage_key_is_refused_by_extra_forbid(tmp_path: Path) -> None:
    """The exemption covers the removed keys only — a typo still fails the boundary.

    Dropping every unknown ``[coverage]`` key would accept ``enabeld = true`` as an
    "embedding ruler leftover" and run with a setting the operator never wrote;
    ``extra="forbid"`` must name it instead (§2.3 边界校验即防呆).
    """
    typo = tmp_path / "typo.toml"
    _write_config(typo, tmp_path / "out-typo", "[coverage]\nenabeld = true\n")

    with pytest.raises(ValueError, match="enabeld"):
        load_config(typo)


# ── §7.1 field mirror ───────────────────────────────────────────────────────


def _leaf_paths(document: Mapping, prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key, value in document.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            paths.extend(_leaf_paths(value, prefix=f"{path}."))
        else:
            paths.append(path)
    return paths


def test_both_configs_mirror_the_coverage_fields() -> None:
    base = tomllib.loads((_REPO_ROOT / "configs" / "config.toml").read_text(encoding="utf-8"))
    sample = tomllib.loads(
        (_REPO_ROOT / "configs" / "config.override.sample.toml").read_text(encoding="utf-8")
    )
    sample_text = (_REPO_ROOT / "configs" / "config.override.sample.toml").read_text(
        encoding="utf-8"
    )

    base_coverage = _leaf_paths(base["coverage"], prefix="coverage.")
    assert base_coverage, "the base config must define the coverage section"
    for path in base_coverage:
        leaf = path.rsplit(".", 1)[-1]
        assert re.search(rf"^\s*#?\s*{re.escape(leaf)}\s*=", sample_text, re.M), (
            f"{path} is missing from config.override.sample.toml"
        )

    sample_paths = set(_leaf_paths(sample))
    assert sample_paths <= set(_leaf_paths(base)), (
        "the sample config must not introduce fields the base config lacks"
    )


# The coverage-only check above missed `[generation] count` when v5 turned N
# into a configuration field: the sample template still said N was not
# configurable, so a user following the README's copy step never saw the field
# and the "N is user-settable" path was broken end to end. Mirror *every*
# section's field names — commented-out template lines included — so a field
# cannot land in one config and be forgotten in the other.

_BOOLEAN_LITERALS = frozenset({"true", "false"})


def _declared_field_names(text: str) -> dict[str, set[str]]:
    """Field names per ``[section]``, counting commented-out template lines."""
    fields: dict[str, set[str]] = {}
    section: str | None = None
    for line in text.splitlines():
        header = re.match(r"\s*\[([^\[\]]+)\]", line)
        if header:
            section = header.group(1)
            fields.setdefault(section, set())
            continue
        field = re.match(r"\s*#?\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        if field and section is not None and field.group(1).lower() not in _BOOLEAN_LITERALS:
            fields[section].add(field.group(1))
    return fields


def test_both_configs_mirror_every_field_name() -> None:
    """config.toml and its override template declare the same field set (§1.4)."""
    base = _declared_field_names(
        (_REPO_ROOT / "configs" / "config.toml").read_text(encoding="utf-8")
    )
    sample = _declared_field_names(
        (_REPO_ROOT / "configs" / "config.override.sample.toml").read_text(encoding="utf-8")
    )

    assert set(base) == set(sample)
    for section in sorted(base):
        missing = sorted(base[section] - sample[section])
        assert not missing, f"[{section}] fields missing from the sample template: {missing}"
        extra = sorted(sample[section] - base[section])
        assert not extra, f"[{section}] fields the sample template adds: {extra}"
