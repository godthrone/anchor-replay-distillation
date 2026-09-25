# test_acceptance_wiring.py — The acceptance phase as the pipeline runs it.
# Responsibility: freeze that ``pipeline.run`` writes results/coverage.json and
# coverage.md after generation and before the manifest, that the structure-only
# path costs zero model calls and announces itself in a WARNING, that the mock-
# embedded metric path reaches the artifacts with its space declared, that a bad
# target set is refused *before* the output directory exists, and that both
# shipped configs carry the same coverage fields (§7.1 mirror).
#
# No test dials a real endpoint: the metric-path test installs an in-process
# ``httpx.MockTransport``, the structure-only test installs a client that fails
# if it is ever constructed.

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest

from ard import pipeline
from ard.backends import embedding_client as ec
from ard.backends.coverage_wiring import CoverageWiringError
from ard.backends.embedding_client import EmbeddingRequestError
from ard.config import load_config
from ard.core.types import AnchorSpec, GeneratedAnchor, TurnSpec
from ard.domain.bank import append_anchor

_REAL_HTTPX_CLIENT = httpx.Client
_REPO_ROOT = Path(__file__).resolve().parents[1]
API_BASE = "https://mock.invalid/v1"
API_KEY = "sk-test-PIPELINE-DEADBEEF"
MODEL = "mock-embed"
DIMENSION = 3
_PLAN_SIZE = 3


# ── Rig ─────────────────────────────────────────────────────────────────────


def _plan() -> list[AnchorSpec]:
    """A small deterministic plan; distinct coordinates keep the noise band empty."""
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
    ontology_path: str = "unused.json",
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


_STRUCTURE_ONLY = '[coverage]\nenabled = true\ntarget_set_path = ""\n'


def _metric_block(target_set_path: Path) -> str:
    return "\n".join(
        [
            "[coverage]",
            "enabled = true",
            f'target_set_path = "{target_set_path}"',
            "",
            "[coverage.embedding]",
            f'api_base = "{API_BASE}"',
            f'api_key = "{API_KEY}"',
            f'model = "{MODEL}"',
            f"dimension = {DIMENSION}",
            "batch_size = 2",
            "normalize = true",
        ]
    )


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


class _MockEmbeddings:
    """Records requests and answers from a text → vector map (or a fixed status)."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vectors: Mapping[str, list[float]],
        *,
        status: int | None = None,
    ) -> None:
        self.vectors = dict(vectors)
        self.status = status
        self.requests: list[httpx.Request] = []
        monkeypatch.setattr(ec.httpx, "Client", self._fake_client)

    def _fake_client(self, *args: object, **kwargs: object) -> httpx.Client:
        kwargs.pop("timeout", None)
        return _REAL_HTTPX_CLIENT(
            transport=httpx.MockTransport(self._handler), timeout=httpx.Timeout(None)
        )

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status is not None:
            return httpx.Response(self.status, json={"error": "mock failure"})
        payload = json.loads(request.content)
        data = [
            {"index": index, "embedding": self.vectors[text]}
            for index, text in enumerate(payload["input"])
        ]
        return httpx.Response(200, json={"data": data, "model": MODEL})


# ── Structure-only path ─────────────────────────────────────────────────────


def test_structure_only_run_writes_the_readout_without_any_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    output_dir = tmp_path / "out"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, output_dir, _STRUCTURE_ONLY)
    _rig(monkeypatch)
    monkeypatch.setattr(
        ec.httpx,
        "Client",
        lambda *args, **kwargs: pytest.fail("the structure readout must cost no model call"),
    )

    with caplog.at_level("WARNING"):
        result_dir = pipeline.run(load_config(config_path), generate_specs=lambda cfg: _plan())

    report = json.loads((result_dir / "results" / "coverage.json").read_text(encoding="utf-8"))
    markdown = (result_dir / "results" / "coverage.md").read_text(encoding="utf-8")
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))

    assert report["metrics"] is None
    assert any("metric readout not measured" in warning for warning in report["warnings"])
    assert report["structure"]["plan_total"] == _PLAN_SIZE
    assert report["structure"]["within_rule"] is False
    assert "not computed" in markdown
    assert manifest["acceptance"]["metric_readout"] is False
    assert manifest["acceptance"]["q95"] is None
    assert any("target_set_path is unset" in record.getMessage() for record in caplog.records)


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

_V4_ONTOLOGY = str(_REPO_ROOT / "ontology" / "anchor_ontology.v4.json")


def test_smoke_run_is_marked_in_all_three_places_and_costs_no_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``--smoke`` plans 8/1826 through the real rule and says so three times over."""
    output_dir = tmp_path / "deliverable"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, output_dir, _STRUCTURE_ONLY, ontology_path=_V4_ONTOLOGY)
    _install_offline_generator(monkeypatch)
    monkeypatch.setattr(
        ec.httpx,
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
    assert report["metrics"] is None
    assert report["structure"]["plan_total"] == 8
    assert report["structure"]["expected_total"] == 1826
    assert report["structure"]["within_rule"] is False
    assert (result_dir / "results" / "coverage.md").is_file()

    # the non-smoke path is untouched: full plan, and no smoke config field
    from ard.config import ARDConfig

    assert "smoke" not in ARDConfig.model_fields
    assert len(pipeline.sample_specs(load_config(config_path))) == 1826


def test_two_smoke_runs_with_the_same_seed_are_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same seed, two smoke runs → the readouts are byte-for-byte the same."""
    _install_offline_generator(monkeypatch)
    reports: list[tuple[bytes, bytes]] = []
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


# ── Metric path ─────────────────────────────────────────────────────────────


def test_metric_run_writes_q95_and_declares_the_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "out"
    target_set_path = tmp_path / "targets.json"
    target_set_path.write_text(
        json.dumps(
            {
                "epsilon": 0.5,
                "targets": [{"text": "target-x"}],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.toml"
    _write_config(config_path, output_dir, _metric_block(target_set_path))
    _rig(monkeypatch)
    mock = _MockEmbeddings(
        monkeypatch,
        {
            "question plumbing-0": [1.0, 0.0, 0.0],
            "question plumbing-1": [0.0, 1.0, 0.0],
            "question plumbing-2": [0.0, 0.0, 1.0],
            "target-x": [1.0, 0.0, 0.0],
        },
    )

    result_dir = pipeline.run(load_config(config_path), generate_specs=lambda cfg: _plan())

    report_text = (result_dir / "results" / "coverage.json").read_text(encoding="utf-8")
    report = json.loads(report_text)
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = report["metrics"]

    assert mock.requests, "the metric path must actually call the embedding endpoint"
    assert metrics["space"]["n_anchor"] == _PLAN_SIZE
    assert metrics["space"]["n_target"] == 1
    assert metrics["space"]["embedder"] == {
        "model": MODEL,
        "dimension": DIMENSION,
        "normalize": True,
    }
    assert metrics["space"]["anchor_field"] == "messages[last].content (the final user-role turn)"
    assert metrics["space"]["target_field"] == "text"
    assert metrics["space"]["epsilon"] == pytest.approx(0.5)
    assert "header 'epsilon'" in metrics["space"]["epsilon_source"]
    assert metrics["quantiles"]["q95"] == pytest.approx(0.0)
    assert metrics["epsilon_band"]["extent_at_100"] == pytest.approx(1.0)
    assert metrics["noise"]["available"] is False
    assert "unavailable" in metrics["noise"]["reason"]
    assert manifest["acceptance"]["q95"] == pytest.approx(0.0)
    assert manifest["acceptance"]["metric_readout"] is True
    # §15: neither the key nor the endpoint reaches the artifacts.
    assert API_KEY not in report_text
    assert API_BASE not in report_text
    assert API_KEY not in (result_dir / "config.json").read_text(encoding="utf-8")


def test_embedding_failure_keeps_the_structure_readout_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed embedding call aborts the run, but never erases the zero-cost readout."""
    output_dir = tmp_path / "out"
    target_set_path = tmp_path / "targets.json"
    target_set_path.write_text(json.dumps(["target-x"]), encoding="utf-8")
    config_path = tmp_path / "config.toml"
    _write_config(config_path, output_dir, _metric_block(target_set_path) + "\nmax_retries = 0")
    _rig(monkeypatch)
    _MockEmbeddings(monkeypatch, {}, status=500)

    with pytest.raises(EmbeddingRequestError):
        pipeline.run(load_config(config_path), generate_specs=lambda cfg: _plan())

    report = json.loads((output_dir / "results" / "coverage.json").read_text(encoding="utf-8"))
    assert report["metrics"] is None
    assert report["structure"]["plan_total"] == _PLAN_SIZE
    assert not (output_dir / "manifest.json").exists()


# ── Boundary: target set is checked before any side effect ──────────────────


def test_missing_target_set_fails_before_the_output_directory_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "out"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, output_dir, _metric_block(tmp_path / "missing.jsonl"))
    _rig(monkeypatch)

    with pytest.raises(CoverageWiringError, match="not found"):
        pipeline.run(load_config(config_path), generate_specs=lambda cfg: _plan())

    assert not output_dir.exists(), "a refused target set must leave no half-built output directory"


def test_config_load_refuses_a_target_set_without_an_embedder(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        output_dir,
        '[coverage]\nenabled = true\ntarget_set_path = "targets.jsonl"\n\n'
        '[coverage.embedding]\nmodel = "mock-embed"\n',
    )
    with pytest.raises(ValueError, match="coverage.embedding"):
        load_config(config_path)


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
