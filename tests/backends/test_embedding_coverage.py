# test_embedding_coverage.py — Tests for the acceptance *wiring* (facility layer).
# Responsibility: freeze the target-set boundary (missing / empty / malformed /
# count-mismatch / dimension-mismatch all refuse loudly), the mock-embedded
# metric path end to end, the noise-band availability rule, and the §15 rule
# that no key or endpoint reaches the acceptance report.
#
# No test dials a real endpoint: every request goes through an in-process
# ``httpx.MockTransport`` installed as the embedding client's ``httpx.Client``,
# and the only base URL used is an unresolvable mock host.

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest

from ard.backends import coverage_wiring
from ard.backends import embedding_client as ec
from ard.backends.embedding_client import EmbeddingRequestError
from ard.config import CoverageEmbedding
from ard.core import acceptance

_REAL_HTTPX_CLIENT = httpx.Client
API_BASE = "https://mock.invalid/v1"
MODEL = "mock-embed"
DIMENSION = 3
API_KEY = "sk-test-COVERAGE-CAFEBABE"

ANCHOR_VECTORS = {
    "anchor-0": [1.0, 0.0, 0.0],
    "anchor-1": [0.0, 1.0, 0.0],
    "anchor-2": [0.0, 0.0, 1.0],
}


class MockEmbeddings:
    """A recording ``httpx.MockTransport`` installed as ``ec.httpx.Client``."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vectors: dict[str, list[float]],
        *,
        status: int | None = None,
    ) -> None:
        self.vectors = vectors
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
        data = []
        for index, text in enumerate(payload["input"]):
            if text not in self.vectors:
                raise AssertionError(f"the test mock has no vector for {text!r}")
            data.append({"index": index, "embedding": self.vectors[text]})
        return httpx.Response(200, json={"data": data, "model": MODEL})


def _embedding(**overrides: object) -> CoverageEmbedding:
    values: dict[str, object] = {
        "api_base": API_BASE,
        "api_key": API_KEY,
        "model": MODEL,
        "dimension": DIMENSION,
        "batch_size": 2,
        "normalize": True,
    }
    values.update(overrides)
    return CoverageEmbedding(**values)


def _record(anchor_id: str, text: str, meta: Mapping[str, str]) -> dict:
    return {
        "id": anchor_id,
        "messages": [{"role": "user", "content": text}],
        "anchor_meta": dict(meta),
    }


def _write(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# ── Target-set boundary ─────────────────────────────────────────────────────


def test_load_target_set_accepts_a_json_array_of_strings(tmp_path: Path) -> None:
    path = _write(tmp_path / "targets.json", ["first", "second"])
    target_set = coverage_wiring.load_target_set(path, expected_dimension=DIMENSION)
    assert target_set.texts == ("first", "second")
    assert target_set.declared_count is None
    assert target_set.path == str(path)


def test_load_target_set_accepts_jsonl_objects(tmp_path: Path) -> None:
    path = tmp_path / "targets.jsonl"
    path.write_text(
        json.dumps({"text": "first", "id": "t1"}) + "\n\n" + json.dumps({"text": "second"}) + "\n",
        encoding="utf-8",
    )
    target_set = coverage_wiring.load_target_set(path, expected_dimension=None)
    assert target_set.texts == ("first", "second")


def test_load_target_set_reads_the_object_header(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "targets.json",
        {
            "expected_count": 1,
            "dimension": DIMENSION,
            "epsilon": 0.25,
            "targets": [{"text": "only"}],
        },
    )
    target_set = coverage_wiring.load_target_set(path, expected_dimension=DIMENSION)
    assert target_set.texts == ("only",)
    assert target_set.declared_count == 1
    assert target_set.declared_dimension == DIMENSION
    assert target_set.declared_epsilon == pytest.approx(0.25)


def test_load_target_set_reports_a_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(coverage_wiring.CoverageWiringError, match="not found"):
        coverage_wiring.load_target_set(missing, expected_dimension=DIMENSION)


@pytest.mark.parametrize("payload", ["", "   \n", "[]", '{"targets": []}'])
def test_load_target_set_reports_an_empty_file(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "targets.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(coverage_wiring.CoverageWiringError, match="empty|carr"):
        coverage_wiring.load_target_set(path, expected_dimension=None)


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ("{not json at all", "not valid JSON"),
        ('{"no_targets": 1}', "must carry a 'targets' array"),
        ("[1, 2]", "must be a string or an object"),
        ('[{"id": "t1"}]', "non-empty 'text' string"),
        ('[{"text": "   "}]', "non-empty 'text' string"),
        ('{"expected_count": 5, "targets": ["a"]}', "expected_count=5"),
        (
            '{"dimension": 1536, "targets": ["a"]}',
            "declares dimension=1536 but coverage.embedding.dimension is 3",
        ),
        ('{"epsilon": -1, "targets": ["a"]}', "finite and >= 0"),
    ],
)
def test_load_target_set_refuses_bad_documents(tmp_path: Path, payload: str, fragment: str) -> None:
    path = tmp_path / "targets.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(coverage_wiring.CoverageWiringError, match=fragment):
        coverage_wiring.load_target_set(path, expected_dimension=DIMENSION)


# ── Metric path (mock endpoint) ─────────────────────────────────────────────


def test_metric_readout_measures_mock_vectors_and_declares_the_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    MockEmbeddings(
        monkeypatch,
        {
            **ANCHOR_VECTORS,
            "target-x": [1.0, 0.0, 0.0],
            "target-y": [0.0, 1.0, 0.0],
        },
    )
    records = [
        _record("a0", "anchor-0", {"knowledge_domain": "k1"}),
        _record("a1", "anchor-1", {"knowledge_domain": "k2"}),
        _record("a2", "anchor-2", {"knowledge_domain": "k3"}),
    ]
    target_set = coverage_wiring.load_target_set(
        _write(
            tmp_path / "targets.json",
            {"epsilon": 0.5, "targets": [{"text": "target-x"}, {"text": "target-y"}]},
        ),
        expected_dimension=DIMENSION,
    )

    readout = coverage_wiring.build_metric_readout(
        embedding=_embedding(),
        records=records,
        anchors_source=str(tmp_path / "anchor_bank.jsonl"),
        target_set=target_set,
    )

    # "target-x" is identical to "anchor-0" and "target-y" to "anchor-1": every
    # nearest-anchor distance is 0, so every quantile and Extent is 0 / 1.
    assert readout.quantiles.q95 == pytest.approx(0.0)
    assert readout.quantiles.r_max == pytest.approx(0.0)
    assert readout.extent == pytest.approx(1.0)
    assert readout.epsilon_band.extent_at_095 == pytest.approx(1.0)
    space = readout.space
    assert space.n_anchor == 3
    assert space.n_target == 2
    assert space.embedder.model == MODEL
    assert space.embedder.dimension == DIMENSION
    assert space.embedder.normalize is True
    assert space.anchor_field == acceptance.ANCHOR_TEXT_FIELD
    assert space.target_field == acceptance.TARGET_TEXT_FIELD
    assert space.epsilon == pytest.approx(0.5)
    assert "header 'epsilon'" in space.epsilon_source
    assert readout.noise.available is False
    assert readout.noise.reason == acceptance.NOISE_UNAVAILABLE_REASON


def test_metric_readout_calibrates_epsilon_from_the_target_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a header epsilon, the target set's own scale is used and named."""
    MockEmbeddings(
        monkeypatch,
        {
            **ANCHOR_VECTORS,
            "target-x": [1.0, 0.0, 0.0],
            "target-y": [0.0, 1.0, 0.0],
        },
    )
    records = [_record("a0", "anchor-0", {"knowledge_domain": "k1"})]
    target_set = coverage_wiring.load_target_set(
        _write(tmp_path / "targets.json", ["target-x", "target-y"]),
        expected_dimension=DIMENSION,
    )
    readout = coverage_wiring.build_metric_readout(
        embedding=_embedding(),
        records=records,
        anchors_source="bank.jsonl",
        target_set=target_set,
    )
    # two orthogonal targets ⇒ nearest-neighbour distance 1 for both
    assert readout.space.epsilon == pytest.approx(1.0)
    assert "nearest-neighbour" in readout.space.epsilon_source


def test_metric_readout_measures_the_noise_band_when_a_cell_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    MockEmbeddings(
        monkeypatch,
        {
            **ANCHOR_VECTORS,
            "target-x": [1.0, 0.0, 0.0],
        },
    )
    repeated = {"knowledge_domain": "k1"}
    records = [
        _record("a0", "anchor-0", repeated),
        _record("a1", "anchor-1", repeated),
        _record("a2", "anchor-0", repeated),
    ]
    target_set = coverage_wiring.load_target_set(
        _write(tmp_path / "targets.json", {"epsilon": 0.5, "targets": ["target-x"]}),
        expected_dimension=DIMENSION,
    )
    readout = coverage_wiring.build_metric_readout(
        embedding=_embedding(),
        records=records,
        anchors_source="bank.jsonl",
        target_set=target_set,
    )
    assert readout.noise.available is True
    assert readout.noise.reason is None
    assert readout.noise.n_repeat_groups == 1
    assert readout.noise.n_pairs == 3
    assert readout.noise.band is not None
    assert readout.noise.band.lower == pytest.approx(1.0)
    assert readout.noise.band.upper == pytest.approx(1.0)


def test_metric_readout_propagates_an_embedding_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed embedding call bubbles up with the client's own exception type."""
    MockEmbeddings(monkeypatch, {}, status=500)
    target_set = coverage_wiring.load_target_set(
        _write(tmp_path / "targets.json", ["target-x"]),
        expected_dimension=DIMENSION,
    )
    with pytest.raises(EmbeddingRequestError):
        coverage_wiring.build_metric_readout(
            embedding=_embedding(max_retries=0),
            records=[_record("a0", "anchor-0", {"knowledge_domain": "k1"})],
            anchors_source="bank.jsonl",
            target_set=target_set,
        )


def test_metric_readout_refuses_a_record_without_a_coordinate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    MockEmbeddings(monkeypatch, {"anchor-0": [1.0, 0.0, 0.0], "target-x": [1.0, 0.0, 0.0]})
    target_set = coverage_wiring.load_target_set(
        _write(tmp_path / "targets.json", {"epsilon": 0.5, "targets": ["target-x"]}),
        expected_dimension=DIMENSION,
    )
    with pytest.raises(acceptance.AcceptanceError, match="anchor_meta"):
        coverage_wiring.build_metric_readout(
            embedding=_embedding(),
            records=[{"id": "a0", "messages": [{"role": "user", "content": "anchor-0"}]}],
            anchors_source="bank.jsonl",
            target_set=target_set,
        )


def test_metric_readout_never_records_the_key_or_the_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§15: the report carries the embedder identity, not its credentials."""
    MockEmbeddings(
        monkeypatch,
        {
            **ANCHOR_VECTORS,
            "target-x": [1.0, 0.0, 0.0],
            "target-y": [0.0, 1.0, 0.0],
        },
    )
    target_set = coverage_wiring.load_target_set(
        _write(tmp_path / "targets.json", ["target-x", "target-y"]),
        expected_dimension=DIMENSION,
    )
    readout = coverage_wiring.build_metric_readout(
        embedding=_embedding(),
        records=[_record("a0", "anchor-0", {"knowledge_domain": "k1"})],
        anchors_source="bank.jsonl",
        target_set=target_set,
    )
    serialized = json.dumps(readout.model_dump(mode="json"))
    assert API_KEY not in serialized
    assert API_BASE not in serialized
