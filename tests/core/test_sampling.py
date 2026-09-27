"""Invariant and negative-control tests for the v5 cycle-shuffle sampler.

The v5 rule has no hard-coded counts: every number a test needs is derived from
the ontology at runtime (``unit_total`` / ``knowledge_leaf_count`` /
``visual_leaf_count`` / ``max_plan_size``), so adding an ontology leaf or block
does not make a test fail.  What is pinned instead is the *behaviour*:

* ``count=None`` is exactly one full cycle;
* within a cycle every unit appears once, so units and coordinates are distinct
  **within a cycle** — a test invariant, not a runtime gate;
* one full cycle covers every knowledge and visual leaf;
* coverage is monotone non-decreasing in N and ``plan(seed, N1)`` is a prefix of
  ``plan(seed, N2)``;
* a coordinate may recur and every occurrence is kept: the anchor id is a plan
  position (``<run>-c<cycle>p<position>``), not a content fingerprint;
* a plan's identity is reproducible across processes and distinguishes a
  different ontology / seed / N / algorithm version;
* anchor ids are pairwise distinct, byte-stable as N grows, and readable;
* only ``count >= 1`` is refused — a large N is reported, never rejected.

The negative control at the bottom is the evidence that the prefix check is not
vacuous: it mutates the free-axis derivation to depend on N and asserts the
suite's own assertion goes red.
"""

from __future__ import annotations

import os
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ard.core import sampling
from ard.core.sampling import (
    MODALITY_IMAGE,
    MODALITY_TEXT,
    PLAN_IDENTITY_ALGORITHM,
    PLAN_IDENTITY_VERSION,
    SAMPLING_ALGORITHM,
    Coordinate,
    CoverageUnit,
    PlanIdentity,
    SamplingError,
    _hash_seed,
    _iter_plan,
    build_specs,
    coverage_units,
    format_anchor_id,
    knowledge_leaf_count,
    max_plan_size,
    ontology_sha256,
    plan_rounds,
    run_key,
    sample_anchors,
    sample_coordinates,
    turn_counts_by_conversation_type,
    unit_total,
    visual_leaf_count,
)
from ard.core.types import AnchorGenerationConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
ONTOLOGY_PATH = REPO_ROOT / "ontology" / "anchor_ontology.v4.json"

RESTRICTED_AXES = (
    "capability",
    "system_prompt_mode",
    "conversation_type",
    "output_format",
    "input_condition",
    "answer_mode",
)

SEED = 20260925


# ── derivation helpers (no expected count is ever written down) ─────────────


def _block_key(coordinate: Coordinate) -> tuple[str, ...]:
    meta = coordinate.as_dict()
    return tuple(meta[axis] for axis in RESTRICTED_AXES)


def _unit_key(coordinate: Coordinate) -> tuple[str, tuple[str, ...]]:
    return (coordinate.modality, _block_key(coordinate))


def _coordinate_key(coordinate: Coordinate) -> tuple[tuple[str, str], ...]:
    return tuple(coordinate.as_dict().items())


def _is_prefix(short: list[Coordinate], long: list[Coordinate]) -> bool:
    return len(short) <= len(long) and short == long[: len(short)]


def _ids(ontology, seed: int, count: int) -> list[str]:
    """Return the ids of the first ``count`` plan positions, from the formula."""
    run = run_key(ontology, seed)
    total = unit_total(ontology)
    return [format_anchor_id(run, *divmod(index, total)) for index in range(count)]


def _identity(ontology, coordinates, *, seed: int, count: int | None) -> PlanIdentity:
    return PlanIdentity.of(
        build_specs(coordinates, ontology=ontology, seed=seed),
        ontology_sha256=ontology_sha256(ontology),
        seed=seed,
        count=count,
        unit_total=unit_total(ontology),
    )


# ── 1. count=None is exactly one full cycle ─────────────────────────────────


def test_count_none_is_exactly_one_full_cycle(ontology) -> None:
    total = unit_total(ontology)
    plan = sample_coordinates(ontology, seed=SEED)

    assert len(plan) == total
    assert len({_unit_key(coordinate) for coordinate in plan}) == total
    declared = {
        (unit.modality, tuple(unit.block.as_dict()[axis] for axis in RESTRICTED_AXES))
        for unit in coverage_units(ontology)
    }
    assert {_unit_key(coordinate) for coordinate in plan} == declared


def test_coverage_units_are_text_then_image_in_enumeration_order(ontology) -> None:
    units = coverage_units(ontology)
    modalities = [unit.modality for unit in units]

    assert all(isinstance(unit, CoverageUnit) for unit in units)
    assert modalities == sorted(modalities, key=lambda m: 0 if m == MODALITY_TEXT else 1)
    assert modalities.count(MODALITY_TEXT) >= 1
    assert modalities.count(MODALITY_IMAGE) >= 1
    assert len(units) == unit_total(ontology)
    assert max_plan_size(ontology) == knowledge_leaf_count(ontology) * len(units)


# ── 2. in-cycle distinctness (a test invariant, no runtime gate) ────────────


@pytest.mark.parametrize("count", [1, 137, 900])
def test_within_a_cycle_units_and_coordinates_are_distinct(ontology, count: int) -> None:
    plan = sample_coordinates(ontology, seed=SEED, count=count)

    assert len(plan) == count
    assert len({_unit_key(coordinate) for coordinate in plan}) == count
    assert len({_coordinate_key(coordinate) for coordinate in plan}) == count


def test_a_full_cycle_repeats_no_unit(ontology) -> None:
    total = unit_total(ontology)
    plan = sample_coordinates(ontology, seed=SEED, count=total)

    assert len({_unit_key(coordinate) for coordinate in plan}) == total


# ── 3. one cycle covers every leaf ──────────────────────────────────────────


@pytest.mark.parametrize("extra", [0, 137])
def test_a_full_cycle_covers_every_knowledge_and_visual_leaf(ontology, extra: int) -> None:
    total = unit_total(ontology)
    plan = sample_coordinates(ontology, seed=SEED, count=total + extra)

    knowledge = {coordinate.knowledge_domain for coordinate in plan}
    visual = {
        coordinate.visual_domain for coordinate in plan if coordinate.visual_domain is not None
    }
    assert knowledge == set(ontology.axis_values("knowledge_domain"))
    assert len(knowledge) == knowledge_leaf_count(ontology)
    assert visual == set(ontology.axis_values("visual_domain"))
    assert len(visual) == visual_leaf_count(ontology)

    for coordinate in plan:
        if coordinate.modality == MODALITY_TEXT:
            assert coordinate.visual_domain is None
        else:
            assert coordinate.modality == MODALITY_IMAGE
            assert coordinate.visual_domain is not None


# ── 4 & 5. monotone coverage and the prefix property ────────────────────────


def test_coverage_is_monotone_non_decreasing_in_n(ontology) -> None:
    total = unit_total(ontology)
    counts = [1, 17, 500, total, total + 1, 2 * total]

    units_seen = 0
    knowledge_seen: set[str] = set()
    visual_seen: set[str] = set()
    knowledge_count = 0
    visual_count = 0
    previous: list[Coordinate] = []

    for count in counts:
        plan = sample_coordinates(ontology, seed=SEED, count=count)
        assert _is_prefix(previous, plan)
        previous = plan

        units = {_unit_key(coordinate) for coordinate in plan}
        assert len(units) >= units_seen
        units_seen = len(units)

        knowledge_seen |= {coordinate.knowledge_domain for coordinate in plan}
        visual_seen |= {
            coordinate.visual_domain for coordinate in plan if coordinate.visual_domain is not None
        }
        assert len(knowledge_seen) >= knowledge_count
        assert len(visual_seen) >= visual_count
        knowledge_count = len(knowledge_seen)
        visual_count = len(visual_seen)

    assert units_seen == total
    assert len(knowledge_seen) == knowledge_leaf_count(ontology)
    assert len(visual_seen) == visual_leaf_count(ontology)


def test_plan_is_a_prefix_of_every_longer_plan(ontology) -> None:
    total = unit_total(ontology)
    short = sample_coordinates(ontology, seed=SEED, count=max(1, total // 3))
    long = sample_coordinates(ontology, seed=SEED, count=3 * total)

    assert _is_prefix(short, long)
    assert short == long[: len(short)]


def test_a_different_seed_builds_a_different_plan(ontology) -> None:
    a = sample_coordinates(ontology, seed=SEED, count=200)
    b = sample_coordinates(ontology, seed=SEED + 1, count=200)

    assert a != b


# ── 6. cross-process determinism ────────────────────────────────────────────

_SUBPROCESS_SCRIPT = """
import sys, types

# Import the package without running `ard/__init__.py`: this test only needs the
# core sampling layer, and pre-registering the package keeps the fingerprint
# check independent of unrelated modules' import side effects.
package = types.ModuleType("ard")
package.__path__ = [{ard_path!r}]
sys.modules["ard"] = package

from ard.backends.ontology_loader import load_ontology_v4
from ard.core import sampling
from ard.core.types import AnchorGenerationConfig

ontology = load_ontology_v4({ontology_path!r})
count = 25
seed = {seed!r}
specs = sampling.sample_anchors(ontology, AnchorGenerationConfig(seed=seed), count=count)
print(sampling.run_key(ontology, seed))
print(" ".join(spec.id for spec in specs[:3]))
print(sampling.PlanIdentity.of(
    specs,
    ontology_sha256=sampling.ontology_sha256(ontology),
    seed=seed,
    count=count,
    unit_total=sampling.unit_total(ontology),
).digest)
"""


def test_run_key_ids_and_identity_are_identical_across_processes(ontology) -> None:
    script = _SUBPROCESS_SCRIPT.format(
        ard_path=str(REPO_ROOT / "src" / "ard"),
        ontology_path=str(ONTOLOGY_PATH),
        seed=SEED,
    )
    outputs = []
    for hash_seed in ("0", "1", "31337"):
        environment = dict(os.environ, PYTHONHASHSEED=hash_seed)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(result.stdout.strip())

    assert len(set(outputs)) == 1

    specs = sample_anchors(ontology, AnchorGenerationConfig(seed=SEED), count=25)
    in_process = "\n".join(
        [
            run_key(ontology, SEED),
            " ".join(spec.id for spec in specs[:3]),
            _identity(
                ontology,
                sample_coordinates(ontology, seed=SEED, count=25),
                seed=SEED,
                count=25,
            ).digest,
        ]
    )
    assert outputs[0] == in_process


# ── 7. identity distinguishes ontology / seed / N / algorithm version ───────


def test_identity_is_stable_for_the_same_inputs(ontology) -> None:
    plan = sample_coordinates(ontology, seed=SEED, count=64)
    first = _identity(ontology, plan, seed=SEED, count=64)
    second = _identity(ontology, plan, seed=SEED, count=64)

    assert first == second
    assert first.version == PLAN_IDENTITY_VERSION == 2
    assert first.algorithm == PLAN_IDENTITY_ALGORITHM
    assert first.sampling == SAMPLING_ALGORITHM == "cycle-shuffle/v1"
    assert first.unit_total == unit_total(ontology)
    assert first.plan_size == 64
    assert first.count == 64
    assert set(first.as_dict()) == {
        "algorithm",
        "version",
        "sampling",
        "ontology_sha256",
        "seed",
        "count",
        "unit_total",
        "plan_size",
        "digest",
    }


def test_identity_distinguishes_ontology_seed_n_and_algorithm(ontology, monkeypatch) -> None:
    plan = sample_coordinates(ontology, seed=SEED, count=64)
    base = _identity(ontology, plan, seed=SEED, count=64)

    other_seed = sample_coordinates(ontology, seed=SEED + 1, count=64)
    assert _identity(ontology, other_seed, seed=SEED + 1, count=64).digest != base.digest

    longer = sample_coordinates(ontology, seed=SEED, count=65)
    assert _identity(ontology, longer, seed=SEED, count=65).digest != base.digest

    foreign_ontology = PlanIdentity.of(
        build_specs(plan, ontology=ontology, seed=SEED),
        ontology_sha256="0" * 64,
        seed=SEED,
        count=64,
        unit_total=unit_total(ontology),
    )
    assert foreign_ontology.digest != base.digest

    monkeypatch.setattr(sampling, "PLAN_IDENTITY_VERSION", PLAN_IDENTITY_VERSION + 1)
    assert _identity(ontology, plan, seed=SEED, count=64).digest != base.digest
    monkeypatch.undo()

    monkeypatch.setattr(sampling, "SAMPLING_ALGORITHM", "cycle-shuffle/v2")
    assert _identity(ontology, plan, seed=SEED, count=64).digest != base.digest


# ── 8. anchor ids: serial numbers, not content fingerprints ─────────────────


def test_run_key_is_content_derived_and_scoped_to_the_run(ontology, monkeypatch) -> None:
    key = run_key(ontology, SEED)

    assert re.fullmatch(r"[0-9a-f]{8}", key)
    assert run_key(ontology, SEED) == key
    assert run_key(ontology, SEED + 1) != key

    mutated = ontology.model_copy(update={"version": "0.0.0-not-the-shipped-one"})
    assert ontology_sha256(mutated) != ontology_sha256(ontology)
    assert run_key(mutated, SEED) != key

    monkeypatch.setattr(sampling, "SAMPLING_ALGORITHM", "cycle-shuffle/v2")
    assert run_key(ontology, SEED) != key


def test_anchor_id_is_readable_and_recovers_the_plan_position(ontology) -> None:
    total = unit_total(ontology)
    run = run_key(ontology, SEED)
    plan = sample_coordinates(ontology, seed=SEED, count=total + 4)
    specs = build_specs(plan, ontology=ontology, seed=SEED)

    for index, spec in enumerate(specs):
        cycle, position = divmod(index, total)
        assert spec.id == f"{run}-c{cycle:05d}p{position:05d}"
        match = re.fullmatch(r"(?P<run>[0-9a-f]{8})-c(?P<cycle>\d{5})p(?P<position>\d{5})", spec.id)
        assert match is not None
        assert int(match["cycle"]) == cycle
        assert int(match["position"]) == position

    assert specs[0].id.endswith("-c00000p00000")
    assert re.fullmatch(r"[0-9a-f]{8}-c\d{5}p\d{5}", specs[-1].id)


def test_anchor_ids_are_byte_stable_as_n_grows(ontology) -> None:
    total = unit_total(ontology)
    steps = [5, 5 + total, 2 * total + 7]

    previous: list[str] = []
    for count in steps:
        specs = sample_anchors(ontology, AnchorGenerationConfig(seed=SEED), count=count)
        ids = [spec.id for spec in specs]
        assert len(ids) == count
        assert ids[: len(previous)] == previous, f"growing N to {count} rewrote an existing id"
        assert len(set(ids)) == count
        previous = ids


def test_anchor_ids_are_pairwise_distinct_with_no_dropped_entries(ontology) -> None:
    total = unit_total(ontology)
    for cycles in (2, 3):
        plan = sample_coordinates(ontology, seed=SEED, count=cycles * total)
        ids = [spec.id for spec in build_specs(plan, ontology=ontology, seed=SEED)]

        assert len(plan) == cycles * total
        assert len(ids) == cycles * total
        assert len(set(ids)) == cycles * total


def test_anchor_ids_stay_distinct_beyond_the_full_rotation(ontology) -> None:
    total = unit_total(ontology)
    count = max_plan_size(ontology) + 3 * total
    ids = _ids(ontology, SEED, count)

    assert len(ids) == count
    assert len(set(ids)) == count


def test_a_recurring_coordinate_is_kept_every_time(ontology) -> None:
    """The same coordinate may be sampled twice; both samples must survive.

    Coordinate content is not identity.  A stochastic generator asked twice with
    the same labels returns a different question and a different answer, so the
    second occurrence is a new sample and no gate may drop it.  The plan is
    streamed only until the first recurrence, which keeps the check to one pass
    over the prefix where content can wrap.
    """
    total = unit_total(ontology)
    cap = max_plan_size(ontology) + 40 * total
    run = run_key(ontology, SEED)

    entries: list[tuple[Coordinate, int]] = []
    first_seen: dict[tuple[tuple[str, str], ...], int] = {}
    recurrence: tuple[int, int] | None = None

    for index, (coordinate, cycle) in enumerate(
        _iter_plan(ontology, SEED, count=cap, per_modality=None)
    ):
        entries.append((coordinate, cycle))
        key = _coordinate_key(coordinate)
        if key in first_seen:
            recurrence = (first_seen[key], index)
            break
        first_seen[key] = index

    assert recurrence is not None, f"no coordinate recurred within {cap} entries"
    earlier, later = recurrence
    assert earlier < later
    assert _coordinate_key(entries[earlier][0]) == _coordinate_key(entries[later][0])

    earlier_id = format_anchor_id(run, *divmod(earlier, total))
    later_id = format_anchor_id(run, *divmod(later, total))
    assert earlier_id != later_id
    assert len(entries) == later + 1, "an entry was dropped from the plan"


def test_growing_n_by_one_cycle_revisits_every_unit_and_keeps_both(ontology) -> None:
    total = unit_total(ontology)
    plan = sample_coordinates(ontology, seed=SEED, count=2 * total)

    first, second = plan[:total], plan[total:]
    assert len(plan) == 2 * total
    assert {_unit_key(coordinate) for coordinate in first} == {
        _unit_key(coordinate) for coordinate in second
    }
    assert len({_unit_key(coordinate) for coordinate in first}) == total
    ids = [spec.id for spec in build_specs(plan, ontology=ontology, seed=SEED)]
    assert len(set(ids)) == 2 * total


def test_plan_rounds_decomposes_a_request(ontology) -> None:
    total = unit_total(ontology)

    assert plan_rounds(ontology, 1) == (0, 1)
    assert plan_rounds(ontology, total - 1) == (0, total - 1)
    assert plan_rounds(ontology, total) == (1, 0)
    assert plan_rounds(ontology, total + 4) == (1, 4)
    assert plan_rounds(ontology, 3 * total) == (3, 0)


def test_count_above_the_full_rotation_is_accepted(ontology) -> None:
    rotation = max_plan_size(ontology)
    units_per_cycle = unit_total(ontology)

    # N has no ceiling: the request is reported, not refused.
    assert plan_rounds(ontology, rotation + 1) == (
        knowledge_leaf_count(ontology),
        1,
    )

    head = []
    for coordinate, cycle in _iter_plan(ontology, SEED, count=rotation + 1, per_modality=None):
        head.append((coordinate, cycle))
        if len(head) == 3:
            break
    assert len(head) == 3
    assert [cycle for _, cycle in head] == [0, 0, 0]
    assert units_per_cycle >= 1


@pytest.mark.parametrize("count", [0, -1])
def test_counts_below_one_are_refused_with_the_plan_shape_in_the_message(
    ontology, count: int
) -> None:
    with pytest.raises(SamplingError) as error:
        sample_coordinates(ontology, seed=SEED, count=count)

    message = str(error.value)
    assert str(count) in message
    assert str(max_plan_size(ontology)) in message
    assert str(unit_total(ontology)) in message
    assert "round" in message

    with pytest.raises(SamplingError):
        plan_rounds(ontology, count)


def test_anchor_id_rejects_a_value_that_would_not_fit_the_field(ontology) -> None:
    run = run_key(ontology, SEED)

    with pytest.raises(ValueError, match="cycle"):
        format_anchor_id(run, -1, 0)
    with pytest.raises(ValueError, match="position"):
        format_anchor_id(run, 0, 100000)


# ── 9. per_modality: the smoke subset of cycle 0 ────────────────────────────


def test_per_modality_takes_the_first_units_of_each_modality_from_cycle_zero(
    ontology,
) -> None:
    plan = sample_coordinates(ontology, seed=SEED, per_modality=(4, 4))

    assert len(plan) == 8
    assert [coordinate.modality for coordinate in plan].count(MODALITY_TEXT) == 4
    assert [coordinate.modality for coordinate in plan].count(MODALITY_IMAGE) == 4

    units = coverage_units(ontology)
    positions = list(range(len(units)))
    random.Random(_hash_seed(SEED, ontology_sha256(ontology), "cycle", 0)).shuffle(positions)
    expected_text = [index for index in positions if units[index].modality == MODALITY_TEXT][:4]
    expected_image = [index for index in positions if units[index].modality == MODALITY_IMAGE][:4]
    expected = expected_text + expected_image

    leaves = ontology.axis_values("knowledge_domain")
    for coordinate, unit_index in zip(plan, expected, strict=True):
        assert coordinate.modality == units[unit_index].modality
        assert coordinate.capability == units[unit_index].block.capability
        assert coordinate.knowledge_domain == leaves[unit_index % len(leaves)]


def test_per_modality_rejects_unusable_requests(ontology) -> None:
    units = coverage_units(ontology)
    text_units = sum(1 for unit in units if unit.modality == MODALITY_TEXT)
    image_units = sum(1 for unit in units if unit.modality == MODALITY_IMAGE)

    with pytest.raises(SamplingError, match="mutually exclusive"):
        sample_coordinates(ontology, seed=SEED, count=8, per_modality=(4, 4))
    with pytest.raises(SamplingError, match="text count"):
        sample_coordinates(ontology, seed=SEED, per_modality=(text_units + 1, 0))
    with pytest.raises(SamplingError, match="image count"):
        sample_coordinates(ontology, seed=SEED, per_modality=(0, image_units + 1))
    with pytest.raises(SamplingError, match=r"\(0, 0\)"):
        sample_coordinates(ontology, seed=SEED, per_modality=(0, 0))
    with pytest.raises(SamplingError, match="tuple"):
        sample_coordinates(ontology, seed=SEED, per_modality=4)  # type: ignore[arg-type]


# ── 10. specs and the v4 record shape ───────────────────────────────────────


def test_build_specs_numbers_positions_and_keeps_the_v4_metadata(ontology) -> None:
    total = unit_total(ontology)
    coordinates = sample_coordinates(ontology, seed=SEED, count=total + 3)
    specs = sample_anchors(ontology, AnchorGenerationConfig(seed=SEED), count=total + 3)
    run = run_key(ontology, SEED)

    assert len(coordinates) == total + 3
    assert len(specs) == len(coordinates)
    turn_counts = turn_counts_by_conversation_type(ontology)
    for index, spec in enumerate(specs):
        assert spec.id == format_anchor_id(run, *divmod(index, total))
        assert "cycle" not in spec.anchor_meta
        assert spec.anchor_meta == coordinates[index].as_dict()
        assert len(spec.turns) == turn_counts[spec.anchor_meta["conversation_type"]]
        assert len(spec.turns) % 2 == 1


def test_declared_turn_counts_are_odd_and_ordered(ontology) -> None:
    counts = turn_counts_by_conversation_type(ontology)

    assert set(counts) == set(ontology.axis_values("conversation_type"))
    assert all(value >= 1 and value % 2 == 1 for value in counts.values())


# ── negative control: the evidence that the prefix check is not vacuous ─────


def test_negative_control_prefix_check_catches_a_count_dependent_plan(
    ontology, monkeypatch
) -> None:
    """A derivation that consults N breaks the prefix property; the check fires.

    The real free-axis derivation depends on the plan index and on nothing else
    (:func:`sampling._free_rng`), which is what makes every longer plan an
    extension of every shorter one.  Here the derivation is mutated to also
    depend on N: ``plan(10)`` is then no longer a prefix of ``plan(20)``, and
    :func:`_is_prefix` — the same assertion
    :func:`test_plan_is_a_prefix_of_every_longer_plan` uses — reports it.
    """
    holder = {"n": 0}

    def count_dependent(seed: int, fingerprint: str, plan_index: int) -> random.Random:
        return random.Random(_hash_seed(seed, fingerprint, "free", plan_index, holder["n"]))

    monkeypatch.setattr(sampling, "_free_rng", count_dependent)

    holder["n"] = 10
    short = sample_coordinates(ontology, seed=SEED, count=10)
    holder["n"] = 20
    long = sample_coordinates(ontology, seed=SEED, count=20)

    assert not _is_prefix(short, long)

    monkeypatch.undo()
    assert _is_prefix(
        sample_coordinates(ontology, seed=SEED, count=10),
        sample_coordinates(ontology, seed=SEED, count=20),
    )


def test_negative_control_free_axis_without_plan_index_is_not_caught_by_prefix(
    ontology, monkeypatch
) -> None:
    """Dropping ``i`` from the free draw is invisible to the prefix check.

    The mutation makes every plan position draw the same four axes, yet each
    coordinate is still built without looking at N, so the prefix property
    survives — which is why the prefix check alone is not a sufficient negative
    control and the count-dependent mutation above is needed.  What the mutation
    does destroy is index-addressability, asserted here directly.
    """
    total = unit_total(ontology)
    monkeypatch.setattr(
        sampling,
        "_free_rng",
        lambda seed, fingerprint, plan_index: random.Random(_hash_seed(seed, fingerprint, "free")),
    )
    plan = sample_coordinates(ontology, seed=SEED, count=2 * total)

    first, second = plan[:total], plan[total : 2 * total]
    assert _is_prefix(first, plan)
    for left, right in zip(first, second, strict=True):
        assert left.language == right.language
        assert left.response_style == right.response_style
        assert left.difficulty == right.difficulty
        assert left.context_length == right.context_length
