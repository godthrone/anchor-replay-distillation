"""Ontology parsing for ARD anchor generation (v4 schema).

Responsibility: validate an **already-decoded** ``anchor_ontology.v4.json``
payload (the axes the ontology declares, its knowledge-domain tree, its
constraint declarations) into pydantic models and expose typed access to axis
value sets.  A payload that does not
match the v4 schema raises :class:`OntologySchemaError` naming the offending
field, the expectation and the received value — it never degrades into an empty
result set.

This module is pure computation (§1.3): it never opens a file.  Reading the JSON
document off disk is the facility half, in
:func:`ard.backends.ontology_loader.load_ontology_v4`, which decodes the bytes
and hands the result to the single schema gate :func:`parse_ontology_v4` below.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError

from ard.core.types import StringList

# ── Schema configuration ───────────────────────────────────────────────────

_MISSING = "<nothing>"

_EXPECTED_BY_ERROR_TYPE: dict[str, str] = {
    "missing": "field present",
    "extra_forbidden": "a declared field",
    "string_type": "str",
    "string_too_short": "str",
    "int_type": "int",
    "int_parsing": "int",
    "bool_type": "bool",
    "bool_parsing": "bool",
    "list_type": "list",
    "dict_type": "dict",
    "tuple_type": "tuple",
    "model_type": "object",
    "literal_error": "one of the declared literal values",
}


class OntologySchemaError(ValueError):
    """Raised when an ontology file does not conform to the v4 schema.

    Attributes:
        path: The ontology path the problem was found in.
        problems: One human-readable problem per schema violation.
    """

    def __init__(self, path: str | Path, problems: list[str]) -> None:
        self.path: Path = Path(path)
        self.problems: tuple[str, ...] = tuple(problems)
        detail = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(
            f"ontology schema error in {self.path} ({len(self.problems)} problem(s)):\n{detail}"
        )


# ── Axis models ────────────────────────────────────────────────────────────


class AxisGroup(BaseModel):
    """One named value group of a grouped axis."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    values: list[str]
    definition: str | None = None


class TurnAttributes(RootModel[dict[str, int | str]]):
    """Turn-count attributes of a single ``conversation_type`` value."""


class SubdomainLeaves(RootModel[dict[str, StringList]]):
    """Leaf coordinate values of one knowledge domain, keyed by subdomain name."""


class KnowledgeDomainTree(RootModel[dict[str, SubdomainLeaves]]):
    """The ``knowledge_domain`` hierarchy: domain name -> subdomains -> leaves."""


class FlatAxis(BaseModel):
    """An axis whose coordinate values are listed verbatim in ``values``."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    layer: str
    cardinality: Literal["single"]
    applies_to: str | dict[str, str]
    values: list[str]


class FlatAxisWithSemantics(FlatAxis):
    """A flat axis that carries an explicit ``semantics`` clause (``language``)."""

    semantics: str


class FlatAxisWithNote(FlatAxis):
    """A flat axis that carries a clarifying ``note``."""

    note: str


class FlatAxisWithDefinitions(FlatAxis):
    """A flat axis whose values each carry a definition (``system_prompt_mode``)."""

    value_definitions: dict[str, str]
    note: str


class FlatAxisWithAttributes(FlatAxis):
    """A flat axis whose values each carry attributes (``conversation_type``)."""

    value_attributes: dict[str, TurnAttributes]
    note: str


class HierarchicalAxis(BaseModel):
    """An axis whose values are the deepest leaves of a named tree."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    layer: str
    cardinality: Literal["single"]
    applies_to: str | dict[str, str]
    value_model: Literal["hierarchical"]
    levels: list[str]
    leaf_is_value: bool
    tree_ref: str


class GroupedAxis(BaseModel):
    """An axis whose values are the concatenation of its groups' values."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    layer: str
    cardinality: Literal["single"]
    applies_to: str | dict[str, str]
    value_model: Literal["grouped"]
    groups: dict[str, AxisGroup]


class ConditionalGroupedAxis(GroupedAxis):
    """A grouped axis that exists only for one modality (``visual_domain``)."""

    absent_when: dict[str, str]
    note: str


AxisSpec = FlatAxis | HierarchicalAxis | GroupedAxis


class Axes(BaseModel):
    """The twelve declared axes, one field per axis name."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    language: FlatAxisWithSemantics
    knowledge_domain: HierarchicalAxis
    capability: GroupedAxis
    system_prompt_mode: FlatAxisWithDefinitions
    conversation_type: FlatAxisWithAttributes
    response_style: FlatAxis
    output_format: FlatAxis
    difficulty: FlatAxis
    context_length: FlatAxis
    input_condition: FlatAxisWithNote
    answer_mode: FlatAxis
    visual_domain: ConditionalGroupedAxis

    def spec(self, axis: str) -> AxisSpec:
        """Return the model of ``axis``.

        Raises:
            KeyError: If ``axis`` is not one of the twelve declared axes.
        """
        if axis not in type(self).model_fields:
            raise KeyError(f"unknown ontology axis: {axis!r}")
        return getattr(self, axis)  # type: ignore[no-any-return]


# ── Constraint models ──────────────────────────────────────────────────────


class AxisRef(BaseModel):
    """A reference to one axis by name, as used inside a constraint rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    axis: str


class AllowedPairRule(BaseModel):
    """One ``from_value -> to_values`` row; ``["*"]`` means every target value."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    from_value: str
    to_values: list[str]


class ConstraintCoverage(BaseModel):
    """A coverage requirement attached to an ``allowed_pairs`` constraint."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    axis: str
    require: str


class ConstraintRelaxation(BaseModel):
    """A documented relaxation applied to one rule row in a previous round."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    from_value: str
    added: str
    reason: str


class AllowedPairsConstraint(BaseModel):
    """A binary ``from.axis -> to.axis`` allowed-pairs table (R1/R2/R3/R4a/R4b)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    type: Literal["allowed_pairs"]
    from_: AxisRef = Field(alias="from")
    to: AxisRef
    rules: list[AllowedPairRule]
    rationale: str
    coverage: ConstraintCoverage | None = None
    relaxations_vs_previous_round: list[ConstraintRelaxation] | None = None


class ConstraintSampleField(BaseModel):
    """A sample field referenced by a constraint rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    values: list[str]


class CapabilityImagePolicy(BaseModel):
    """Which capabilities may carry an image, per the modality gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    must_support_image: list[str]
    may_support_image: list[str]
    text_only: list[str]


class ModalityGateConstraint(BaseModel):
    """The conditional-axis gate that switches ``visual_domain`` on and off (R5)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    type: Literal["modality_gate"]
    sample_field: ConstraintSampleField
    gated_axis: str
    rule: str
    capability_image_policy: CapabilityImagePolicy
    rationale: str


class SourceLanguagePolicy(BaseModel):
    """Where the source language lives when ``language`` is the output language."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    carrier: str
    invariant: str
    required_prompt_metadata: list[str]
    prompt_pairing: str


class SemanticsDeclarationConstraint(BaseModel):
    """A meaning declaration with no combinatorial effect (R6)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    type: Literal["semantics_declaration"]
    axis: str
    meaning: str
    applies_to_capabilities: str
    special_capabilities: dict[str, str]
    source_language_policy: SourceLanguagePolicy
    rejected_alternative: str
    cost: list[str]
    rationale: str


class OrthogonalityConstraint(BaseModel):
    """A declaration that a set of axes is mutually free (R7)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    type: Literal["orthogonality"]
    axes: list[str]
    rule: str
    exceptions: list[str]
    rationale: str


OntologyConstraint = Annotated[
    AllowedPairsConstraint
    | ModalityGateConstraint
    | SemanticsDeclarationConstraint
    | OrthogonalityConstraint,
    Field(discriminator="type"),
]


# ── Top-level models ───────────────────────────────────────────────────────


class SampleField(BaseModel):
    """A top-level ``sample_fields`` entry (a field that is not an axis)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    values: list[str]
    role: str


class PromptWordingLocation(BaseModel):
    """Recommended location for generator-facing prompt wording."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    target: str
    shape: str
    status: str


class WordingPolicy(BaseModel):
    """The separation between coordinate declarations and prompt wording."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    ontology_holds: str
    banned_in_ontology: list[str]
    prompt_wording_location_recommendation: PromptWordingLocation
    rationale: str


class OntologyV4(BaseModel):
    """A fully validated ``anchor_ontology.v4.json`` coordinate ontology."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    ontology_id: str
    version: str
    supersedes: str
    scope_note: str
    axis_order: list[str]
    reading_rules: dict[str, str]
    axes: Axes
    knowledge_domain_tree: KnowledgeDomainTree
    constraints: list[OntologyConstraint]
    sample_fields: dict[str, SampleField]
    wording_policy: WordingPolicy

    def axis_names(self) -> tuple[str, ...]:
        """Return the declared axis order."""
        return tuple(self.axis_order)

    def axis_values(self, axis: str) -> tuple[str, ...]:
        """Return every coordinate value of ``axis``.

        Resolution follows ``reading_rules.axis_value_set``: flat axes use
        ``values``, hierarchical axes use the deepest tree leaves, grouped axes
        use the concatenation of each group's values.

        Raises:
            KeyError: If ``axis`` is not one of the twelve declared axes.
            OntologySchemaError: If a hierarchical axis points at an unknown tree.
        """
        spec = self.axes.spec(axis)
        if isinstance(spec, HierarchicalAxis):
            if spec.tree_ref != "knowledge_domain_tree":
                raise OntologySchemaError(
                    "<in-memory ontology>",
                    [
                        f"axes.{axis}.tree_ref: expected 'knowledge_domain_tree', "
                        f"received {spec.tree_ref!r}"
                    ],
                )
            return tuple(
                leaf
                for subdomains in self.knowledge_domain_tree.root.values()
                for leaves in subdomains.root.values()
                for leaf in leaves
            )
        if isinstance(spec, GroupedAxis):
            return tuple(value for group in spec.groups.values() for value in group.values)
        return tuple(spec.values)

    def allowed_pairs_constraints(self) -> tuple[AllowedPairsConstraint, ...]:
        """Return every ``allowed_pairs`` constraint, in declaration order."""
        return tuple(c for c in self.constraints if isinstance(c, AllowedPairsConstraint))

    def modality_gate(self) -> ModalityGateConstraint:
        """Return the single modality-gate constraint.

        Raises:
            OntologySchemaError: If the file does not declare exactly one gate.
        """
        gates = [c for c in self.constraints if isinstance(c, ModalityGateConstraint)]
        if len(gates) != 1:
            raise OntologySchemaError(
                "<in-memory ontology>",
                [
                    "constraints: expected exactly one modality_gate constraint, "
                    f"received {len(gates)}"
                ],
            )
        return gates[0]


# ── Loading ────────────────────────────────────────────────────────────────


def _format_problem(error: Mapping[str, Any]) -> str:
    """Render one pydantic error as ``field: expectation (received: ...)``."""
    location = ".".join(str(part) for part in error["loc"]) or "<root>"
    error_type = str(error.get("type", "unknown"))
    expected = _EXPECTED_BY_ERROR_TYPE.get(error_type, str(error.get("msg", "valid value")))
    if error_type == "missing":
        received = _MISSING
    else:
        received = repr(error.get("input"))
    return f"{location}: expected {expected} (received: {received})"


def _format_validation_error(error: ValidationError) -> list[str]:
    """Render every pydantic problem of ``error`` as a readable line."""
    return [_format_problem(item) for item in error.errors()]


def parse_ontology_v4(raw: object, source: str | Path = "<inline>") -> OntologyV4:
    """Validate an already-decoded ontology v4 payload.

    Args:
        raw: The decoded JSON document (the ``json.loads`` result of a file).
        source: Label used in error messages only — normally the path the payload
            came from, so a schema problem names where to look.  Nothing is read
            from the filesystem here.

    Returns:
        The validated :class:`OntologyV4` model.

    Raises:
        OntologySchemaError: If *raw* is not a JSON object, or does not match the
            v4 schema (missing / extra / wrongly typed field).
    """
    if not isinstance(raw, dict):
        raise OntologySchemaError(
            source,
            [f"<root>: expected object (received: {type(raw).__name__})"],
        )
    try:
        return OntologyV4.model_validate(raw)
    except ValidationError as exc:
        raise OntologySchemaError(source, _format_validation_error(exc)) from exc
