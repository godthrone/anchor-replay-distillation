"""Constraint evaluation and legal restricted-block enumeration for ontology v4.

Responsibility: evaluate the v4 ``constraints`` (allowed-pairs rules R1/R2/R3/
R4a/R4b and the R5 modality gate) over the six restricted axes, and answer
"how many legal restricted blocks exist and what are their six-axis
coordinates".  Pure computation over an already-parsed
:class:`~ard.core.ontology.OntologyV4`: no filesystem, network or subprocess
access (§1.3).  Reading the ontology file is
:func:`ard.backends.ontology_loader.load_ontology_v4`.
"""

from pydantic import BaseModel, ConfigDict

from ard.core.ontology import (
    AllowedPairsConstraint,
    OntologyV4,
)
from ard.core.types import AxisValuesByAxis, StringPairs

# The 11 universal axes split into 5 orthogonal (free) and 6 restricted axes.
# Both tuples are validated against the ontology's own R5/R7 declarations at
# construction time, so a drifting ontology fails loudly instead of silently.
FREE_AXES: tuple[str, ...] = (
    "language",
    "knowledge_domain",
    "response_style",
    "difficulty",
    "context_length",
)
RESTRICTED_AXES: tuple[str, ...] = (
    "capability",
    "system_prompt_mode",
    "conversation_type",
    "output_format",
    "input_condition",
    "answer_mode",
)


class ConstraintEvaluationError(ValueError):
    """Raised when the ontology's constraints cannot be evaluated as declared."""


class RestrictedBlock(BaseModel):
    """One six-axis coordinate inside the restricted subspace."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    capability: str
    system_prompt_mode: str
    conversation_type: str
    output_format: str
    input_condition: str
    answer_mode: str

    def as_dict(self) -> dict[str, str]:
        """Return the block as an ordinary ``axis -> value`` mapping."""
        return {axis: str(getattr(self, axis)) for axis in RESTRICTED_AXES}

    def label(self) -> str:
        """Return a compact ``axis=value, ...`` rendering for logs and reports."""
        return ", ".join(f"{axis}={value}" for axis, value in self.as_dict().items())


class AxisPairTable(BaseModel):
    """One resolved ``allowed_pairs`` table: from-values -> allowed to-values."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    constraint_id: str
    from_axis: str
    to_axis: str
    targets: AxisValuesByAxis


class LegalBlockCounts(BaseModel):
    """Counts of the restricted-block subspace and of the free-axis product."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    raw_restricted_block: int
    legal_restricted_block: int
    legal_restricted_block_image_capable: int
    free_axis_product: int


class ConstraintEvaluator:
    """Evaluate v4 allowed-pairs and modality-gate constraints.

    Built once from a validated :class:`~ard.core.ontology.OntologyV4`; exposes
    per-block legality checks, legal-block enumeration and the derived counts.
    """

    def __init__(self, ontology: OntologyV4) -> None:
        self.ontology: OntologyV4 = ontology
        self.free_axes: tuple[str, ...] = FREE_AXES
        self.restricted_axes: tuple[str, ...] = RESTRICTED_AXES
        self.axis_values: AxisValuesByAxis = {
            axis: ontology.axis_values(axis) for axis in RESTRICTED_AXES
        }
        self.image_capable: frozenset[str] = self._resolve_image_capable()
        self.text_only: frozenset[str] = self._resolve_text_only()
        self.tables: tuple[AxisPairTable, ...] = self._build_tables()
        self._validate_axis_roles()

    # ── Construction helpers ───────────────────────────────────────────────

    def _resolve_image_capable(self) -> frozenset[str]:
        policy = self.ontology.modality_gate().capability_image_policy
        return frozenset(policy.must_support_image) | frozenset(policy.may_support_image)

    def _resolve_text_only(self) -> frozenset[str]:
        return frozenset(self.ontology.modality_gate().capability_image_policy.text_only)

    def _build_tables(self) -> tuple[AxisPairTable, ...]:
        """Resolve every allowed-pairs constraint into a lookup table."""
        tables: list[AxisPairTable] = []
        for constraint in self.ontology.allowed_pairs_constraints():
            tables.append(self._build_table(constraint))
        return tuple(tables)

    def _build_table(self, constraint: AllowedPairsConstraint) -> AxisPairTable:
        from_axis = constraint.from_.axis
        to_axis = constraint.to.axis
        for axis in (from_axis, to_axis):
            if axis not in self.axis_values:
                raise ConstraintEvaluationError(
                    f"{constraint.id}: axis {axis!r} is not a restricted axis "
                    f"(expected one of {list(RESTRICTED_AXES)})"
                )
        from_values = self.axis_values[from_axis]
        to_values = self.axis_values[to_axis]
        targets: AxisValuesByAxis = {}
        for rule in constraint.rules:
            if rule.from_value not in from_values:
                raise ConstraintEvaluationError(
                    f"{constraint.id}: from_value {rule.from_value!r} is not a value of "
                    f"{from_axis!r}"
                )
            if rule.to_values == ["*"]:
                allowed = to_values
            else:
                unknown = [value for value in rule.to_values if value not in to_values]
                if unknown:
                    raise ConstraintEvaluationError(
                        f"{constraint.id}: to_values {unknown} are not values of {to_axis!r}"
                    )
                allowed = tuple(rule.to_values)
            targets[rule.from_value] = allowed
        uncovered = [value for value in from_values if value not in targets]
        if uncovered:
            raise ConstraintEvaluationError(
                f"{constraint.id}: from-axis {from_axis!r} has no rule for value(s) "
                f"{uncovered}; every from-value needs a row"
            )
        return AxisPairTable(
            constraint_id=constraint.id,
            from_axis=from_axis,
            to_axis=to_axis,
            targets=targets,
        )

    def _validate_axis_roles(self) -> None:
        """Check the free/restricted split against the ontology's own R5/R7."""
        declared_axes = set(self.ontology.axis_names())
        gate_axis = self.ontology.modality_gate().gated_axis
        conditional = declared_axes - set(FREE_AXES) - set(RESTRICTED_AXES)
        if conditional != {gate_axis}:
            raise ConstraintEvaluationError(
                f"axis roles: expected exactly the gated conditional axis {gate_axis!r} "
                f"outside FREE_AXES + RESTRICTED_AXES, received {sorted(conditional)}"
            )
        orthogonality = [
            constraint
            for constraint in self.ontology.constraints
            if constraint.type == "orthogonality"
        ]
        declared_free = {
            axis
            for constraint in orthogonality
            for axis in constraint.axes  # type: ignore[union-attr]
        }
        if declared_free != set(FREE_AXES):
            raise ConstraintEvaluationError(
                f"axis roles: orthogonality declares free axes {sorted(declared_free)}, "
                f"expected {sorted(FREE_AXES)}"
            )

    # ── Queries ────────────────────────────────────────────────────────────

    def raw_restricted_block_count(self) -> int:
        """Return the unconstrained product of the six restricted axes."""
        product = 1
        for axis in RESTRICTED_AXES:
            product *= len(self.axis_values[axis])
        return product

    def free_axis_product(self) -> int:
        """Return the product of the five orthogonal free axes."""
        product = 1
        for axis in FREE_AXES:
            product *= len(self.ontology.axis_values(axis))
        return product

    def is_legal_block(self, block: RestrictedBlock) -> bool:
        """Return whether ``block`` satisfies every allowed-pairs constraint."""
        values = block.as_dict()
        for table in self.tables:
            if values[table.to_axis] not in table.targets[values[table.from_axis]]:
                return False
        return True

    def enumerate_legal_blocks(
        self, *, image_capable_only: bool = False
    ) -> tuple[RestrictedBlock, ...]:
        """Enumerate every legal restricted block, in declared axis order.

        Args:
            image_capable_only: Restrict ``capability`` to the 18 image-capable
                values of the R5 policy before enumerating.
        """
        capabilities = self.axis_values["capability"]
        if image_capable_only:
            capabilities = tuple(c for c in capabilities if c in self.image_capable)
        blocks: list[RestrictedBlock] = []
        for capability in capabilities:
            for system_prompt_mode in self.axis_values["system_prompt_mode"]:
                for conversation_type in self.axis_values["conversation_type"]:
                    for output_format in self.axis_values["output_format"]:
                        for input_condition in self.axis_values["input_condition"]:
                            for answer_mode in self.axis_values["answer_mode"]:
                                block = RestrictedBlock(
                                    capability=capability,
                                    system_prompt_mode=system_prompt_mode,
                                    conversation_type=conversation_type,
                                    output_format=output_format,
                                    input_condition=input_condition,
                                    answer_mode=answer_mode,
                                )
                                if self.is_legal_block(block):
                                    blocks.append(block)
        return tuple(blocks)

    def unsolvable_capability_input_pairs(self) -> StringPairs:
        """Return the (capability, input_condition) pairs with no legal block."""
        solved = {
            (block.capability, block.input_condition) for block in self.enumerate_legal_blocks()
        }
        return tuple(
            (capability, input_condition)
            for capability in self.axis_values["capability"]
            for input_condition in self.axis_values["input_condition"]
            if (capability, input_condition) not in solved
        )

    def legal_block_counts(self) -> LegalBlockCounts:
        """Return the four derived counts of the coordinate space."""
        return LegalBlockCounts(
            raw_restricted_block=self.raw_restricted_block_count(),
            legal_restricted_block=len(self.enumerate_legal_blocks()),
            legal_restricted_block_image_capable=len(
                self.enumerate_legal_blocks(image_capable_only=True)
            ),
            free_axis_product=self.free_axis_product(),
        )


def enumerate_legal_blocks(
    ontology: OntologyV4, *, image_capable_only: bool = False
) -> tuple[RestrictedBlock, ...]:
    """Convenience wrapper: enumerate legal restricted blocks for ``ontology``."""
    return ConstraintEvaluator(ontology).enumerate_legal_blocks(
        image_capable_only=image_capable_only
    )
