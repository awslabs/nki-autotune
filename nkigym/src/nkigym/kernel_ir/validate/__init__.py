"""Legality checks and emission placement."""

from nkigym.kernel_ir.validate.emission import (
    Placement,
    compute_staged_set,
    material_blocking_dims,
    op_emission_placement,
    producer_stage_placement,
)
from nkigym.kernel_ir.validate.rules import validate

__all__ = [
    "Placement",
    "compute_staged_set",
    "material_blocking_dims",
    "op_emission_placement",
    "producer_stage_placement",
    "validate",
]
