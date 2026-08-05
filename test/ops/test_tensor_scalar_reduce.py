"""Tests for the fused tensor-scalar reduction instruction."""

from __future__ import annotations

import numpy as np

from nkigym.ops.base import ReductionContract, reduction_combinator
from nkigym.ops.tensor_scalar_reduce import NKITensorScalarReduce


def test_cpu_sim_maps_and_reduces() -> None:
    """CPU simulation returns the reduced mapped tile."""
    data = np.arange(24, dtype=np.float32).reshape(3, 8)
    actual = NKITensorScalarReduce()._run(data=data, operand0=0.25, op0="multiply", reduce_op="max")
    expected = np.max(data * 0.25, axis=1)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_contract_resolves_affine_map_and_reducer() -> None:
    """A literal multiply is represented as a scaled identity map."""
    contract = NKITensorScalarReduce.algebraic_contract({"operand0": 0.25, "op0": "multiply", "reduce_op": "maximum"})
    assert contract == ReductionContract(
        input_operand="data",
        output_operand="reduce_res",
        reduction_axis="F",
        combinator=reduction_combinator("maximum"),
        map_operator="copy",
        scale=0.25,
    )
