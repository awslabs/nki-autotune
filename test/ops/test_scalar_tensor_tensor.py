"""Tests for the scalar-tensor-tensor recurrence instruction."""

from __future__ import annotations

import numpy as np

from nkigym.ops.base import PointwiseSequenceContract
from nkigym.ops.scalar_tensor_tensor import NKIScalarTensorTensor


def test_cpu_sim_applies_scale_then_add() -> None:
    """CPU simulation broadcasts the scale and adds the tensor contribution."""
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    scale = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    contribution = np.full((3, 4), 5.0, dtype=np.float32)
    actual = NKIScalarTensorTensor()._run(data=data, operand0=scale, operand1=contribution, op0="multiply", op1="add")
    expected = data * scale[:, np.newaxis] + contribution
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_contract_and_metadata_match_isa() -> None:
    """The op declares its native operands and two-step algebra."""
    contract = NKIScalarTensorTensor.algebraic_contract({"op0": "multiply", "op1": "add"})
    assert contract == PointwiseSequenceContract(
        operators=("multiply", "add"),
        input_operands=("data", "operand0", "operand1"),
        output_operand="dst",
        broadcast_operands=frozenset({"operand0"}),
        reverse=(False, False),
    )
    assert NKIScalarTensorTensor.NAME == "scalar_tensor_tensor"
    assert NKIScalarTensorTensor.RMW_OPERANDS == frozenset()
