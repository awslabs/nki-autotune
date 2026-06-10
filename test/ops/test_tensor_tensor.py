"""Tests for the NKITensorTensor elementwise op (CPU sim + operand classification)."""

from __future__ import annotations

import numpy as np
import pytest

from nkigym.ops.tensor_tensor import NKITensorTensor


@pytest.mark.parametrize("op,expected", [("add", 5.0), ("subtract", 1.0), ("multiply", 6.0)])
def test_run_applies_op_elementwise(op: str, expected: float) -> None:
    """CPU sim applies the named op elementwise over data1, data2."""
    data1 = np.full((3, 4), 3.0, dtype=np.float32)
    data2 = np.full((3, 4), 2.0, dtype=np.float32)
    out = NKITensorTensor()._run(data1=data1, data2=data2, op=op)
    np.testing.assert_allclose(out, np.full((3, 4), expected, dtype=np.float32), atol=1e-6)


def test_operand_axes_and_rmw() -> None:
    """data1 is the RMW accumulator; data2 is the read-only input; slots mirror the ISA."""
    assert NKITensorTensor.OPERAND_AXES == {"data1": ("P", "F"), "data2": ("P", "F"), "dst": ("P", "F")}
    assert NKITensorTensor.RMW_OPERANDS == frozenset({"data1"})
    assert NKITensorTensor.INPUT_OPERANDS == frozenset({"data2"})


def test_name_is_isa_call() -> None:
    """NAME matches the nisa call the generic renderer emits."""
    assert NKITensorTensor.NAME == "tensor_tensor"
