"""Tests for the NKITensorTensor elementwise op (CPU sim + operand classification)."""

from __future__ import annotations

import numpy as np

from nkigym.ops.tensor_tensor import NKITensorTensor


def test_run_applies_supported_ops_elementwise() -> None:
    """CPU simulation applies every supported named operation elementwise."""
    data1 = np.full((3, 4), 3.0, dtype=np.float32)
    data2 = np.full((3, 4), 2.0, dtype=np.float32)
    for op, expected in (("add", 5.0), ("subtract", 1.0), ("multiply", 6.0)):
        out = NKITensorTensor()._run(data1=data1, data2=data2, op=op)
        np.testing.assert_allclose(out, np.full((3, 4), expected, dtype=np.float32), atol=1e-6, err_msg=op)


def test_operation_metadata_matches_isa() -> None:
    """Operand roles, axes, and renderer name match the ISA."""
    assert NKITensorTensor.OPERAND_AXES == {"data1": ("P", "F"), "data2": ("P", "F"), "dst": ("P", "F")}
    assert NKITensorTensor.RMW_OPERANDS == frozenset({"data1"})
    assert NKITensorTensor.INPUT_OPERANDS == frozenset({"data2"})
    assert NKITensorTensor.NAME == "tensor_tensor"
