"""Two ordered FP32 scalar operations through one native tensor-scalar call."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, PointwiseSequenceContract, _operand_role
from nkigym.ops.tensor_scalar import _OPS


class NKITensorScalarSequence(NKIOp):
    """Apply two scalar or per-partition operators in one Vector Engine instruction."""

    NAME: ClassVar[str] = "tensor_scalar"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("P", "F"),
        "operand0": ("P", "B"),
        "operand1": ("P", "B"),
        "dst": ("P", "F"),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "operand0", "operand1"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {
        name: frozenset({"sbuf", "psum"}) for name in INPUT_OPERANDS
    }
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"B": 1}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"operand0": "float32", "operand1": "float32"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    INPLACE_OPERANDS: ClassVar[dict[str, frozenset[str]]] = {"dst": frozenset({"data"})}

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseSequenceContract:
        """Describe the two ordered scalar operations without reassociation."""
        return PointwiseSequenceContract(
            operators=(str(kwargs["op0"]), str(kwargs["op1"])),
            input_operands=("data", "operand0", "operand1"),
            output_operand="dst",
            broadcast_operands=frozenset({"operand0", "operand1"}),
            reverse=(bool(kwargs.get("reverse0", False)), bool(kwargs.get("reverse1", False))),
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require arithmetic operators, on-chip operands, and the Vector Engine."""
        for index in range(2):
            if kwargs[f"op{index}"] not in _OPS or kwargs[f"op{index}"] == "divide":
                raise ValueError("scalar sequences require supported non-division arithmetic operators")
        if kwargs.get("engine") != "vector":
            raise ValueError("scalar sequences require the Vector Engine")
        if any(_operand_role(kwargs[name]) not in {None, "sbuf", "psum"} for name in self.INPUT_OPERANDS):
            raise TypeError("scalar sequences require on-chip operands")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Evaluate each operation in float32 with scalar or row-wise broadcasting."""
        result = np.asarray(kwargs["data"], dtype=np.float32)
        for index in range(2):
            operand = np.asarray(kwargs[f"operand{index}"], dtype=np.float32)
            if operand.ndim + 1 == result.ndim:
                operand = operand[..., None]
            operands = (operand, result) if kwargs.get(f"reverse{index}", False) else (result, operand)
            result = _OPS[kwargs[f"op{index}"]](*operands)
        return result


__all__ = ["NKITensorScalarSequence"]
