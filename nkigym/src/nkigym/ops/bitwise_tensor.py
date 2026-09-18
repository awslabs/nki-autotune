"""Integer bit-pattern operations through native ``tensor_tensor``."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role
from nkigym.ops.bitwise_scalar import _BITWISE


class NKIBitwiseTensor(NKIOp):
    """Apply one native bitwise operator to matching uint32 tiles."""

    NAME: ClassVar[str] = "tensor_tensor"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data1": ("P", "F"), "data2": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data1", "data2"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {operand: frozenset({"sbuf"}) for operand in INPUT_OPERANDS}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {
        operand: frozenset({"uint32"}) for operand in INPUT_OPERANDS
    }
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {operand: "uint32" for operand in INPUT_OPERANDS}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_DTYPE: ClassVar[str | None] = "uint32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "uint32"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract:
        """Describe the elementwise bitwise mapping."""
        return PointwiseContract(operator=str(kwargs["op"]), input_operands=("data1", "data2"), output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require two SBUF uint32 operands and a supported operator."""
        for operand in self.INPUT_OPERANDS:
            if _operand_role(kwargs[operand]) not in {None, "sbuf"} or np.asarray(kwargs[operand]).dtype != np.uint32:
                raise TypeError(f"NKIBitwiseTensor.{operand} requires SBUF uint32")
        if kwargs["op"] not in _BITWISE:
            raise ValueError("NKIBitwiseTensor requires a bitwise operator")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Apply one exact uint32 operation."""
        return _BITWISE[kwargs["op"]](kwargs["data1"], kwargs["data2"])
