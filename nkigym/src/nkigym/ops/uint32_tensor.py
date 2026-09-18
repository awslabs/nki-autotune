"""Exact word arithmetic through native tensor-tensor instructions."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role

_OPERATIONS = {"add": np.add, "subtract": np.subtract}


class NKIUInt32Tensor(NKIOp):
    """Apply exact modulo-word addition or subtraction to uint32 tiles."""

    NAME: ClassVar[str] = "tensor_tensor"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data1": ("P", "F"), "data2": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data1", "data2"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {operand: frozenset({"sbuf"}) for operand in INPUT_OPERANDS}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {operand: "uint32" for operand in INPUT_OPERANDS}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_DTYPE: ClassVar[str | None] = "uint32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "uint32"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require SBUF uint32 inputs and a supported native operator."""
        for operand in self.INPUT_OPERANDS:
            if _operand_role(kwargs[operand]) not in {None, "sbuf"} or np.asarray(kwargs[operand]).dtype != np.uint32:
                raise TypeError(f"NKIUInt32Tensor.{operand} requires SBUF uint32")
        if kwargs["op"] not in _OPERATIONS:
            raise ValueError(f"unsupported uint32 tensor operator {kwargs['op']!r}")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Preserve all bits of modulo-32-bit arithmetic."""
        return _OPERATIONS[kwargs["op"]](kwargs["data1"], kwargs["data2"]).astype(np.uint32)
