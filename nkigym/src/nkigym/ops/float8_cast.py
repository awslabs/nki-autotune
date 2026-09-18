"""Float8 cast through Vector Engine ``nisa.tensor_copy``."""

from collections.abc import Mapping
from typing import Any, ClassVar

import ml_dtypes
import numpy as np

from nkigym.ops.base import CopyContract, NKIOp, _operand_role
from nkigym.ops.float32_cast import _vector_copy_parameters


class NKIFloat8Cast(NKIOp):
    """Copy one on-chip tensor into float8 storage."""

    NAME: ClassVar[str] = "activation"
    ISA_OPERAND_NAMES: ClassVar[dict[str, str]] = {"data": "src"}
    native_parameters = staticmethod(_vector_copy_parameters)
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"sbuf", "psum"})}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"data": "float32"}
    """Preserve the quantization input until the explicit float8 rounding step."""
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    OUTPUT_DTYPE: ClassVar[str | None] = "float8_e4m3"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float8_e4m3"

    def __init__(self) -> None:
        """Configure the native activation as an explicit float8 copy."""
        super().__init__(op="copy")

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> CopyContract:
        """Return the cast's copy dataflow contract."""
        _ = kwargs
        return CopyContract(input_operand="data", output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an on-chip source tensor."""
        role = _operand_role(kwargs["data"])
        if role is not None and role not in {"sbuf", "psum"}:
            raise TypeError(f"NKIFloat8Cast(data=<role={role}>) expects sbuf or psum")

    def _run(self, **kwargs: Any) -> Any:
        """Return the float8-quantized CPU result."""
        return np.asarray(kwargs["data"]).astype(ml_dtypes.float8_e4m3)


__all__ = ["NKIFloat8Cast"]
