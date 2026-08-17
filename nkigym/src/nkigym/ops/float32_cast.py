"""Cast one local tile to float32 with ``nisa.activation``."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role


class NKIFloat32Cast(NKIOp):
    """Copy one tile into a float32 destination."""

    NAME: ClassVar[str] = "activation"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self) -> None:
        """Configure the native activation as an explicit float32 copy."""
        super().__init__(op="copy")

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract:
        """Return the value-preserving cast contract."""
        _ = kwargs
        return PointwiseContract(operator="copy", input_operands=("data",), output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an on-chip source tile."""
        role = _operand_role(kwargs["data"])
        if role is not None and role not in {"sbuf", "psum"}:
            raise TypeError(f"NKIFloat32Cast(data=<role={role}>) expects sbuf or psum")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the source values represented as float32."""
        return np.asarray(kwargs["data"], dtype=np.float32)


__all__ = ["NKIFloat32Cast"]
