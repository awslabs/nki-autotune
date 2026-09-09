"""Activation over grouped row vectors."""

from collections.abc import Mapping
from numbers import Real
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role

_OPERATIONS = {"copy": lambda data: data, "log": np.log}


class NKIGroupedVectorActivation(NKIOp):
    """Apply one activation while retaining a grouped row axis."""

    NAME: ClassVar[str] = "activation"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "G"), "dst": ("P", "G")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("P",), ("G",)) for slot in OPERAND_AXES
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "P": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions"})
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, partitions: int, op: str, scale: float = 1.0, bias: float = 0.0) -> None:
        """Configure one grouped vector activation."""
        if op not in _OPERATIONS:
            raise ValueError(f"unsupported grouped vector activation {op!r}")
        super().__init__(groups=groups, partitions=partitions, op=op, scale=scale, bias=bias)

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract:
        """Return the configured unary pointwise contract."""
        return PointwiseContract(
            operator=str(kwargs["op"]),
            input_operands=("data",),
            output_operand="dst",
            scale=float(kwargs.get("scale", 1.0)),
            bias=float(kwargs["bias"]) if isinstance(kwargs.get("bias"), Real) else 0.0,
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one on-chip source."""
        if (role := _operand_role(kwargs["data"])) is not None and role not in {"sbuf", "psum"}:
            raise TypeError(f"NKIGroupedVectorActivation(data=<role={role}>) expects on-chip data")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Apply the configured activation in fp32."""
        data = np.asarray(kwargs["data"], dtype=np.float32)
        return np.asarray(
            _OPERATIONS[str(kwargs["op"])](data * float(kwargs.get("scale", 1.0)) + float(kwargs.get("bias", 0.0))),
            dtype=np.float32,
        )


__all__ = ["NKIGroupedVectorActivation"]
