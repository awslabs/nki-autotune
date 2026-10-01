"""Scale grouped chunk partials with one native activation instruction."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role


class NKIGroupedChunkScale(NKIOp):
    """Scale each chunk's row vector while retaining separate group and chunk axes."""

    NAME: ClassVar[str] = "activation"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "G", "T"), "dst": ("P", "G", "T")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("P",), ("G", "T")) for slot in OPERAND_AXES
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"sbuf", "psum"})}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"data": "float32"}
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"P": "partitions", "G": "groups", "T": "chunks"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "PGT"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "G": 1, "T": 1}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions", "chunks"})
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    INPLACE_OPERANDS: ClassVar[dict[str, frozenset[str]]] = {"dst": frozenset({"data"})}

    def __init__(self, groups: int, partitions: int, chunks: int, scale: float) -> None:
        """Configure positive grouped row extents and one constant FP32 multiplier."""
        if min(groups, partitions, chunks) < 1:
            raise ValueError("grouped chunk scale dimensions must be positive")
        super().__init__(groups=groups, partitions=partitions, chunks=chunks, scale=scale, op="copy")

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract:
        """Describe the independent scalar multiplication of each partial."""
        return PointwiseContract(
            operator="copy", input_operands=("data",), output_operand="dst", scale=float(kwargs["scale"])
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require a matching FP32 matrix in on-chip storage."""
        data = np.asarray(kwargs["data"])
        shape = int(kwargs["partitions"]), int(kwargs["groups"]) * int(kwargs["chunks"])
        if (
            _operand_role(kwargs["data"]) not in {None, "sbuf", "psum"}
            or data.dtype != np.float32
            or data.shape != shape
        ):
            raise TypeError("grouped chunk scale requires a matching on-chip FP32 matrix")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Match standalone activation-copy scaling in FP32."""
        return np.asarray(kwargs["data"], dtype=np.float32) * float(kwargs["scale"]) + 0.0


__all__ = ["NKIGroupedChunkScale"]
