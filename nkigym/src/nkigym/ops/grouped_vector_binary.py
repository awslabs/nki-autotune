"""Binary pointwise math over grouped row vectors."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role

_OPERATIONS = {"add": np.add, "maximum": np.maximum, "subtract": np.subtract}


class NKIGroupedVectorBinary(NKIOp):
    """Apply one binary operation while retaining grouped row order."""

    NAME: ClassVar[str] = "tensor_tensor"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data1": ("P", "G"), "data2": ("P", "G"), "dst": ("P", "G")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("P",), ("G",)) for slot in OPERAND_AXES
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data1", "data2"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "P": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions"})
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, partitions: int, op: str) -> None:
        """Configure one grouped vector binary operation."""
        if op not in _OPERATIONS:
            raise ValueError(f"unsupported grouped vector binary operation {op!r}")
        super().__init__(groups=groups, partitions=partitions, op=op)

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract:
        """Return the configured binary pointwise contract."""
        return PointwiseContract(operator=str(kwargs["op"]), input_operands=("data1", "data2"), output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require on-chip operands."""
        if any(_operand_role(kwargs[name]) not in {None, "sbuf", "psum"} for name in ("data1", "data2")):
            raise TypeError("NKIGroupedVectorBinary expects on-chip operands")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Apply the configured binary operation in fp32."""
        return np.asarray(
            _OPERATIONS[str(kwargs["op"])](np.asarray(kwargs["data1"]), np.asarray(kwargs["data2"])), dtype=np.float32
        )


__all__ = ["NKIGroupedVectorBinary"]
