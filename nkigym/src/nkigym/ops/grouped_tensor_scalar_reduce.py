"""Grouped tensor-scalar maps with fused free-axis reductions."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, ReductionContract, _operand_role, reduction_combinator

_OPERATIONS = {"equal": np.equal, "less": np.less}
_REDUCTIONS = {"add": np.sum}


class NKIGroupedTensorScalarReduce(NKIOp):
    """Reduce comparisons against grouped thresholds over one shared matrix."""

    NAME: ClassVar[str] = "tensor_scalar_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("P", "F"),
        "operand0": ("G", "P", "O"),
        "dst": ("G", "P", "F"),
        "reduce_res": ("G", "P", "O"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "data": (("P",), ("F",)),
        "operand0": (("G", "P"), ("O",)),
        "dst": (("G", "P"), ("F",)),
        "reduce_res": (("G", "P"), ("O",)),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "operand0"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions", "O": 1}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"operand0": "float32"}
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "P": 1, "F": 1, "O": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "F": None, "O": 1}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions"})
    OUTPUT_DTYPES: ClassVar[dict[str, str]] = {"dst": "float32", "reduce_res": "float32"}
    OUTPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"dst": "float32", "reduce_res": "float32"}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, partitions: int, op0: str, reduce_op: str) -> None:
        """Configure grouped thresholds and the fused reduction."""
        if op0 not in _OPERATIONS or reduce_op not in _REDUCTIONS:
            raise ValueError(f"unsupported grouped tensor-scalar reduction {op0!r}/{reduce_op!r}")
        super().__init__(groups=groups, partitions=partitions, op0=op0, reduce_op=reduce_op)

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> ReductionContract:
        """Return the configured comparison-reduction contract."""
        return ReductionContract(
            input_operand="data",
            output_operand="reduce_res",
            reduction_axis="F",
            combinator=reduction_combinator(str(kwargs["reduce_op"])),
            map_operator=str(kwargs["op0"]),
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require both operands to reside on chip."""
        if any(_operand_role(kwargs[name]) not in {None, "sbuf", "psum"} for name in ("data", "operand0")):
            raise TypeError("NKIGroupedTensorScalarReduce expects on-chip operands")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Apply the grouped comparison and reduce every shared source row."""
        groups, partitions = int(kwargs["groups"]), int(kwargs["partitions"])
        data = np.asarray(kwargs["data"]).reshape(1, partitions, -1)
        thresholds = np.asarray(kwargs["operand0"]).reshape(groups, partitions, 1)
        mapped = _OPERATIONS[str(kwargs["op0"])](data, thresholds)
        reduced = _REDUCTIONS[str(kwargs["reduce_op"])](mapped, axis=2)
        return np.asarray(reduced, dtype=np.float32).reshape(groups * partitions, 1)


__all__ = ["NKIGroupedTensorScalarReduce"]
