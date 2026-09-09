"""Reduce uniform chunk partials within each grouped row."""

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, ReductionContract, _operand_role, reduction_combinator

_REDUCTIONS = {"add": np.sum, "max": np.max}


class NKIGroupedTileReduce(NKIOp):
    """Reduce the chunk axis while retaining grouped row order."""

    NAME: ClassVar[str] = "tensor_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "G", "T"), "dst": ("P", "G")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "data": (("P",), ("G", "T")),
        "dst": (("P",), ("G",)),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions", "T": "chunks"}
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"T": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "GPT"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "T": None}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"T"})
    RFACTOR_RECIPE: ClassVar[Literal["rmw", "slot"] | None] = "slot"
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions", "chunks"})
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, partitions: int, chunks: int, op: str) -> None:
        """Configure one grouped chunk reduction."""
        if op not in _REDUCTIONS:
            raise ValueError(f"unsupported grouped tile reduction {op!r}")
        super().__init__(
            groups=groups, partitions=partitions, chunks=chunks, op="maximum" if op == "max" else op, axis=1
        )

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> ReductionContract:
        """Return the configured chunk reduction contract."""
        return ReductionContract(
            input_operand="data",
            output_operand="dst",
            reduction_axis="T",
            combinator=reduction_combinator(str(kwargs["op"])),
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one SBUF source."""
        if (role := _operand_role(kwargs["data"])) is not None and role != "sbuf":
            raise TypeError(f"NKIGroupedTileReduce(data=<role={role}>) expects SBUF")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return one reduced scalar per grouped row."""
        groups, partitions, chunks = (int(kwargs[name]) for name in ("groups", "partitions", "chunks"))
        data = np.asarray(kwargs["data"]).reshape(partitions, groups, chunks)
        operation = "max" if kwargs["op"] == "maximum" else str(kwargs["op"])
        return np.asarray(_REDUCTIONS[operation](data, axis=2), dtype=np.float32)


__all__ = ["NKIGroupedTileReduce"]
