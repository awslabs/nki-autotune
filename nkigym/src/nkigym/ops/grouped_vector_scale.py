"""Grouped partition-vector scaling."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIGroupedVectorScale(NKIOp):
    """Scale packed output tiles by one partition vector per group/query."""

    NAME: ClassVar[str] = "tensor_scalar"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("G", "Q", "P", "F"),
        "operand0": ("P", "G", "Q"),
        "dst": ("G", "Q", "P", "F"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "data": (("G", "Q", "P"), ("F",)),
        "operand0": (("P",), ("G", "Q")),
        "dst": (("G", "Q", "P"), ("F",)),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "operand0"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "Q": "queries", "P": "partitions", "F": "width"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "GQPF"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "Q": 1, "P": 128, "F": 512}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "queries", "partitions", "width"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    OUTPUT_DTYPE: ClassVar[str | None] = "bfloat16"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "bfloat16"

    def __init__(self, groups: int, queries: int, partitions: int, width: int) -> None:
        """Configure packed output and scale extents."""
        super().__init__(groups=groups, queries=queries, partitions=partitions, width=width, op0="multiply")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require on-chip data and scale vectors."""
        for slot in ("data", "operand0"):
            if (role := _operand_role(kwargs[slot])) is not None and role not in {"sbuf", "psum"}:
                raise TypeError(f"NKIGroupedVectorScale({slot}=<role={role}>) expects on-chip data")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return scaled packed output tiles."""
        g, q, p, f = (int(kwargs[key]) for key in ("groups", "queries", "partitions", "width"))
        data = np.asarray(kwargs["data"]).reshape(g, q, p, f)
        scale = np.asarray(kwargs["operand0"]).reshape(p, g, q).transpose(1, 2, 0)[..., None]
        return (data * scale).reshape(g * q * p, f)


__all__ = ["NKIGroupedVectorScale"]
