"""Grouped exponential and partial-sum reduction."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, _operand_role


class NKIGroupedActivationReduce(NKIOp):
    """Apply a grouped broadcast bias, exponentiate, and partially sum."""

    NAME: ClassVar[str] = "activation_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("G", "Q", "P", "T", "F"),
        "bias": ("P", "G", "Q"),
        "dst": ("G", "Q", "P", "T", "F"),
        "reduce_res": ("P", "G", "Q", "T"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "data": (("G", "Q", "P"), ("T", "F")),
        "bias": (("P",), ("G", "Q")),
        "dst": (("G", "Q", "P"), ("T", "F")),
        "reduce_res": (("P",), ("G", "Q", "T")),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "bias"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {
        "G": "groups",
        "Q": "queries",
        "T": "tiles",
        "P": "partitions",
        "F": "width",
    }
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "GQTPF"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "Q": 1, "T": 1, "P": 128, "F": 512}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "queries", "tiles", "partitions", "width"})
    OUTPUT_DTYPES: ClassVar[dict[str, str]] = {"dst": "bfloat16", "reduce_res": "float32"}
    OUTPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"dst": "bfloat16", "reduce_res": "float32"}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, queries: int, tiles: int, partitions: int, width: int) -> None:
        """Configure grouped score extents."""
        super().__init__(
            groups=groups, queries=queries, tiles=tiles, partitions=partitions, width=width, op="exp", reduce_op="add"
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require on-chip data and bias."""
        for slot in ("data", "bias"):
            if (role := _operand_role(kwargs[slot])) is not None and role != "sbuf":
                raise TypeError(f"NKIGroupedActivationReduce({slot}=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return exponentials and one partial sum per score tile."""
        g, q, t, p, f = (int(kwargs[key]) for key in ("groups", "queries", "tiles", "partitions", "width"))
        data = np.asarray(kwargs["data"], dtype=np.float32).reshape(g, q, p, t, f)
        bias = np.asarray(kwargs["bias"]).reshape(p, g, q).transpose(1, 2, 0)[..., None, None]
        exponential = np.exp(data + bias)
        partial = np.sum(exponential, axis=4).transpose(2, 0, 1, 3)
        return exponential.reshape(g * q * p, t * f), partial.reshape(p, g * q * t)


__all__ = ["NKIGroupedActivationReduce"]
