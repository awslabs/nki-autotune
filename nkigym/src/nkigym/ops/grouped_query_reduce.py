"""Per-query reduction over packed grouped tile partials."""

from typing import Any, ClassVar

import numpy as np
from torch.fx import Node

from nkigym.ops.base import AxisRole, NKIOp, _operand_role

_REDUCTIONS = {"add": np.sum, "max": np.max, "maximum": np.max}


def packed_maximum(node: Node) -> bool:
    """Recognize native row maxima whose result contains values only."""
    name = str(getattr(node.target, "__name__", node.target)).removeprefix("wrapped_")
    dimension = node.kwargs.get("dim", node.kwargs.get("axis"))
    indexed = all(
        str(getattr(user.target, "__name__", user.target)).removeprefix("wrapped_") == "getitem" for user in node.users
    )
    values_only = indexed and all(user.args[1] == 0 for user in node.users)
    return (
        name == "max"
        and node.meta.get("arithmetic") == "native"
        and isinstance(dimension, int)
        and dimension == -1
        and node.kwargs.get("keepdim", node.kwargs.get("keepdims")) is True
        and (values_only if node.op == "call_method" else not indexed)
    )


class NKIGroupedQueryReduce(NKIOp):
    """Reduce tile partials independently for each group and query."""

    NAME: ClassVar[str] = "tensor_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "G", "Q", "T"), "dst": ("P", "G", "Q")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "data": (("P",), ("G", "Q", "T")),
        "dst": (("P",), ("G", "Q")),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {"data": (("P",), ("T",))}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "Q": "queries", "T": "tiles", "P": "partitions"}
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"T": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "GQTP"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "Q": 1, "T": None, "P": 128}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"T", "P"})
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "queries", "tiles", "partitions"})
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, queries: int, tiles: int, partitions: int, op: str, negate: bool = False) -> None:
        """Configure one packed group/query reduction."""
        if op not in _REDUCTIONS:
            raise ValueError(f"unsupported grouped query reduction {op!r}")
        kwargs: dict[str, object] = {
            "groups": groups,
            "queries": queries,
            "tiles": tiles,
            "partitions": partitions,
            "op": "maximum" if op == "max" else op,
            "axis": 1,
        }
        if negate:
            kwargs["negate"] = True
        super().__init__(**kwargs)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one SBUF source."""
        if (role := _operand_role(kwargs["data"])) is not None and role != "sbuf":
            raise TypeError(f"NKIGroupedQueryReduce(data=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return one partition vector per group and query."""
        g, q, t, p = (int(kwargs[key]) for key in ("groups", "queries", "tiles", "partitions"))
        data = np.asarray(kwargs["data"]).reshape(p, g, q, t)
        result = np.asarray(_REDUCTIONS[str(kwargs["op"])](data, axis=3))
        return (-result if kwargs.get("negate", False) else result).reshape(p, g * q)


__all__ = ["NKIGroupedQueryReduce"]
