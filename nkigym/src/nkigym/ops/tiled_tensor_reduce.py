"""Reduction over flattened tile axes through ``nisa.tensor_reduce``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role

_REDUCTIONS = {"add": np.sum, "max": np.max, "maximum": np.max}


class NKITiledTensorReduce(NKIOp):
    """Reduce chunk and tile axes while preserving packed groups."""

    NAME: ClassVar[str] = "tensor_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("M", "C", "T", "G", "N"), "dst": ("M", "G", "N")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "data": (("M",), ("C", "T", "G", "N")),
        "dst": (("M",), ("G", "N")),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "data": (("M",), ("G",), ("N",), ("C", "T"))
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"C": "chunks", "T": "tiles", "G": "groups", "N": "queries"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "MCTGN"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {axis: None for axis in "MCTGN"}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset("MCTGN")
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"chunks", "tiles", "groups", "queries"})
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, chunks: int, tiles: int, groups: int, queries: int, op: str, negate: bool = False) -> None:
        """Configure the flattened reduction extent and combinator."""
        if op not in _REDUCTIONS:
            raise ValueError(f"unsupported tiled reduction {op!r}")
        kwargs: dict[str, object] = {
            "chunks": chunks,
            "tiles": tiles,
            "groups": groups,
            "queries": queries,
            "op": "maximum" if op == "max" else op,
            "axis": 3,
        }
        if negate:
            kwargs["negate"] = True
        super().__init__(**kwargs)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one SBUF source."""
        if (role := _operand_role(kwargs["data"])) is not None and role != "sbuf":
            raise TypeError(f"NKITiledTensorReduce(data=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the tiled reduction for CPU validation."""
        c, t, g, n = (int(kwargs[key]) for key in ("chunks", "tiles", "groups", "queries"))
        data = np.asarray(kwargs["data"]).reshape(-1, c, t, g, n).transpose(0, 3, 4, 1, 2)
        result = np.asarray(_REDUCTIONS[str(kwargs["op"])](data, axis=(3, 4)))
        return (-result if kwargs.get("negate", False) else result).reshape(data.shape[0], g * n)


__all__ = ["NKITiledTensorReduce"]
