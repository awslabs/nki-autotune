"""Packed tile sums through ``nisa.nc_matmul``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, _operand_role


class NKITiledSumMatmul(NKIOp):
    """Sum partition tiles into one column per chunk and tile."""

    NAME: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "stationary": ("K", "C", "T", "G", "N"),
        "moving": ("K", "O"),
        "dst": ("G", "N", "C", "T"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "stationary": (("K",), ("C", "T", "G", "N")),
        "moving": (("K",), ("O",)),
        "dst": (("G", "N"), ("C", "T")),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"stationary", "moving"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {
        "C": "chunks",
        "T": "tiles",
        "G": "groups",
        "N": "queries",
        "O": 1,
    }
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"K": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "CTGNO"}
    MIN_TILE_SIZE.update({"K": 128})
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"C": 1, "T": 1, "G": None, "N": None, "K": 128, "O": 1}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"chunks", "tiles", "groups", "queries"})
    OUTPUT_ROLE: ClassVar[str] = "psum"
    OUTPUT_LOCATION: ClassVar[str] = "psum"
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"

    def __init__(self, chunks: int, tiles: int, groups: int, queries: int) -> None:
        """Configure the packed reduction layout."""
        super().__init__(chunks=chunks, tiles=tiles, groups=groups, queries=queries, accumulate=False)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require both operands in SBUF."""
        for slot in ("stationary", "moving"):
            if (role := _operand_role(kwargs[slot])) is not None and role != "sbuf":
                raise TypeError(f"NKITiledSumMatmul({slot}=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return one partition sum per packed tile."""
        c, t, g, n = (int(kwargs[key]) for key in ("chunks", "tiles", "groups", "queries"))
        data = np.asarray(kwargs["stationary"], dtype=np.float32).reshape(-1, c, t, g, n)
        ones = np.asarray(kwargs["moving"], dtype=np.float32)
        output = np.empty((g * n, c * t), dtype=np.float32)
        for chunk in range(c):
            for tile in range(t):
                output[:, chunk * t + tile : chunk * t + tile + 1] = (
                    data[:, chunk, tile].reshape(data.shape[0], g * n).T @ ones
                )
        return output


__all__ = ["NKITiledSumMatmul"]
