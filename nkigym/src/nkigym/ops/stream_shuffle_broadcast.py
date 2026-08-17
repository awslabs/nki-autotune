"""Partition broadcast with ``nisa.nc_stream_shuffle``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIStreamShuffleBroadcast(NKIOp):
    """Broadcast one SBUF partition to at most one 32-partition quadrant."""

    NAME: ClassVar[str] = "nc_stream_shuffle"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("S", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"S": 1, "P": "partitions"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"S": 1, "P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"S": 1, "P": 32, "F": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"partitions"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, partitions: int) -> None:
        """Configure the destination partition count and broadcast mask."""
        if not 1 <= partitions <= 32:
            raise ValueError("NKIStreamShuffleBroadcast partitions must be between 1 and 32")
        super().__init__(partitions=partitions, shuffle_mask=[0] * 32)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require exactly one SBUF source partition."""
        if (role := _operand_role(kwargs["src"])) is not None and role != "sbuf":
            raise TypeError(f"NKIStreamShuffleBroadcast(src=<role={role}>) expects SBUF")
        if np.asarray(kwargs["src"]).shape[0] != 1:
            raise ValueError("NKIStreamShuffleBroadcast requires one source partition")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Replicate the source row across the configured partitions."""
        return np.tile(np.asarray(kwargs["src"]), (int(kwargs["partitions"]), 1))


__all__ = ["NKIStreamShuffleBroadcast"]
