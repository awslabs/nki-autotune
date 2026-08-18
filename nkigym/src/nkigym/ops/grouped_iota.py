"""Grouped affine value generation through ``nisa.iota``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp


class NKIGroupedIota(NKIOp):
    """Generate one floating affine pattern across packed groups."""

    NAME: ClassVar[str] = "iota"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"dst": ("G", "P", "F")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {"dst": (("G", "P"), ("F",))}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset()
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions", "F": "width"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "F": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions", "width"})
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Generate the configured pattern for every packed partition."""
        pattern = tuple((int(step), int(size)) for step, size in kwargs["pattern"])
        grid = np.indices(tuple(size for _step, size in pattern), dtype=np.int64)
        values = np.sum([step * grid[axis] for axis, (step, _size) in enumerate(pattern)], axis=0)
        channels = int(kwargs["groups"]) * int(kwargs["partitions"])
        offsets = np.arange(channels, dtype=np.int64)[:, None] * int(kwargs.get("channel_multiplier", 0))
        return np.asarray(values.reshape(1, int(kwargs["width"])) + offsets, dtype=np.float32)


__all__ = ["NKIGroupedIota"]
