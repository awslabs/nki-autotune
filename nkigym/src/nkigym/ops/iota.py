"""Constant pattern generation with ``nisa.iota``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp


class NKIIota(NKIOp):
    """Generate one affine integer pattern in an SBUF tile."""

    NAME: ClassVar[str] = "iota"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset()
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"P": "partitions", "F": "width"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"partitions", "width"})
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Generate the configured affine pattern for CPU validation."""
        pattern = tuple((int(step), int(size)) for step, size in kwargs["pattern"])
        width, partitions = int(kwargs["width"]), int(kwargs["partitions"])
        if int(np.prod([size for _step, size in pattern])) != width:
            raise ValueError(f"iota pattern {pattern} does not contain {width} free-axis elements")
        grid = np.indices(tuple(size for _step, size in pattern), dtype=np.int64)
        free = np.sum([step * grid[axis] for axis, (step, _size) in enumerate(pattern)], axis=0).reshape(1, width)
        channels = np.arange(partitions, dtype=np.int64)[:, None] * int(kwargs.get("channel_multiplier", 0))
        return np.asarray(free + channels + int(kwargs.get("offset", 0)), dtype=np.float32)


__all__ = ["NKIIota"]
