"""Unsigned 16-bit index generation with native ``iota``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.ops.base import NKIOp


class NKIUInt16Iota(NKIOp):
    """Generate an affine pattern directly into a uint16 SBUF tile."""

    NAME: ClassVar[str] = "iota"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset()
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"P": "partitions", "F": "width"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"partitions", "width"})
    OUTPUT_DTYPE: ClassVar[str | None] = "uint16"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "uint16"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Evaluate integer arithmetic before the native unsigned conversion."""
        pattern = tuple((int(step), int(size)) for step, size in kwargs["pattern"])
        width, partitions = int(kwargs["width"]), int(kwargs["partitions"])
        if int(np.prod([size for _step, size in pattern])) != width:
            raise ValueError(f"iota pattern {pattern} does not contain {width} free-axis elements")
        grid = np.indices(tuple(size for _step, size in pattern), dtype=np.int64)
        free = np.sum([step * grid[axis] for axis, (step, _size) in enumerate(pattern)], axis=0).reshape(1, width)
        channels = np.arange(partitions, dtype=np.int64)[:, None] * int(kwargs.get("channel_multiplier", 0))
        return np.asarray(free + channels + int(kwargs.get("offset", 0)), dtype=np.uint16)


def emit_indices(emit: TorchArithmetic, width: int) -> str:
    """Generate original positions in the narrowest exact unsigned storage."""
    operation = "NKIUInt16Iota" if width <= 65536 else "NKIIndexIota"
    groups = "" if width <= 65536 else "groups=1, "
    return emit.emit(
        operation, "", f"{groups}partitions=1, width={width}, pattern=[[1, {width}]], channel_multiplier=0"
    )


__all__ = ["NKIUInt16Iota"]
