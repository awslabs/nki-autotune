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


def uniform_order(width: int, count: int, sorted_output: bool, full_sort: bool) -> tuple[int, ...]:
    """Specialize the existing sort/selection algorithm to an always-false comparator."""
    if not 1 <= count <= width:
        raise ValueError("selection count must be inside the input width")
    indices = list(range(width))

    def partition(first: int, last: int) -> int:
        """Apply median-of-three selection and symmetric equal-key partitioning."""
        middle = (first + last) // 2
        indices[first], indices[middle] = indices[middle], indices[first]
        left, right = first + 1, last - 1
        while left < right:
            indices[left], indices[right] = indices[right], indices[left]
            left, right = left + 1, right - 1
        return left

    def sort(first: int, last: int) -> None:
        """Apply introsort partitions; equal-key insertion cleanup leaves their order."""
        pending = [(first, last)]
        while pending:
            first, last = pending.pop()
            if last - first > 16:
                split = partition(first, last)
                pending.extend(((first, split), (split, last)))

    def adjust(hole: int, length: int, value: int) -> None:
        """Repair an equal-key heap by following its right child when available."""
        child = 2 * hole + 2
        while child < length:
            indices[hole] = indices[child]
            hole = child
            child = 2 * hole + 2
        if child - 1 < length:
            indices[hole] = indices[child - 1]
            hole = child - 1
        indices[hole] = value

    if full_sort:
        sort(0, width)
    elif count * 64 <= width:
        for parent in range(count // 2 - 1, -1, -1):
            adjust(parent, count, indices[parent])
        for last in range(count - 1, 0, -1):
            saved, indices[last] = indices[last], indices[0]
            adjust(0, last, saved)
    elif width > 3:
        first, last = 0, width
        while last - first > 3:
            split = partition(first, last)
            if split < count:
                first = split
            else:
                last = split
        if sorted_output:
            sort(0, count - 1)
    return tuple(indices[:count])


__all__ = ["NKIUInt16Iota"]
