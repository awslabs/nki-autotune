"""In-place SBUF insertion through ``nisa.tensor_copy``."""

from collections.abc import Callable
from typing import Any, ClassVar, Literal

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.dynamic_slice_copy import emit_compact_intervals
from nkigym.ops.register_load import ControlEmitter
from nkigym.ops.transpose import emit_merge_disjoint_rows


class NKIInplaceTensorCopy(NKIOp):
    """Copy one packed source tile into an existing destination interval."""

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("G", "P", "O"), "dst": ("G", "P", "F")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("G", "P"), ("O",)),
        "dst": (("G", "P"), ("F",)),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset({"dst"})
    RETURN_RMW_OPERAND: ClassVar[str | None] = "dst"
    SYNTHESIZE_RMW_INITIALIZER: ClassVar[bool] = False
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions", "O": "width"}
    INPUT_SLICES: ClassVar[dict[str, tuple[tuple[int, str, str], ...]]] = {"dst": ((1, "start", "width"),)}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions", "start", "width"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "P": 1, "O": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "O": None, "F": None}

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an on-chip source and an SBUF destination."""
        if _operand_role(kwargs["src"]) not in {None, "sbuf", "psum"}:
            raise TypeError("NKIInplaceTensorCopy.src expects SBUF or PSUM")
        if _operand_role(kwargs["dst"]) not in {None, "sbuf"}:
            raise TypeError("NKIInplaceTensorCopy.dst expects SBUF")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Insert the source into the selected free-axis interval."""
        start, width = int(kwargs["start"]), int(kwargs["width"])
        result = kwargs["dst"]
        result[:, start : start + width] = np.asarray(kwargs["src"])
        return result


def emit_parallel_sort_partitions(
    emit: ControlEmitter,
    sources: tuple[str, str],
    shape: tuple[int, int],
    partition: Callable[[tuple[str, str], tuple[str, str]], tuple[tuple[str, str], str, str]],
    index_cast: Literal["NKIUInt16Cast", "NKIUInt32Cast"],
) -> tuple[tuple[str, str], str]:
    """Process disjoint introsort intervals by depth, retaining unfinished heap ranges.

    Every active interval contains at least seventeen elements, so
    ``width // 17`` rows suffice. Processing independent intervals together
    preserves their median pivots and the serial algorithm's depth limit.
    """
    partitions, width = shape
    zero = emit.iota(1)
    one = emit.scalar("add", zero, 1.0)
    lows = emit.scalar("multiply", emit.iota(partitions), 0.0)
    highs = emit.copy(lows)
    emit.copy_windows((highs,), (emit.scalar("add", zero, float(width)),), emit.cast("NKIUInt32Cast", zero))
    remaining = emit.copy(one)
    with emit.repeat(2 * (width.bit_length() - 1)):
        with emit.guard(emit.scalar("greater", remaining, 0.0)):
            first, last = (emit.emit("NKIDMATranspose", f"src={source}") for source in (lows, highs))
            values, indices = (
                emit.emit("NKIStreamShuffleBroadcast", f"src={source}", f"partitions={partitions}")
                for source in sources
            )
            selected, cut, inside = partition((values, indices), (first, last))
            emit_merge_disjoint_rows(emit, sources, selected, inside, shape, index_cast)
            split = emit.emit("NKIDMATranspose", f"src={cut}")
            next_bounds, count = emit_compact_intervals(emit, (lows, highs), split, partitions)
            emit.copy_windows((lows, highs), next_bounds, emit.cast("NKIUInt32Cast", zero))
            emit.write(remaining, count, one)
    return (lows, highs), remaining


__all__ = ["NKIInplaceTensorCopy"]
