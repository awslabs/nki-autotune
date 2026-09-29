"""Constant pattern generation with ``nisa.iota``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchValue
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


def emit_first_match_scores(emit: TorchArithmetic, selected: str, rows: int, width: int) -> str:
    """Rank selected positions ahead of unselected ones with exact FP32 integers."""
    if width <= 1 << 23:
        positions = emit.emit(
            "NKIIota",
            "",
            f"partitions={rows}, width={width}, pattern=[[-1, {width}]], channel_multiplier=0, offset={-width}",
        )
        scores = emit.emit(
            "NKIScalarTensorTensor",
            f"data={selected}, operand0={float(width)}, operand1={positions}",
            "op0='multiply', op1='add'",
        )
    else:
        positions = emit.emit(
            "NKIIota", "", f"partitions={rows}, width={width}, pattern=[[1, {width}]], channel_multiplier=0"
        )
        penalty = emit.emit(
            "NKITensorScalarSequence",
            f"data={selected}, operand0=1.0, operand1={float(width)}",
            "op0='subtract', op1='multiply', engine='vector'",
        )
        scores = emit.binary("subtract", penalty, positions)
    return scores


def emit_first_sorted_value(
    emit: TorchArithmetic, source: str, shape: tuple[int, int]
) -> tuple[TorchValue, TorchValue]:
    """Select the first stable descending element with the requested NaN order."""
    rows, width = shape
    valid = emit.binary("equal", source, source)
    zeros = emit.emit("NKIIota", "", f"partitions={rows}, width={width}, pattern=[[0, {width}]], channel_multiplier=0")
    floor = emit.scalar("add", zeros, "float('-inf')")
    clean = emit.select(valid, source, floor)
    maximum = emit.emit("NKITensorReduce", f"data={clean}", "op='max', axis=1")
    matches, missing = emit.scalar("equal", clean, maximum), emit.inverse(valid)
    if emit.nan_first:
        any_missing = emit.emit("NKITensorReduce", f"data={missing}", "op='max', axis=1")
        selected = emit.binary("maximum", emit.scalar("multiply", matches, emit.inverse(any_missing)), missing)
    else:
        all_missing = emit.emit("NKITensorReduce", f"data={missing}", "op='minimum', axis=1")
        selected = emit.scalar("maximum", emit.binary("multiply", matches, valid), all_missing)
    scores = emit_first_match_scores(emit, selected, rows, width)
    first = emit.emit("NKITensorReduce", f"data={scores}", "op='max', axis=1")
    indices = emit.cast("NKIUInt32Cast", emit.scalar("subtract", emit.slice(zeros, 0, 1), first))
    values = emit.emit("NKINCGather", f"data={source}, indices={indices}")
    return TorchValue(values, (rows, 1), storage_dtype="float32"), TorchValue(
        indices, (rows, 1), storage_dtype="uint32"
    )


__all__ = ["NKIIota"]
