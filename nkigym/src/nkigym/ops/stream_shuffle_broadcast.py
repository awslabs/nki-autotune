"""Partition broadcast with ``nisa.nc_stream_shuffle``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.nc_gather import emit_clamped_gather
from nkigym.ops.register_load import ControlEmitter
from nkigym.ops.transpose import emit_partition_row_sum

_POINTWISE_OPERATIONS = frozenset({"add", "maximum", "multiply", "subtract"})


def _supports_free_broadcast(matrix_shape: tuple[int, ...], vector_shape: tuple[int, ...], operation: str) -> bool:
    """Return whether one free-axis broadcast fits a stream-shuffle quadrant."""
    return len(vector_shape) == 2 and matrix_shape[0] <= 32 and operation in _POINTWISE_OPERATIONS


def _stream_shuffle_source(target: str, source: str, partitions: int) -> str:
    """Render one frontend stream-shuffle call."""
    return f"{target} = NKIStreamShuffleBroadcast(partitions={partitions})(src={source})"


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


def emit_local_sort_permutation(emit: ControlEmitter, source: str, width: int, span: int) -> str:
    """Compute insertion-equivalent ranks for values displaced by fewer than ``span`` positions."""
    if not 2 <= span <= min(16, width):
        raise ValueError("local rank sorting requires a span between 2 and 16 within the row width")

    def positions(partitions: int, step: int, offset: int) -> str:
        """Generate neighboring source positions across one shuffle quadrant."""
        return emit.emit(
            "NKIIota",
            "",
            f"partitions={partitions}, width={width}, pattern=[[1, {width}]], "
            f"channel_multiplier={step}, offset={offset}",
        )

    data = emit.emit("NKIStreamShuffleBroadcast", f"src={source}", f"partitions={span - 1}")
    right, left = positions(span - 1, 1, 1), positions(span - 1, -1, -1)
    later = emit_clamped_gather(emit, data, right, width)
    missing = emit.scalar("subtract", emit.binary("equal", later, later), 1.0, reverse=True)
    before = emit.binary(
        "maximum",
        emit.binary("greater", later, data),
        emit.binary("multiply", missing, emit.binary("equal", data, data)),
    )
    inverted = emit.binary("multiply", before, emit.scalar("less", right, float(width)))
    incoming = emit_clamped_gather(emit, inverted, left, width)
    delta = emit.binary(
        "subtract", inverted, emit.binary("multiply", incoming, emit.scalar("greater_equal", left, 0.0))
    )
    ranks = emit.binary("add", emit.iota(width), emit_partition_row_sum(emit, delta, width))
    partitions = 2 * span - 1
    replicated = emit.emit("NKIStreamShuffleBroadcast", f"src={ranks}", f"partitions={partitions}")
    target, candidate = positions(partitions, 0, 0), positions(partitions, 1, 1 - span)
    valid = emit.binary(
        "multiply", emit.scalar("greater_equal", candidate, 0.0), emit.scalar("less", candidate, float(width))
    )
    safe = emit.emit(
        "NKITensorScalarSequence",
        f"data={candidate}, operand0=0.0, operand1={float(width - 1)}",
        "op0='maximum', op1='minimum', engine='vector'",
    )
    matches = emit.binary("equal", emit_clamped_gather(emit, replicated, safe, width), target)
    weighted = emit.binary("multiply", emit.binary("multiply", matches, valid), safe)
    return emit_partition_row_sum(emit, weighted, width)


__all__ = ["NKIStreamShuffleBroadcast"]
