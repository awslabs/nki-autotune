"""Eager mode data types and constants.

Shared dataclasses used by the eager tracer and renderer.
"""

from dataclasses import dataclass
from typing import Any

from nkigym.ops.base import NKIOp

SBUF_PMAX = 128


@dataclass
class TracedOp:
    """A recorded op call from math function tracing.

    Attributes:
        op_idx: Sequential index in the math function.
        op: The NKIOp instance.
        output_names: Names assigned to outputs.
        output_shapes: Shapes of output arrays (from numpy simulation).
        operand_names: Maps operand slot name to traced tensor name.
        operand_shapes: Maps operand slot name to numpy shape.
        config_kwargs: Non-tensor keyword arguments.
    """

    op_idx: int
    op: NKIOp
    output_names: list[str]
    output_shapes: list[tuple[int, ...]]
    operand_names: dict[str, str]
    operand_shapes: dict[str, tuple[int, ...]]
    config_kwargs: dict[str, Any]


@dataclass
class DimInfo:
    """Dimension metadata after unification.

    Attributes:
        dim_id: Unique dimension identifier (e.g. ``"d0"``).
        total_size: Total number of elements in this dimension.
        tile_size: Tile size for this dimension.
        num_blocks: Number of blocks (total_size / tile_size).
        tiles_per_block: Always 1 in eager mode.
    """

    dim_id: str
    total_size: int
    tile_size: int
    num_blocks: int
    tiles_per_block: int


@dataclass
class TensorInfo:
    """Tensor metadata from tracing.

    Attributes:
        name: Tensor name in the math function.
        dims: Ordered dimension IDs for this tensor.
        shape_2d: Original 2D shape from numpy.
        is_input: Whether this is a kernel input parameter.
        producer_op: Index of the op that produces this tensor.
    """

    name: str
    dims: tuple[str, ...]
    shape_2d: tuple[int, ...]
    is_input: bool
    producer_op: int
