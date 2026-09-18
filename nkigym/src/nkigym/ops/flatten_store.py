"""Flatten one SBUF tile into a single-row HBM buffer with ``nisa.dma_copy``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.nc_gather import emit_clamped_gather
from nkigym.ops.nonzero_with_count import emit_compacted_pairs
from nkigym.ops.register_load import ControlEmitter
from nkigym.ops.tensor_scalar import emit_scan_stop


class NKIFlattenStore(NKIOp):
    """Store a two-dimensional SBUF tile as one contiguous HBM row."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("R", "O")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"R": 1, "O": "width"}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"P", "F"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1, "R": 1, "O": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None, "R": 1, "O": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"width"})
    OUTPUT_ROLE: ClassVar[str] = "shared_hbm"
    OUTPUT_LOCATION: ClassVar[str] = "shared_hbm"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require SBUF input with exactly the configured element count."""
        if (role := _operand_role(kwargs["src"])) is not None and role != "sbuf":
            raise TypeError(f"NKIFlattenStore(src=<role={role}>) expects SBUF")
        if np.asarray(kwargs["src"]).size != int(kwargs["width"]):
            raise ValueError("NKIFlattenStore width must equal the source element count")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the source flattened into one HBM row."""
        return np.asarray(kwargs["src"]).reshape(1, int(kwargs["width"])).copy()


def emit_parallel_partition(
    emit: ControlEmitter, sources: tuple[str, str], bounds: tuple[str, str], shape: tuple[int, int]
) -> tuple[tuple[str, str], str, str]:
    """Apply identical median-pivot partitions to independent SBUF rows."""
    values, indices = sources
    first, last = bounds
    partitions, width = shape
    grid = emit.emit(
        "NKIIota", "", f"partitions={partitions}, width={width}, pattern=[[1, {width}]], channel_multiplier=0"
    )
    left = emit.scalar("add", first, 1.0)
    middle = emit.binary("add", first, emit.half(emit.binary("subtract", last, first)))
    right = emit.scalar("subtract", last, 1.0)
    a, b, c = (emit_clamped_gather(emit, values, position, width) for position in (left, middle, right))
    ab, ac, bc = emit.before(a, b), emit.before(a, c), emit.before(b, c)
    chosen = emit.select(
        ab, emit.select(bc, middle, emit.select(ac, right, left)), emit.select(ac, left, emit.select(bc, right, middle))
    )
    first_grid = emit.scalar("add", emit.scalar("multiply", grid, 0.0), first)
    chosen_grid = emit.scalar("add", emit.scalar("multiply", grid, 0.0), chosen)
    mapping = emit.select(
        emit.scalar("equal", grid, first),
        chosen_grid,
        emit.select(emit.scalar("equal", grid, chosen), first_grid, grid),
    )
    positions = emit.cast("NKIUInt32Cast", emit.clamp(mapping, 0.0, float(width - 1)))
    values, indices = (emit.emit("NKINCGather", f"data={source}, indices={positions}") for source in (values, indices))
    pivot = emit_clamped_gather(emit, values, first, width)
    masks = tuple(emit_scan_stop(emit, values, grid, (left, last), pivot, side) for side in (False, True))
    tables, counts, partners, swaps, swapped = emit_compacted_pairs(
        emit, (masks[0], masks[1]), [grid, grid], (0.0, 0.0), width, partitions
    )
    mapping = emit.select(swaps[1], partners[1], emit.select(swaps[0], partners[0], grid))
    positions = emit.cast("NKIUInt32Cast", emit.clamp(mapping, 0.0, float(width - 1)))
    outputs = tuple(emit.emit("NKINCGather", f"data={source}, indices={positions}") for source in (values, indices))
    next_left = emit.scalar("minimum", emit_clamped_gather(emit, tables[0], swapped, width), last)
    previous = emit_clamped_gather(emit, tables[1], emit.binary("subtract", counts[1], swapped), width)
    cut = emit.select(emit.scalar("greater", swapped, 0.0), emit.binary("minimum", next_left, previous), next_left)
    inside = emit.binary("multiply", emit.scalar("greater_equal", grid, first), emit.scalar("less", grid, last))
    return (outputs[0], outputs[1]), cut, inside


__all__ = ["NKIFlattenStore"]
