"""Stable nonzero compaction with ``nisa.nonzero_with_count``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.nc_gather import emit_clamped_gather


class NKINonzeroWithCount(NKIOp):
    """Return stable nonzero indices, padding, and the final count."""

    NAME: ClassVar[str] = "nonzero_with_count"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "G", "F"), "dst": ("P", "G", "O")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("P",), ("G", "F")),
        "dst": (("P",), ("G", "O")),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"P": 1, "F": "input_width", "O": "output_width"}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"G"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "G": 1, "F": 1, "O": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 1, "G": 1, "F": None, "O": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"columns", "input_width", "output_width"})
    OUTPUT_DTYPE: ClassVar[str | None] = "int32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "int32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, columns: int, input_width: int, output_width: int, padding_val: int = -1) -> None:
        """Configure the extents and native int32 padding value."""
        super().__init__(
            columns=columns, input_width=input_width, output_width=output_width, index_offset=0, padding_val=padding_val
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require packed on-chip rows and exactly one count per row."""
        source = np.asarray(kwargs["src"])
        role = _operand_role(kwargs["src"])
        if role is not None and role != "sbuf":
            raise TypeError(f"NKINonzeroWithCount(src=<role={role}>) expects sbuf")
        columns = int(kwargs["columns"])
        input_width = int(kwargs["input_width"])
        if columns < 1 or source.shape != (1, columns * input_width) or int(kwargs["output_width"]) != input_width + 1:
            raise ValueError("NKINonzeroWithCount requires packed rows and output_width=input_width+1")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Compact each packed row and append its count."""
        columns = int(kwargs["columns"])
        input_width = int(kwargs["input_width"])
        output_width = int(kwargs["output_width"])
        source = np.asarray(kwargs["src"]).reshape(columns, input_width)
        output = np.full((columns, output_width), int(kwargs["padding_val"]), dtype=np.int32)
        for column, row in enumerate(source):
            indices = np.flatnonzero(row).astype(np.int32)
            output[column, : indices.size] = indices
            output[column, -1] = indices.size
        return output.reshape(1, columns * output_width)


def emit_compacted_pairs(
    emit: TorchArithmetic,
    masks: tuple[str, str],
    grids: list[str],
    starts: tuple[str | float, str | float],
    width: int,
    partitions: int = 1,
) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str], tuple[str, str], str]:
    """Pair ascending left candidates with descending right candidates."""

    tables, counts, prefixes = [], [], []
    for mask in masks:
        source = mask
        if partitions > 1:
            stored = emit.emit("NKIFlattenStore", f"src={mask}", f"width={partitions * width}")
            source = emit.emit("NKILoad", f"src={stored}")
        raw = emit.emit(
            "NKINonzeroWithCount",
            f"src={source}",
            f"columns={partitions}, input_width={width}, output_width={width + 1}, padding_val={width}",
        )
        if partitions > 1:
            stored = emit.emit("NKIReshapeStore", f"src={raw}", f"rows={partitions}")
            raw = emit.emit("NKILoad", f"src={stored}")
        tables.append(emit.cast("NKIFloat32Cast", emit.emit("NKITensorSlice", f"src={raw}", f"start=0, width={width}")))
        counts.append(emit.cast("NKIFloat32Cast", emit.emit("NKITensorSlice", f"src={raw}", f"start={width}, width=1")))
        prefixes.append(emit.emit("NKITensorScalarCumulative", f"src={mask}", "op0='add', op1='add', imm0=0.0"))
    ranks = emit.scalar("subtract", prefixes[0], 1.0), emit.scalar("subtract", prefixes[1], counts[1], True)
    reverse = emit.scalar("subtract", ranks[0], counts[1], True)
    partners = (
        emit.scalar(
            "add", emit_clamped_gather(emit, tables[1], emit.scalar("subtract", reverse, 1.0), width), starts[1]
        ),
        emit.scalar("add", emit_clamped_gather(emit, tables[0], ranks[1], width), starts[0]),
    )
    swaps = []
    for side in range(2):
        mask = emit.binary("multiply", masks[side], emit.scalar("less", ranks[side], counts[1 - side]))
        ordered = (
            emit.binary("greater", partners[0], grids[0])
            if side == 0
            else emit.binary("greater", grids[1], partners[1])
        )
        swaps.append(emit.binary("multiply", mask, ordered))
    prefix = emit.emit("NKITensorScalarCumulative", f"src={swaps[0]}", "op0='add', op1='add', imm0=0.0")
    count = emit.emit("NKITensorSlice", f"src={prefix}", f"start={width - 1}, width=1")
    return (tables[0], tables[1]), (counts[0], counts[1]), partners, (swaps[0], swaps[1]), count


__all__ = ["NKINonzeroWithCount"]
