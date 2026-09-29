"""Reduce uniform chunk partials within each grouped row."""

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchSegments, TorchValue
from nkigym.ops.base import AxisRole, NKIOp, ReductionContract, _operand_role, reduction_combinator
from nkigym.ops.grouped_free_gather import emit_pairwise_maximum, folded_max_layout

_REDUCTIONS = {"add": np.sum, "max": np.max}


class NKIGroupedTileReduce(NKIOp):
    """Reduce the chunk axis while retaining grouped row order."""

    NAME: ClassVar[str] = "tensor_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "G", "T"), "dst": ("P", "G")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "data": (("P",), ("G", "T")),
        "dst": (("P",), ("G",)),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "data": (("P",), ("G",), ("T",)),
        "dst": (("P",), ("G",)),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions", "T": "chunks"}
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"T": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "GPT"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": None, "P": 128, "T": None}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"T"})
    TILABLE_FIXED_AXES: ClassVar[frozenset[str]] = frozenset({"G"})
    RFACTOR_RECIPE: ClassVar[Literal["rmw", "slot"] | None] = "slot"
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions", "chunks"})
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, partitions: int, chunks: int, op: str) -> None:
        """Configure one grouped chunk reduction."""
        if op not in _REDUCTIONS:
            raise ValueError(f"unsupported grouped tile reduction {op!r}")
        super().__init__(
            groups=groups, partitions=partitions, chunks=chunks, op="maximum" if op == "max" else op, axis=2
        )

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> ReductionContract:
        """Return the configured chunk reduction contract."""
        return ReductionContract(
            input_operand="data",
            output_operand="dst",
            reduction_axis="T",
            combinator=reduction_combinator(str(kwargs["op"])),
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one SBUF source."""
        if (role := _operand_role(kwargs["data"])) is not None and role != "sbuf":
            raise TypeError(f"NKIGroupedTileReduce(data=<role={role}>) expects SBUF")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return one reduced scalar per grouped row."""
        groups, partitions, chunks = (int(kwargs[name]) for name in ("groups", "partitions", "chunks"))
        data = np.asarray(kwargs["data"]).reshape(partitions, groups, chunks)
        operation = "max" if kwargs["op"] == "maximum" else str(kwargs["op"])
        return np.asarray(_REDUCTIONS[operation](data, axis=2), dtype=np.float32)


def emit_transposed_maxima(
    emit: TorchArithmetic, values: tuple[TorchValue, TorchValue], shape: tuple[int, int], chunk: int
) -> tuple[TorchSegments, TorchSegments]:
    """Merge consecutive partial maxima, preserving the first NaN or first maximum."""
    rows, parts = shape
    groups = 2 if rows * parts > 128 else 1
    if rows <= 1 or parts <= 1 or not 0 < chunk * parts <= 1 << 24:
        raise ValueError("grouped maxima require bounded consecutive partial rows")
    tables = []
    for value in values:
        source = value.name if value.storage_dtype == "float32" else emit.cast("NKIFloat32Cast", value.name)
        tables.append(emit.emit("NKIDMATranspose", f"src={source}"))
    if parts == 2:
        return emit_pairwise_maximum(emit, tables, rows, chunk, groups)
    config = f"partitions=1, groups={rows}, chunks={parts}"
    missing = emit.binary("not_equal", tables[0], tables[0])
    nan_groups = emit.emit("NKIGroupedTileReduce", f"data={missing}", f"{config}, op='max'")
    maximum = emit.emit("NKIGroupedTileReduce", f"data={tables[0]}", f"{config}, op='max'")
    matched = emit.emit("NKIGroupedTensorBroadcast", f"data1={tables[0]}, data2={maximum}", f"{config}, op='equal'")
    valid = emit.inverse(nan_groups)
    matched = emit.emit("NKIGroupedTensorBroadcast", f"data1={matched}, data2={valid}", f"{config}, op='multiply'")
    chosen = emit.binary("maximum", matched, missing)
    negative = emit.emit(
        "NKIIota",
        "",
        f"partitions=1, width={rows * parts}, pattern=[[0, {rows}], [-1, {parts}]], "
        f"channel_multiplier=0, offset={-parts}",
    )
    scores = emit.binary("add", emit.scalar("multiply", chosen, float(parts)), negative)
    top = emit.emit("NKIGroupedTileReduce", f"data={scores}", f"{config}, op='max'")
    offsets = emit.emit(
        "NKIIota",
        "",
        f"partitions=1, width={rows}, pattern=[[0, {groups}], [{parts}, {rows // groups}]], channel_multiplier=0",
    )
    positions = emit.cast("NKIUInt32Cast", emit.binary("subtract", offsets, top))
    selected = [
        emit.emit("NKIGroupedFreeGather", f"data={table}, indices={positions}", f"groups={groups}") for table in tables
    ]
    absolute = emit.binary("add", selected[1], emit.scalar("multiply", top, float(-chunk)))
    index = emit.cast("NKIUInt32Cast", absolute)
    return (
        TorchSegments((TorchValue(selected[0], (rows, 1), transposed=True, storage_dtype="float32"),)),
        TorchSegments((TorchValue(index, (rows, 1), transposed=True, storage_dtype="uint32"),)),
    )


__all__ = ["NKIGroupedTileReduce"]
