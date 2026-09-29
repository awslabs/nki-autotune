"""Find top-value indices with ``nisa.nc_find_index8``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchSegments, TorchValue
from nkigym.ops.base import AxisRole, NKIOp, _operand_role
from nkigym.ops.broadcast_find_index8 import emit_grouped_maxima, emit_indexed_max
from nkigym.ops.grouped_tile_reduce import emit_transposed_maxima, folded_max_layout
from nkigym.ops.register_load import ControlEmitter
from nkigym.ops.transpose import emit_partition_sum


class NKIFindIndex8(NKIOp):
    """Return first-occurrence indices for eight values per partition."""

    NAME: ClassVar[str] = "nc_find_index8"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "vals": ("P", "K"), "dst": ("P", "K")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "vals"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"K": 8}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"F"})
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.SEQUENTIAL}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 8, "K": 8}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": 16384, "K": 8}
    OUTPUT_DTYPE: ClassVar[str | None] = "uint32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "uint32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require both inputs to reside on chip."""
        for slot in ("data", "vals"):
            if (role := _operand_role(kwargs[slot])) is not None and role != "sbuf":
                raise TypeError(f"NKIFindIndex8({slot}=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Assign unused matches in query order, leaving exhausted queries unmatched."""
        data, values = np.asarray(kwargs["data"]), np.asarray(kwargs["vals"])
        result = np.full(values.shape, np.iinfo(np.uint32).max, dtype=np.uint32)
        for row in range(data.shape[0]):
            used = np.zeros(data.shape[1], dtype=np.bool_)
            for column in range(values.shape[1]):
                matches = np.flatnonzero(data[row] == values[row, column])
                if not matches.size:
                    raise ValueError("NKIFindIndex8 value is absent from its data row")
                available = matches[~used[matches]]
                if available.size:
                    result[row, column] = available[0]
                    used[available[0]] = True
        return result


def emit_small_prefix(
    emit: ControlEmitter, source: str, width: int, count: int, partitions: int, per_partition: bool
) -> tuple[str, str, str]:
    """Select a sanitized prefix and accept only finite, strictly ordered source rows."""
    finite = emit.emit(
        "NKIUInt16ScalarSequence",
        f"data={source}, operand0=0.0, operand1=0.0",
        "op0='multiply', op1='equal', engine='vector'",
    )
    totals = emit.emit("NKITensorReduce", f"data={finite}", "op='add', axis=1")
    valid = emit.scalar("equal", totals, float(width))
    working = emit.emit("NKISelectReduce", f"on_true={source}, predicate={finite}", "on_false=0.0")
    values = emit.emit("NKIMax8", f"src={working}")
    indices = emit.emit("NKIFindIndex8", f"data={working}, vals={values}")
    selected = emit.slice(values, 0, count)
    strict = emit.binary("greater", selected, emit.slice(values, 1, count))
    scanned = emit.emit("NKITensorScalarCumulative", f"src={strict}", "op0='add', op1='add', imm0=0.0")
    strict_count = emit.slice(scanned, count - 1, 1)
    guard = emit.scalar("multiply", emit.scalar("equal", strict_count, float(count)), valid)
    if partitions > 1 and not per_partition:
        guard = emit.scalar("equal", emit_partition_sum(emit, guard), float(partitions))
    return selected, emit.slice(indices, 0, count), guard


def emit_max_with_indices(
    source: TorchValue, stem: str, body: list[str], imports: set[str]
) -> tuple[TorchSegments, TorchSegments]:
    """Load contiguous reduction groups directly, then combine their ordered maxima."""
    rows, width = source.shape
    if source.transposed or not 1 <= width <= 1 << 24:
        raise ValueError("max with indices requires exactly representable row positions")
    groups, parts = folded_max_layout(rows, width)
    if width <= 512 or parts == 1:
        if source.is_hbm:
            imports.add("NKILoad")
            loaded = f"sbuf_{source.name}"
            body.append(f"{loaded} = NKILoad()(src={source.name})")
            source = TorchValue(loaded, source.shape, storage_dtype=source.storage_dtype)
        return emit_indexed_max(source, stem, body, imports)
    chunk = width // parts
    emit = TorchArithmetic(f"sbuf_{stem}_chunks", body, imports)
    stored = source.name if source.is_hbm else emit.emit("NKIStore", f"src={source.name}")
    loaded = emit.emit("NKIGroupedLoad", f"src={stored}", f"groups={groups}, rows={rows // groups}, stages={parts}")
    local_values, local_indices = emit_indexed_max(
        TorchValue(loaded, (rows * parts, chunk), storage_dtype=source.storage_dtype), f"{stem}_local", body, imports
    )
    if rows > 1 and source.storage_dtype == "float32":
        return emit_transposed_maxima(emit, (local_values.values[0], local_indices.values[0]), (rows, parts), chunk)
    tables = emit_grouped_maxima(emit, (local_values.values[0], local_indices.values[0]), (rows, parts), chunk)
    values, ranks = emit_indexed_max(
        TorchValue(tables[0], (rows, parts), storage_dtype=source.storage_dtype), f"{stem}_final", body, imports
    )
    selected = emit.emit("NKINCGather", f"data={tables[1]}, indices={ranks.values[0].name}")
    rank = emit.cast("NKIFloat32Cast", ranks.values[0].name)
    offset = emit.scalar("multiply", rank, float(chunk))
    absolute = emit.binary("add", offset, emit.cast("NKIFloat32Cast", selected))
    indices = emit.cast("NKIUInt32Cast", absolute)
    return values, TorchSegments((TorchValue(indices, (rows, 1), storage_dtype="uint32"),))


__all__ = ["NKIFindIndex8"]
