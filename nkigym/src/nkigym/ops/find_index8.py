"""Find top-value indices with ``nisa.nc_find_index8``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchSegments, TorchValue
from nkigym.ops.base import AxisRole, NKIOp, _operand_role


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
            role = _operand_role(kwargs[slot])
            if role is not None and role != "sbuf":
                raise TypeError(f"NKIFindIndex8({slot}=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Find the first matching index of each requested value."""
        data, values = np.asarray(kwargs["data"]), np.asarray(kwargs["vals"])
        result = np.empty(values.shape, dtype=np.uint32)
        for row in range(data.shape[0]):
            used = np.zeros(data.shape[1], dtype=np.bool_)
            for column in range(values.shape[1] - 1, -1, -1):
                matches = np.flatnonzero((data[row] == values[row, column]) & ~used)
                if not matches.size:
                    raise ValueError("NKIFindIndex8 value is absent from its data row")
                result[row, column] = matches[0]
                used[matches[0]] = True
        return result


def _emit_max_with_indices(
    source: TorchValue, stem: str, body: list[str], imports: set[str]
) -> tuple[TorchSegments, TorchSegments]:
    """Return the first maximum index, giving NaNs precedence and preserving the selected value."""
    rows, width = source.shape
    emit = TorchArithmetic(f"sbuf_{stem}", body, imports)
    data = source.name if source.storage_dtype == "float32" else emit.cast("NKIFloat32Cast", source.name)
    finite = emit.binary("equal", data, data)
    missing = emit.scalar("subtract", finite, 1.0, reverse=True)
    nan_row = emit.emit("NKITensorReduce", f"data={missing}", "op='max', axis=1")
    maximum = emit.emit("NKITensorReduce", f"data={data}", "op='max', axis=1")
    matches = emit.scalar("equal", data, maximum)
    present = emit.scalar("subtract", nan_row, 1.0, reverse=True)
    selected = emit.binary("maximum", emit.scalar("multiply", matches, present), missing)
    positions = emit.emit(
        "NKIIota", "", f"partitions={rows}, width={width}, pattern=[[1, {width}]], channel_multiplier=0"
    )
    penalty = emit.emit(
        "NKITensorScalarSequence",
        f"data={selected}, operand0=1.0, operand1={float(width)}",
        "op0='subtract', op1='multiply', engine='vector'",
    )
    negative = emit.binary("subtract", penalty, positions)
    best = emit.emit("NKITensorReduce", f"data={negative}", "op='max', axis=1")
    zero = emit.emit("NKIIota", "", f"partitions={rows}, width=1, pattern=[[0, 1]], channel_multiplier=0")
    rank = emit.emit(
        "NKITensorScalarSequence",
        f"data={zero}, operand0={best}, operand1=-1.0",
        "op0='add', op1='multiply', engine='vector'",
    )
    indices = emit.cast("NKIUInt32Cast", rank)
    values = emit.emit("NKINCGather", f"data={source.name}, indices={indices}")
    return (
        TorchSegments((TorchValue(values, (rows, 1), storage_dtype=source.storage_dtype),)),
        TorchSegments((TorchValue(indices, (rows, 1), storage_dtype="uint32"),)),
    )


def emit_max_with_indices(
    source: TorchValue, stem: str, body: list[str], imports: set[str]
) -> tuple[TorchSegments, TorchSegments]:
    """Reduce contiguous chunks in parallel, then reduce their ordered maxima."""
    rows, width = source.shape
    if source.transposed or source.is_hbm or not 1 <= width <= 1 << 24:
        raise ValueError("max with indices requires a materialized row with exactly representable positions")
    parts = next((size for size in range(128 // rows, 1, -1) if width % size == 0), 1)
    if width <= 512 or parts == 1 or (parts == 2 and width < 8192):
        return _emit_max_with_indices(source, stem, body, imports)
    chunk = width // parts
    emit = TorchArithmetic(f"sbuf_{stem}_chunks", body, imports)
    stored = emit.emit("NKIReshapeStore", f"src={source.name}", f"rows={parts}")
    loaded = emit.emit("NKILoad", f"src={stored}")
    local_values, local_indices = _emit_max_with_indices(
        TorchValue(loaded, (rows * parts, chunk), storage_dtype=source.storage_dtype), f"{stem}_local", body, imports
    )
    tables = []
    for value in (local_values.values[0], local_indices.values[0]):
        transposed = emit.emit("NKIDMATranspose", f"src={value.name}")
        shaped = emit.emit("NKIReshapeStore", f"src={transposed}", f"rows={rows}")
        tables.append(emit.emit("NKILoad", f"src={shaped}"))
    values, ranks = _emit_max_with_indices(
        TorchValue(tables[0], (rows, parts), storage_dtype=source.storage_dtype), f"{stem}_final", body, imports
    )
    selected = emit.emit("NKINCGather", f"data={tables[1]}, indices={ranks.values[0].name}")
    rank = emit.cast("NKIFloat32Cast", ranks.values[0].name)
    offset = emit.scalar("multiply", rank, float(chunk))
    absolute = emit.binary("add", offset, emit.cast("NKIFloat32Cast", selected))
    indices = emit.cast("NKIUInt32Cast", absolute)
    return values, TorchSegments((TorchValue(indices, (rows, 1), storage_dtype="uint32"),))


__all__ = ["NKIFindIndex8"]
