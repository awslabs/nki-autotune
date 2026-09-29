"""Store contiguous SBUF row segments as separate HBM rows."""

from contextlib import nullcontext
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.index_iota import native_prefix
from nkigym.ops.register_load import ControlEmitter


class NKIReshapeStore(NKIOp):
    """Store each source row as equally sized contiguous destination rows."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "G", "F"), "dst": ("P", "G", "F")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("P",), ("G", "F")),
        "dst": (("P", "G"), ("F",)),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("P",), ("F",)) for slot in OPERAND_AXES
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"})}
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "rows"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "G": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "G": 1, "F": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"rows"})
    OUTPUT_ROLE: ClassVar[str] = "stored"
    OUTPUT_LOCATION: ClassVar[str] = "shared_hbm"

    def __init__(self, rows: int) -> None:
        """Configure the number of destination rows per source row."""
        if rows < 1:
            raise ValueError("row reshape requires a positive row count")
        super().__init__(rows=rows)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an on-chip matrix with equally divisible source rows."""
        source = np.asarray(kwargs["src"])
        if _operand_role(kwargs["src"]) not in {None, "sbuf"} or source.ndim != 2:
            raise TypeError("NKIReshapeStore requires a rank-two SBUF source")
        if source.shape[1] % int(kwargs["rows"]):
            raise ValueError("row reshape must divide the source width exactly")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Copy source elements into the requested row-major shape."""
        source = np.asarray(kwargs["src"])
        rows = int(kwargs["rows"])
        return source.reshape(source.shape[0] * rows, source.shape[1] // rows).copy()


def emit_partitioned_prefix(
    emit: ControlEmitter, source: str, width: int, count: int, rows: int = 1, source_hbm: bool = False
) -> tuple[str, str, str] | None:
    """Select bounded local prefixes and prove they contain the complete global prefix."""
    if source_hbm and (rows < 2 or width < 2048 or count < 8):
        return None
    first = max(2, (width + 511) // 512)
    first = min(first, 128 // rows) if rows > 1 else first
    parts = next((size for size in range(first, min(128 // rows, width) + 1) if width % size == 0), 1)
    chunk = width // parts
    local_count = (count // 8 + 1) * 8 if rows > 1 else max(8, ((4 * count + 8 * parts - 1) // (8 * parts)) * 8)
    if local_count + 8 > chunk:
        local_count = max(8, ((3 * count + 16 * parts - 1) // (16 * parts)) * 8)
    eligible = count >= (16 if rows == 1 else 8) and parts > 1 and local_count + 8 <= chunk
    if not eligible or not count + 8 <= parts * local_count <= 16384:
        return None
    zero = emit.iota(1)
    one = emit.scalar("add", zero, 1.0)
    seed = emit.emit("NKIIota", "", f"partitions={rows}, width={count}, pattern=[[0, {count}]], channel_multiplier=0")
    result_values = emit.copy(seed)
    result_indices = emit.cast("NKIUInt32Cast", seed)
    result_valid = emit.copy(zero) if rows == 1 else emit.slice(seed, 0, 1)
    with emit.guard(one) if rows == 1 else nullcontext():
        reshaped = source if source_hbm else emit.emit("NKIReshapeStore", f"src={source}", f"rows={parts}")
        load_op = "NKIGroupedLoad" if source_hbm else "NKILoad"
        load_config = f"groups=1, rows={rows}, stages={parts}" if source_hbm else ""
        loaded = emit.emit(load_op, f"src={reshaped}", load_config)
        loaded = emit.cast("NKIFloat32Cast", loaded) if source_hbm else loaded
        local_values, local_indices, valid = native_prefix(emit, loaded, chunk, local_count, parts * rows, rows > 1)
        offset_config = f"partitions={parts * rows}, width=1, pattern=[[0, 1]], channel_multiplier={chunk}"
        offsets = emit.emit("NKIIota", "", offset_config)
        absolute = emit.scalar("add", emit.cast("NKIFloat32Cast", local_indices), offsets)
        absolute = emit.cast("NKIUInt32Cast", absolute) if rows == 1 else absolute
        flattened = []
        for value in (local_values, absolute):
            store_op = "NKIFlattenStore" if rows == 1 else "NKIGroupedStore"
            store_config = f"width={parts * local_count}" if rows == 1 else f"groups=1, rows={rows}, stages={parts}"
            stored = emit.emit(store_op, f"src={value}", store_config)
            flattened.append(emit.emit("NKILoad", f"src={stored}"))
        values, ranks, merged_valid = native_prefix(emit, flattened[0], parts * local_count, count, rows, rows > 1)
        indices = emit.emit("NKINCGather", f"data={flattened[1]}, indices={ranks}")
        if rows > 1:
            base_config = f"partitions={rows}, width=1, pattern=[[0, 1]], channel_multiplier={width}"
            bases = emit.emit("NKIIota", "", base_config)
            indices = emit.cast("NKIUInt32Cast", emit.scalar("subtract", emit.cast("NKIFloat32Cast", indices), bases))
        boundary = emit.emit("NKITensorSlice", f"src={local_values}", f"start={local_count - 1}, width=1")
        boundary = emit.emit("NKIDMATranspose", f"src={boundary}")
        config = f"partitions=1, groups={rows}, chunks={parts}"
        reduce_op = "NKITensorReduce" if rows == 1 else "NKIGroupedTileReduce"
        reduce_config = "op='max', axis=1" if rows == 1 else f"{config}, op='max'"
        cutoff = emit.emit(reduce_op, f"data={boundary}", reduce_config)
        if rows > 1:
            cutoff = emit.emit("NKIDMATranspose", f"src={cutoff}")
            valid = emit.emit("NKIDMATranspose", f"src={valid}")
            valid = emit.emit("NKIGroupedTileReduce", f"data={valid}", f"{config}, op='add'")
            valid = emit.scalar("equal", emit.emit("NKIDMATranspose", f"src={valid}"), float(parts))
        last = emit.emit("NKITensorSlice", f"src={values}", f"start={count - 1}, width=1")
        separated = emit.scalar("greater", last, cutoff)
        accepted = emit.binary("multiply", emit.binary("multiply", valid, merged_valid), separated)
        offset = emit.cast("NKIUInt32Cast", zero)
        emit.copy_windows((result_values, result_indices, result_valid), (values, indices, accepted), offset)
    return result_values, result_indices, result_valid


__all__ = ["NKIReshapeStore"]
