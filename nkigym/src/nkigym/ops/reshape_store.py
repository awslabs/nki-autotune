"""Store contiguous SBUF row segments as separate HBM rows."""

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


def emit_partitioned_prefix(emit: ControlEmitter, source: str, width: int, count: int) -> tuple[str, str, str] | None:
    """Select bounded local prefixes and prove they contain the complete global prefix."""
    first = max(2, (width + 511) // 512)
    parts = next((size for size in range(first, min(128, width) + 1) if width % size == 0), 1)
    chunk = width // parts
    local_count = max(8, ((4 * count + 8 * parts - 1) // (8 * parts)) * 8)
    if local_count + 8 > chunk:
        local_count = max(8, ((3 * count + 16 * parts - 1) // (16 * parts)) * 8)
    if count < 16 or parts == 1 or local_count + 8 > chunk or not count + 8 <= parts * local_count <= 16384:
        return None
    zero = emit.iota(1)
    one = emit.scalar("add", zero, 1.0)
    seed = emit.emit("NKIIota", "", f"partitions=1, width={count}, pattern=[[0, {count}]], channel_multiplier=0")
    result_values = emit.copy(seed)
    result_indices = emit.cast("NKIUInt32Cast", seed)
    result_valid = emit.copy(zero)
    with emit.guard(one):
        reshaped = emit.emit("NKIReshapeStore", f"src={source}", f"rows={parts}")
        loaded = emit.emit("NKILoad", f"src={reshaped}")
        local_values, local_indices, valid = native_prefix(emit, loaded, chunk, local_count, parts)
        offsets = emit.emit("NKIIota", "", f"partitions={parts}, width=1, pattern=[[0, 1]], channel_multiplier={chunk}")
        absolute = emit.scalar("add", emit.cast("NKIFloat32Cast", local_indices), offsets)
        absolute = emit.cast("NKIUInt32Cast", absolute)
        flattened = []
        for value in (local_values, absolute):
            stored = emit.emit("NKIFlattenStore", f"src={value}", f"width={parts * local_count}")
            flattened.append(emit.emit("NKILoad", f"src={stored}"))
        values, ranks, merged_valid = native_prefix(emit, flattened[0], parts * local_count, count)
        indices = emit.emit("NKINCGather", f"data={flattened[1]}, indices={ranks}")
        boundary = emit.emit("NKITensorSlice", f"src={local_values}", f"start={local_count - 1}, width=1")
        boundary = emit.emit("NKIDMATranspose", f"src={boundary}")
        cutoff = emit.emit("NKITensorReduce", f"data={boundary}", "op='max', axis=1")
        last = emit.emit("NKITensorSlice", f"src={values}", f"start={count - 1}, width=1")
        separated = emit.scalar("greater", last, cutoff)
        accepted = emit.binary("multiply", emit.binary("multiply", valid, merged_valid), separated)
        offset = emit.cast("NKIUInt32Cast", zero)
        emit.copy_windows((result_values, result_indices, result_valid), (values, indices, accepted), offset)
    return result_values, result_indices, result_valid


__all__ = ["NKIReshapeStore"]
