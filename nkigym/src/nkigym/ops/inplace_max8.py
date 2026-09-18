"""In-place top-eight destination slices through ``nisa.max8``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, _operand_role
from nkigym.ops.index_iota import native_prefix
from nkigym.ops.register_load import ControlEmitter
from nkigym.ops.reshape_store import emit_partitioned_prefix


class NKIInplaceMax8(NKIOp):
    """Write one top-eight result into an existing packed destination."""

    NAME: ClassVar[str] = "max8"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("G", "P", "F"), "dst": ("G", "P", "K")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("G", "P"), ("F",)),
        "dst": (("G", "P"), ("K",)),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset({"dst"})
    RETURN_RMW_OPERAND: ClassVar[str | None] = "dst"
    SYNTHESIZE_RMW_INITIALIZER: ClassVar[bool] = False
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions"}
    INPUT_SLICES: ClassVar[dict[str, tuple[tuple[int, str, str], ...]]] = {
        "src": ((1, "source_start", "source_width"),),
        "dst": ((1, "output_start", "output_width"),),
    }
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset(
        {"groups", "partitions", "source_start", "source_width", "output_start", "output_width"}
    )
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "P": 1, "F": 8, "K": 8}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "F": None, "K": None}

    def _check_roles(self, **kwargs: Any) -> None:
        """Require SBUF operands."""
        if any(_operand_role(kwargs[name]) not in {None, "sbuf"} for name in ("src", "dst")):
            raise TypeError("NKIInplaceMax8 expects SBUF operands")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Write descending top-eight values into the selected output interval."""
        source = np.asarray(kwargs["src"])[:, : int(kwargs["source_width"])]
        start, width = int(kwargs["output_start"]), int(kwargs["output_width"])
        result = kwargs["dst"]
        result[:, start : start + width] = np.sort(source, axis=1)[:, -width:][:, ::-1]
        return result


def native_prefix_chunk(width: int, count: int) -> int | None:
    """Choose a divisor whose local and merged top-eight prefixes fit the ISA."""
    padded = (count // 8 + 1) * 8
    if width <= 16384:
        return width if padded <= width else None
    chunk = next(size for size in range(min(4096, width), 0, -1) if width % size == 0)
    return chunk if padded + 8 <= chunk and padded * (width // chunk) <= 16384 else None


def emit_native_prefix(emit: ControlEmitter, source: str, width: int, count: int) -> tuple[str, str, str]:
    """Merge exact local prefixes and reject ties or nonfinite candidates."""
    if (partitioned := emit_partitioned_prefix(emit, source, width, count)) is not None:
        return partitioned
    chunk = native_prefix_chunk(width, count)
    if chunk is None:
        raise ValueError("native prefix has no supported chunk layout")
    if chunk == width:
        return native_prefix(emit, source, width, count)
    local_count = (count // 8 + 1) * 8
    total = local_count * (width // chunk)
    zero = emit.emit("NKIIota", "", "partitions=1, width=1, pattern=[[0, 1]], channel_multiplier=0")
    seed = emit.emit("NKIIota", "", f"partitions=1, width={total}, pattern=[[0, {total}]], channel_multiplier=0")
    values = emit.emit("NKITensorCopy", f"src={seed}", "engine='vector'")
    positions = emit.cast("NKIUInt32Cast", seed)
    valid = emit.scalar("add", zero, 1.0)
    accepted_result = emit.scalar("multiply", zero, 0.0)
    result_values = emit.emit("NKITensorSlice", f"src={values}", f"start=0, width={count}")
    result_indices = emit.emit("NKITensorSlice", f"src={positions}", f"start=0, width={count}")
    emit.imports.add("NKIDynamicSliceCopy")
    with emit.guard(valid):
        for start in range(0, width, chunk):
            piece = emit.emit("NKITensorSlice", f"src={source}", f"start={start}, width={chunk}")
            selected, indices, accepted = native_prefix(emit, piece, chunk, local_count)
            absolute = emit.scalar("add", emit.cast("NKIFloat32Cast", indices), float(start))
            absolute = emit.cast("NKIUInt32Cast", absolute)
            offset = emit.cast("NKIUInt32Cast", emit.scalar("add", zero, float(start // chunk * local_count)))
            for destination, value in ((values, selected), (positions, absolute)):
                emit.line(f"{destination} = NKIDynamicSliceCopy()(src={value}, offset={offset}, dst={destination})")
            valid = emit.binary("multiply", valid, accepted)
        selected, ranks, accepted = native_prefix(emit, values, total, count)
        indices = emit.emit("NKINCGather", f"data={positions}, indices={ranks}")
        final_guard, offset = emit.binary("multiply", valid, accepted), emit.cast("NKIUInt32Cast", zero)
        for destination, value in (
            (result_values, selected),
            (result_indices, indices),
            (accepted_result, final_guard),
        ):
            emit.line(f"{destination} = NKIDynamicSliceCopy()(src={value}, offset={offset}, dst={destination})")
    return result_values, result_indices, accepted_result


__all__ = ["NKIInplaceMax8"]
