"""Independent gathers over contiguous groups along a tensor's free axis."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchSegments, TorchValue
from nkigym.ops.base import NKIOp, _operand_role


class NKIGroupedFreeGather(NKIOp):
    """Gather local indices within each contiguous free-axis group."""

    NAME: ClassVar[str] = "nc_n_gather"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("P", "G", "F"),
        "indices": ("P", "G", "N"),
        "dst": ("P", "G", "N"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("P",), ("G", axis)) for slot, axis in {"data": "F", "indices": "N", "dst": "N"}.items()
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("P",), (axis,)) for slot, axis in {"data": "F", "indices": "N", "dst": "N"}.items()
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "indices"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {slot: frozenset({"sbuf"}) for slot in INPUT_OPERANDS}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"indices": "uint32"}
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "PGFN"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "G": 1, "F": None, "N": None}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"F"})
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int) -> None:
        """Configure the number of independent contiguous source groups."""
        if groups < 1:
            raise ValueError("grouped free gather requires a positive group count")
        super().__init__(groups=groups)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require matching SBUF partitions and unsigned local indices."""
        data, indices = np.asarray(kwargs["data"]), np.asarray(kwargs["indices"])
        if any(_operand_role(kwargs[name]) not in {None, "sbuf"} for name in self.INPUT_OPERANDS):
            raise TypeError("grouped free gather requires SBUF inputs")
        if data.ndim != 2 or indices.ndim != 2 or data.shape[0] != indices.shape[0] or indices.dtype != np.uint32:
            raise ValueError("grouped free gather requires matching rank-two inputs and uint32 indices")
        if data.shape[1] % int(kwargs["groups"]) or indices.shape[1] % int(kwargs["groups"]):
            raise ValueError("grouped free gather requires equal-sized groups")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Gather without crossing a group's free-axis bounds."""
        data, indices = np.asarray(kwargs["data"]), np.asarray(kwargs["indices"])
        groups, partitions = int(kwargs["groups"]), data.shape[0]
        source = data.reshape(partitions, groups, -1)
        selected = indices.reshape(partitions, groups, -1).astype(np.int64)
        if np.any(selected >= source.shape[2]):
            raise ValueError("grouped free gather index exceeds its local source group")
        return np.take_along_axis(source, selected, axis=2).reshape(indices.shape)


def folded_max_layout(rows: int, width: int) -> tuple[int, int]:
    """Choose exact chunks whose groups fit the physical partition limit."""
    capacity = 256 if rows > 1 and rows % 2 == 0 else 128
    limit = min(capacity // rows, max(1, width // 128))
    parts = next((size for size in range(limit, 1, -1) if width % size == 0), 1)
    return (2 if rows * parts > 128 else 1), parts


def emit_pairwise_maximum(
    emit: TorchArithmetic, tables: list[str], rows: int, chunk: int, groups: int
) -> tuple[TorchSegments, TorchSegments]:
    """Select the first NaN or first maximum from two consecutive partials."""
    config = f"groups={groups}, source_width={2 * rows // groups}, pattern={((2, rows // groups),)!r}"
    left = emit.emit("NKIGroupedStridedCopy", f"src={tables[0]}", f"{config}, offset=0")
    right = emit.emit("NKIGroupedStridedCopy", f"src={tables[0]}", f"{config}, offset=1")
    choose_left = emit.binary(
        "maximum", emit.binary("greater_equal", left, right), emit.binary("not_equal", left, left)
    )
    choose_right = emit.inverse(choose_left)
    offsets = emit.emit(
        "NKIIota",
        "",
        f"partitions=1, width={rows}, pattern=[[0, {groups}], [2, {rows // groups}]], channel_multiplier=0",
    )
    positions = emit.cast("NKIUInt32Cast", emit.binary("add", offsets, choose_right))
    selected = [
        emit.emit("NKIGroupedFreeGather", f"data={table}, indices={positions}", f"groups={groups}") for table in tables
    ]
    absolute = emit.binary("add", selected[1], emit.scalar("multiply", choose_right, float(chunk)))
    indices = emit.cast("NKIUInt32Cast", absolute)
    return (
        TorchSegments((TorchValue(selected[0], (rows, 1), transposed=True, storage_dtype="float32"),)),
        TorchSegments((TorchValue(indices, (rows, 1), transposed=True, storage_dtype="uint32"),)),
    )


__all__ = ["NKIGroupedFreeGather"]
