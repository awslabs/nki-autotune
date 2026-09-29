"""Exact float32 movement within one native partition-shuffle quadrant."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.ops.base import NKIOp, _operand_role


class NKIStreamShuffle(NKIOp):
    """Copy selected source partitions into an existing float32 window."""

    NAME: ClassVar[str] = "nc_stream_shuffle"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("S", "F"), "dst": ("P", "O")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {slot: frozenset({"sbuf"}) for slot in ("src", "dst")}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {
        slot: frozenset({"float32"}) for slot in ("src", "dst")
    }
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset({"dst"})
    RETURN_RMW_OPERAND: ClassVar[str | None] = "dst"
    SYNTHESIZE_RMW_INITIALIZER: ClassVar[bool] = False
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"S", "F", "P", "O"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in ("S", "F", "P", "O")}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"S": 128, "P": 128, "F": None, "O": None}
    PARTITION_SLICES: ClassVar[dict[str, tuple[str, str]]] = {
        "src": ("source_start", "source_rows"),
        "dst": ("destination_start", "destination_rows"),
    }
    INPUT_SLICES: ClassVar[dict[str, tuple[tuple[int, str, str], ...]]] = {"dst": ((1, "start", "width"),)}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset(
        {"source_start", "source_rows", "destination_start", "destination_rows", "start", "width"}
    )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require float32 windows, allowing an implicit unit free axis on the source."""
        arrays = {name: np.asarray(kwargs[name]) for name in ("src", "dst")}
        if arrays["src"].ndim == 1:
            arrays["src"] = arrays["src"].reshape(-1, 1)
        if any(_operand_role(kwargs[name]) not in {None, "sbuf"} for name in arrays):
            raise TypeError("partition shuffle requires SBUF operands")
        if any(value.ndim != 2 or value.dtype != np.float32 for value in arrays.values()):
            raise TypeError("partition shuffle requires rank-two float32 operands")
        for slot, prefix in (("src", "source"), ("dst", "destination")):
            start, rows = int(kwargs[f"{prefix}_start"]), int(kwargs[f"{prefix}_rows"])
            if start < 0 or start % 32 or not 1 <= rows <= 32 or start + rows > arrays[slot].shape[0]:
                raise ValueError("shuffle partition windows must stay within one aligned quadrant")
        start, width = int(kwargs["start"]), int(kwargs["width"])
        if start < 0 or width != arrays["src"].shape[1] or start + width > arrays["dst"].shape[1]:
            raise ValueError("shuffle source and destination free extents must match")
        mask = kwargs["shuffle_mask"]
        if len(mask) != 32 or any(
            type(index) is not int or index != 255 and not 0 <= index < kwargs["source_rows"] for index in mask
        ):
            raise ValueError("shuffle mask must contain valid source lanes or 255")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Preserve inactive lanes while copying each selected source row."""
        source, result = np.asarray(kwargs["src"]).reshape(len(kwargs["src"]), -1).copy(), kwargs["dst"]
        start, width = int(kwargs["start"]), int(kwargs["width"])
        for lane, index in enumerate(kwargs["shuffle_mask"][: int(kwargs["destination_rows"])]):
            if index != 255:
                result[int(kwargs["destination_start"]) + lane, start : start + width] = source[
                    int(kwargs["source_start"]) + index
                ]
        return result


def emit_partition_reshape(emit: TorchArithmetic, source: str, shape: tuple[int, int], rows: int) -> str:
    """Pack consecutive source rows into wider rows using native shuffles."""
    source_rows, width = shape
    if not 1 < rows < source_rows <= 128 or source_rows % rows:
        raise ValueError("partition reshape requires an exact row grouping within 128 partitions")
    groups = source_rows // rows
    result = emit.emit(
        "NKIIota",
        "",
        f"partitions={rows}, width={groups * width}, pattern=[[0, {groups * width}]], channel_multiplier=0",
    )
    for destination_start in range(0, rows, 32):
        destination_rows = min(32, rows - destination_start)
        for group in range(groups):
            for source_start in range(0, source_rows, 32):
                source_count = min(32, source_rows - source_start)
                positions = [(destination_start + lane) * groups + group - source_start for lane in range(32)]
                mask = [
                    index if lane < destination_rows and 0 <= index < source_count else 255
                    for lane, index in enumerate(positions)
                ]
                if any(index != 255 for index in mask):
                    result = emit.emit(
                        "NKIStreamShuffle",
                        f"src={source}, dst={result}",
                        f"source_start={source_start}, source_rows={source_count}, "
                        f"destination_start={destination_start}, destination_rows={destination_rows}, "
                        f"start={group * width}, width={width}, shuffle_mask={mask}",
                    )
    return result


__all__ = ["NKIStreamShuffle"]
