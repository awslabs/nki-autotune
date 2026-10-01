"""Load independent row and chunk groups from a flat HBM column."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.ops.base import NKIOp, _operand_role


class NKIFlatChunkLoad(NKIOp):
    """Reshape a flat HBM column into independently loaded row and chunk groups."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "G", "T", "F", "I"), "dst": ("P", "G", "T", "F")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("P", "G", "T", "F"), ("I",)),
        "dst": (("P",), ("G", "T", "F")),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {"src": (("P",), ("G", "T", "F"))}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"shared_hbm"})}
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {
        "P": "partitions",
        "G": "groups",
        "T": "chunks",
        "F": "width",
        "I": 1,
    }
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "PGTFI"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "G": 1, "T": 1, "F": None, "I": 1}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions", "chunks", "width"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, partitions: int, chunks: int, width: int) -> None:
        """Configure positive independent row and chunk domains."""
        if min(groups, partitions, chunks, width) < 1:
            raise ValueError("flat chunk load dimensions must be positive")
        super().__init__(groups=groups, partitions=partitions, chunks=chunks, width=width)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require a flat HBM column containing exactly the configured elements."""
        source = np.asarray(kwargs["src"])
        count = int(np.prod([kwargs[name] for name in ("groups", "partitions", "chunks", "width")]))
        if _operand_role(kwargs["src"]) not in {None, "param", "shared_hbm", "stored"} or source.shape != (count, 1):
            raise ValueError("flat chunk load requires a matching HBM column")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Preserve the complete row-major element order."""
        return np.asarray(kwargs["src"]).reshape(int(kwargs["partitions"]), -1).copy()


def restore_chunk_rows(source: str, rows: int, factor: int, emit: TorchArithmetic) -> str:
    """Restore consecutive partition groups to the original row and chunk order."""
    if factor == 1:
        return source
    stored = emit.emit("NKIGroupedStore", f"src={source}", f"groups=1, rows={rows}, stages={factor}")
    return emit.emit("NKILoad", f"src={stored}")


def repeat_row_vector(source: str, rows: int, factor: int, emit: TorchArithmetic) -> str:
    """Repeat each row scalar in consecutive partitions without numeric conversion."""
    if factor == 1:
        return source
    partitions = rows * factor
    result = emit.emit("NKIIota", "", f"partitions={partitions}, width=1, pattern=[[0, 1]], channel_multiplier=0")
    for start in range(0, partitions, 32):
        count = min(32, partitions - start)
        source_start = (start // factor // 32) * 32
        mask = [(start + lane) // factor - source_start if lane < count else 255 for lane in range(32)]
        result = emit.emit(
            "NKIStreamShuffle",
            f"src={source}, dst={result}",
            f"source_start={source_start}, source_rows={min(32, rows - source_start)}, "
            f"destination_start={start}, destination_rows={count}, start=0, width=1, shuffle_mask={mask}",
        )
    return result


__all__ = ["NKIFlatChunkLoad"]
