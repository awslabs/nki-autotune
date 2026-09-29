"""Load an HBM vector as consecutive SBUF partition rows."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.strided_tensor_copy import emit_torch_sum


class NKIVectorPartitionLoad(NKIOp):
    """Reshape contiguous vector elements through one native DMA copy."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("P", "F"),),
        "dst": (("P",), ("F",)),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("P",), ("F",)) for slot in OPERAND_AXES
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"shared_hbm"})}
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"P": "partitions"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"partitions"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, partitions: int) -> None:
        """Select the number of consecutive equal-length vector pieces."""
        if not 1 <= partitions <= 128:
            raise ValueError("vector partition load requires between 1 and 128 partitions")
        super().__init__(partitions=partitions)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one HBM vector whose length divides into complete rows."""
        source = np.asarray(kwargs["src"])
        if _operand_role(kwargs["src"]) not in {None, "param", "shared_hbm", "stored"} or source.ndim != 1:
            raise TypeError("vector partition load requires one HBM vector")
        if source.size % kwargs["partitions"]:
            raise ValueError("vector length must divide evenly among partitions")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the row-major reshape without arithmetic or dtype conversion."""
        return np.asarray(kwargs["src"]).reshape(int(kwargs["partitions"]), -1).copy()


def correct_overflow_sum(
    source: TorchValue, total: TorchValue, logical: tuple[int, int], body: list[str], imports: set[str]
) -> TorchValue:
    """Retain reference addition order without runtime control flow."""
    rows, width = logical
    emit = TorchArithmetic(f"{total.name}_reference", body, imports)
    lanes = 8 if width >= 8 else 1
    units = width // (4 * lanes)
    span = (1 << max(4, (max(1, units) - 1).bit_length() // 4)) * 4 * lanes
    if source.shape[0] != rows and (width % span or source.shape[1] % span):
        stored = emit.emit(
            "NKIGroupedStore", f"src={source.name}", f"groups=1, rows={rows}, stages={source.shape[0] // rows}"
        )
        loaded = emit.emit("NKILoad", f"src={stored}")
        source = TorchValue(loaded, logical, storage_dtype=source.storage_dtype)
    return emit_torch_sum(source, f"{total.name}_ordered", body, imports, logical if source.shape[0] != rows else None)


__all__ = ["NKIVectorPartitionLoad"]
