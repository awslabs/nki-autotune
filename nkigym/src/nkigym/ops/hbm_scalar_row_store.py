"""Store SBUF rows into a runtime-selected HBM interval."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKIHBMScalarRowStore(NKIOp):
    """Replace consecutive HBM rows through one scalar-indexed DMA."""

    NAME: ClassVar[str] = "dma_copy"
    INDIRECT_DMA_MODE: ClassVar[str | None] = "scalar_scatter"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "indices": ("I", "J"), "dst": ("E", "L")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src", "indices"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {
        "src": frozenset({"sbuf"}),
        "indices": frozenset({"sbuf", "register"}),
    }
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset({"dst"})
    RETURN_RMW_OPERAND: ClassVar[str | None] = "dst"
    SYNTHESIZE_RMW_INITIALIZER: ClassVar[bool] = False
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"indices": "uint32"}
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"I": 1, "J": 1}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"E", "L", "I", "J"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"E": 1, "L": 1, "I": 1, "J": 1, "P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"E": None, "L": None, "I": 1, "J": 1, "P": 128, "F": None}
    SPLIT_OFFSET_KWARGS: ClassVar[dict[str, tuple[str, str]]] = {
        "P": ("row_offset", "src"),
        "F": ("column_offset", "src"),
    }
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"row_offset", "column_offset"})
    OUTPUT_ROLE: ClassVar[str] = "stored"
    OUTPUT_LOCATION: ClassVar[str] = "shared_hbm"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require matching matrix dtypes and an in-bounds uint32 row index."""
        if _operand_role(kwargs["src"]) not in {None, "sbuf"}:
            raise TypeError("NKIHBMScalarRowStore.src expects SBUF")
        if _operand_role(kwargs["dst"]) not in {None, "stored", "shared_hbm"}:
            raise TypeError("NKIHBMScalarRowStore.dst expects HBM")
        if _operand_role(kwargs["indices"]) not in {None, "sbuf", "register"}:
            raise TypeError("NKIHBMScalarRowStore.indices expects SBUF or a register")
        source, indices, destination = (np.asarray(kwargs[name]) for name in ("src", "indices", "dst"))
        if source.ndim != 2 or destination.ndim != 2 or source.shape[1] != destination.shape[1]:
            raise ValueError("scalar row stores require matrices with matching row widths")
        if source.dtype != destination.dtype or indices.dtype != np.uint32 or indices.shape != (1, 1):
            raise TypeError("scalar row stores require matching data dtypes and one uint32 index")
        if int(indices[0, 0]) + source.shape[0] > destination.shape[0]:
            raise ValueError("scalar row store exceeds the destination row extent")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Replace the selected rows and preserve every other destination row."""
        source, destination = np.asarray(kwargs["src"]), kwargs["dst"]
        start = int(np.asarray(kwargs["indices"])[0, 0])
        destination[start : start + source.shape[0]] = source.copy()
        return destination


__all__ = ["NKIHBMScalarRowStore"]
