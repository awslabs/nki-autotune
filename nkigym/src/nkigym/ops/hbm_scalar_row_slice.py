"""Load one dynamically selected HBM matrix with ``nisa.dma_copy``."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import NKIOp, PartitionTileBatchingContract, _operand_role


def emit_fp32_row(emit: TorchArithmetic, source: TorchValue, register: str | None) -> str:
    """Load one HBM matrix row and preserve explicit low-precision promotion."""
    rows, width = source.shape
    if rows != 1 and register is None:
        raise ValueError("multiple HBM rows require an explicit row selector")
    loaded = emit.emit(
        "NKILoad" if rows == 1 else "NKIHBMScalarRowSlice",
        f"src={source.name}" + (f", indices={register}, index=0" if rows != 1 else ""),
        "" if rows == 1 else f"rows=1, width={width}",
    )
    return loaded if source.storage_dtype == "float32" else emit.cast("NKIFloat32Cast", loaded)


class NKIHBMScalarRowSlice(NKIOp):
    """Load one matrix selected by a configured scalar SBUF index."""

    NAME: ClassVar[str] = "dma_copy"
    INDIRECT_DMA_MODE: ClassVar[str | None] = "scalar_gather"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("E", "L"), "indices": ("I", "J"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src", "indices"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {
        "src": frozenset({"shared_hbm"}),
        "indices": frozenset({"sbuf", "register"}),
    }
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"P": "rows", "F": "width"}
    TILABLE_FIXED_AXES: ClassVar[frozenset[str]] = frozenset({"F"})
    """The configured matrix width is an iteration extent, not a DMA tile width."""
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"E", "L", "I", "J"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"E": 1, "L": 1, "I": 1, "J": 1, "P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"E": None, "L": None, "I": 1, "J": None, "P": 128, "F": None}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"indices": "uint32"}
    SPLIT_OFFSET_KWARGS: ClassVar[dict[str, tuple[str, str]]] = {
        "P": ("row_offset", "dst"),
        "F": ("column_offset", "dst"),
    }
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"index", "rows", "width", "row_offset", "column_offset"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    @classmethod
    def partition_tile_batching_contract(cls, kwargs: Mapping[str, Any]) -> PartitionTileBatchingContract:
        """Batch contiguous rows while retaining the same scalar matrix selector."""
        _ = kwargs
        return PartitionTileBatchingContract(operands=("dst",))

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one flattened HBM matrix and one scalar SBUF index."""
        if (role := _operand_role(kwargs["src"])) not in {None, "param", "shared_hbm", "stored"}:
            raise TypeError(f"NKIHBMScalarRowSlice(src=<role={role}>) expects an HBM source")
        if (role := _operand_role(kwargs["indices"])) is not None and role not in {"sbuf", "register"}:
            raise TypeError(f"NKIHBMScalarRowSlice(indices=<role={role}>) expects SBUF or a scalar register")
        source, indices = np.asarray(kwargs["src"]), np.asarray(kwargs["indices"])
        index = int(kwargs.get("index", 0))
        rows, width = int(kwargs["rows"]), int(kwargs["width"])
        if (
            source.ndim != 2
            or source.shape[1] != rows * width
            or indices.ndim != 2
            or indices.shape[0] != 1
            or index < 0
            or index >= indices.shape[1]
        ):
            raise ValueError("scalar HBM row slice requires flattened expert matrices and one index")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the dynamically selected expert matrix."""
        source = np.asarray(kwargs["src"])
        selected = int(np.asarray(kwargs["indices"])[0, int(kwargs.get("index", 0))])
        if selected < 0 or selected >= source.shape[0]:
            raise ValueError("scalar HBM row slice index exceeds the expert extent")
        return source[selected].reshape(int(kwargs["rows"]), int(kwargs["width"])).copy()


__all__ = ["NKIHBMScalarRowSlice"]
