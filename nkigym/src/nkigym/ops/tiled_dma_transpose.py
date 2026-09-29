"""Transpose independent free-axis tiles with one native batched DMA."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, PermutationContract, _operand_role


class NKITiledDMATranspose(NKIOp):
    """Transpose each free-axis tile while retaining a packed tile axis."""

    NAME: ClassVar[str] = "dma_transpose"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "T", "F"), "dst": ("F", "T", "P")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("P",), ("T", "F")),
        "dst": (("F",), ("T", "P")),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("P",), ("T",), (), ("F",)),
        "dst": (("F",), ("T",), (), ("P",)),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"})}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"float32", "bfloat16", "float16"})}
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"T": "tiles", "F": "width"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "T": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "T": None, "F": 128}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"P", "T", "F"})
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"tiles", "width"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    OUTPUT_TILE_ALIGNMENT_BYTES: ClassVar[dict[str, int]] = {"dst": 32}

    def __init__(self, tiles: int, width: int) -> None:
        """Configure positive tiles whose transposed partition width fits DMA."""
        if tiles < 1 or not 1 <= width <= 128:
            raise ValueError("tiled DMA transpose requires positive tiles and width at most 128")
        super().__init__(tiles=tiles, width=width, axes=(3, 1, 2, 0))

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PermutationContract:
        """Describe the complete permutation of partition, tile, and free axes."""
        return PermutationContract(input_operand="src", output_operand="dst", permutation=(2, 1, 0))

    def _check_roles(self, **kwargs: Any) -> None:
        """Require compatible SBUF data and complete source tiles."""
        source = np.asarray(kwargs["src"])
        if _operand_role(kwargs["src"]) not in {None, "sbuf"} or source.ndim != 2:
            raise TypeError("tiled DMA transpose requires rank-two SBUF data")
        if source.shape[0] > 128 or source.shape[1] != kwargs["tiles"] * kwargs["width"]:
            raise ValueError("tiled DMA transpose shape does not match its configuration")
        if str(source.dtype) not in {"float32", "bfloat16", "float16"}:
            raise TypeError("tiled DMA transpose requires a supported floating-point dtype")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Transpose each tile without arithmetic or dtype conversion."""
        source = np.asarray(kwargs["src"])
        tiles, width = int(kwargs["tiles"]), int(kwargs["width"])
        return source.reshape(source.shape[0], tiles, width).transpose(2, 1, 0).reshape(width, -1).copy()
