"""Grouped four-dimensional DMA transpose."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import BatchedPermutationContract, NKIOp, PermutationContract, _operand_role


class NKIGroupedDMATranspose(NKIOp):
    """Transpose grouped score tiles into stationary PV operands."""

    NAME: ClassVar[str] = "dma_transpose"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "src": ("G", "Q", "P", "T", "D", "K"),
        "dst": ("T", "D", "K", "G", "Q", "P"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("G", "Q", "P"), ("T", "D", "K")),
        "dst": (("T", "D", "K"), ("G", "Q", "P")),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("P",), (), ("D",), ("K",)),
        "dst": (("K",), (), ("D",), ("P",)),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {
        "G": "groups",
        "Q": "queries",
        "T": "tiles",
        "D": "subtiles",
        "P": "partitions",
        "K": "partition_width",
    }
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "GQTDKP"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "Q": 1, "T": 1, "D": None, "K": 128, "P": 128}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"D", "K", "P"})
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset(
        {"groups", "queries", "tiles", "subtiles", "partitions", "partition_width"}
    )
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    OUTPUT_DTYPE: ClassVar[str | None] = "bfloat16"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "bfloat16"
    OUTPUT_TILE_ALIGNMENT_BYTES: ClassVar[dict[str, int]] = {"dst": 32}

    def __init__(self, groups: int, queries: int, tiles: int, subtiles: int, partitions: int) -> None:
        """Configure grouped score and transpose extents."""
        super().__init__(
            groups=groups,
            queries=queries,
            tiles=tiles,
            subtiles=subtiles,
            partitions=partitions,
            partition_width=128,
            axes=(3, 1, 2, 0),
        )

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PermutationContract:
        """Return the grouped score transpose contract."""
        _ = kwargs
        return PermutationContract(
            input_operand="src",
            output_operand="dst",
            permutation=(3, 4, 5, 0, 1, 2),
            batching=BatchedPermutationContract(permutation=(3, 1, 2, 0), input_axes=(0, 3), batch_axis=2),
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one SBUF source."""
        if (role := _operand_role(kwargs["src"])) is not None and role != "sbuf":
            raise TypeError(f"NKIGroupedDMATranspose(src=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the grouped four-dimensional permutation."""
        g, q, t, d, p = (int(kwargs[key]) for key in ("groups", "queries", "tiles", "subtiles", "partitions"))
        source = np.asarray(kwargs["src"]).reshape(g, q, p, t, d, 128)
        return source.transpose(3, 4, 5, 0, 1, 2).reshape(t * d * 128, g * q * p)


__all__ = ["NKIGroupedDMATranspose"]
