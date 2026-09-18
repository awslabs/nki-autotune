"""Indirect flattened-element HBM gather."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.grouped_store import rotational_topk_generated_inputs


class NKIHBMColumnGather(NKIOp):
    """Gather one flattened HBM element per partition and group."""

    NAME: ClassVar[str] = "dma_copy"
    INDIRECT_DMA_MODE: ClassVar[str | None] = "column_gather"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"src": ("R", "F"), "indices": ("P", "G"), "dst": ("P", "G")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src", "indices"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"P": "partitions", "G": "groups"}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"R", "F"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"R": 1, "F": 1, "P": 1, "G": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"R": None, "F": None, "P": 128, "G": 1}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"indices": "uint32"}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"partitions", "groups"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require HBM data and on-chip absolute offsets."""
        if _operand_role(kwargs["src"]) != "param" or _operand_role(kwargs["indices"]) != "sbuf":
            raise TypeError("NKIHBMColumnGather expects HBM data and SBUF indices")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Gather absolute flattened offsets from the source tensor."""
        source, indices = np.asarray(kwargs["src"]).reshape(-1), np.asarray(kwargs["indices"], dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= source.size):
            raise ValueError("HBM column-gather offsets exceed the source")
        return source[indices]


def generated_kernel_inputs(
    specs: Mapping[str, tuple[tuple[int, ...], str]], existing: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Add flattened HBM views and other generated kernel inputs."""
    result = dict(existing)
    result.update(
        {
            name: result[name.removesuffix("_flat")].reshape(shape)
            for name, (shape, _dtype) in specs.items()
            if name.endswith("_flat") and name not in result and name.removesuffix("_flat") in result
        }
    )
    result.update(rotational_topk_generated_inputs(specs, result))
    return result


__all__ = ["NKIHBMColumnGather", "generated_kernel_inputs"]
