"""Indirect HBM row gather through ``nisa.dma_copy``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchSegments, TorchValue
from nkigym.ops.base import NKIOp, _operand_role


class NKIGather(NKIOp):
    """Gather valid HBM rows selected by one SBUF index per partition."""

    NAME: ClassVar[str] = "dma_copy"
    INDIRECT_DMA_MODE: ClassVar[str | None] = "gather"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("S", "F"), "indices": ("P", "I"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src", "indices"})
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"S"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"S": 1, "P": 1, "F": 1, "I": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"S": None, "P": 128, "F": None, "I": 1}
    OUTPUT_ROLE: ClassVar[str] = "sbuf"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an HBM source and SBUF indices."""
        if (role := _operand_role(kwargs["src"])) is not None and role != "param":
            raise TypeError(f"NKIGather(src=<role={role}>) expects an HBM parameter")
        if (role := _operand_role(kwargs["indices"])) is not None and role != "sbuf":
            raise TypeError(f"NKIGather(indices=<role={role}>) expects SBUF indices")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Gather rows after rejecting indices outside the HBM source."""
        source = np.asarray(kwargs["src"])
        indices = np.asarray(kwargs["indices"]).reshape(-1).astype(np.int64)
        valid = (indices >= 0) & (indices < source.shape[0])
        if not np.all(valid):
            raise ValueError("NKIGather indices must select valid source rows")
        result = source[indices]
        return result


def emit_routed_gather(
    source: str, indices: TorchValue | TorchSegments, width: int, stem: str, body: list[str], imports: set[str]
) -> TorchValue | TorchSegments:
    """Emit one wide or segmented routed HBM gather."""
    values = indices.values if isinstance(indices, TorchSegments) else (indices,)
    chunks: list[TorchValue] = []
    imports.add("NKIGather")
    for index, value in enumerate(values):
        if len(value.shape) != 2:
            raise ValueError(f"routed gather indices must be rank two, got {value.shape}")
        offsets = TorchValue(value.name, tuple(reversed(value.shape)), not value.transposed)
        if offsets.transposed:
            offsets = TorchValue(f"sbuf_{stem}_indices_{index}", offsets.shape)
            imports.add("NKIDMATranspose")
            body.append(f"{offsets.name} = NKIDMATranspose()(src={value.name})")
        target = TorchValue(f"sbuf_{stem}_{index}", (offsets.shape[0], width))
        body.append(f"{target.name} = NKIGather()(src={source}, indices={offsets.name})")
        chunks.append(target)
    return TorchSegments(tuple(chunks), axis=0) if isinstance(indices, TorchSegments) else chunks[0]


__all__ = ["NKIGather", "emit_routed_gather"]
