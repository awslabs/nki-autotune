"""HBM-to-SBUF DMA load with an fp32 destination."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import CopyContract, NKIOp, _operand_role


class NKIFloat32Load(NKIOp):
    """Load one HBM tensor directly into fp32 SBUF storage."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> CopyContract:
        """Return the value-preserving load contract."""
        _ = kwargs
        return CopyContract(input_operand="src", output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one HBM source."""
        if (role := _operand_role(kwargs["src"])) is not None and role not in {"param", "shared_hbm", "stored"}:
            raise TypeError(f"NKIFloat32Load(src=<role={role}>) expects HBM")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the HBM source represented in fp32."""
        return np.asarray(kwargs["src"], dtype=np.float32).copy()


__all__ = ["NKIFloat32Load"]
