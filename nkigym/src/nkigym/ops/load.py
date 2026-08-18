"""HBM → SBUF ``nisa.dma_copy`` operation."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import CopyContract, NKIOp, _operand_role


class NKILoad(NKIOp):
    """Copy an HBM tensor into an SBUF buffer with identical logical layout."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    """``nisa.dma_copy`` has no tile-size constraint beyond
    ``src.size == dst.size`` and partition-dim validation. Only the
    partition axis is capped by the NeuronCore's 128-partition SBUF
    layout; the free axis is unbounded."""
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    PREFERRED_TILE_SIZE: ClassVar[dict[str, int]] = {"F": 512}
    OUTPUT_ROLE: ClassVar[str] = "sbuf"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> CopyContract:
        """Return the value-preserving load contract."""
        _ = kwargs
        return CopyContract(input_operand="src", output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """``src`` must be HBM-resident (``param``)."""
        role = _operand_role(kwargs["src"])
        if role is not None and role not in {"param", "shared_hbm", "stored"}:
            raise TypeError(f"NKILoad(src=<role={role}>) expects an HBM tensor")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate and return a copy of ``src``."""
        src: np.ndarray = kwargs["src"]
        return np.array(src)
