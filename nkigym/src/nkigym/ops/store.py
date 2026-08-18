"""SBUF → HBM ``nisa.dma_copy`` operation."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import CopyContract, NKIOp, _operand_role


class NKIStore(NKIOp):
    """Copy an SBUF buffer back to HBM with identical logical layout."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    """Same story as ``NKILoad``: ``nisa.dma_copy`` only caps the
    partition axis (128) — the free axis is unbounded."""
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_ROLE: ClassVar[str] = "stored"
    OUTPUT_LOCATION: ClassVar[str] = "shared_hbm"
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"})}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"program_ownership"})

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> CopyContract:
        """Return the value-preserving store contract."""
        _ = kwargs
        return CopyContract(input_operand="src", output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """``src`` must be SBUF-resident."""
        role = _operand_role(kwargs["src"])
        if role is not None and role != "sbuf":
            raise TypeError(f"NKIStore(src=<role={role}>) expects sbuf; did you forget to stage through SBUF?")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate and return a copy of ``src`` in HBM."""
        src: np.ndarray = kwargs["src"]
        return np.array(src)
