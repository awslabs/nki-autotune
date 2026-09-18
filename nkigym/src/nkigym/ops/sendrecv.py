"""Peer SBUF exchange operation: maps to ``nisa.sendrecv``."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, PartitionTileBatchingContract, PeerExchangeContract, _operand_role


class NKISendRecv(NKIOp):
    """Exchange one SBUF tile with the peer logical NeuronCore."""

    NAME: ClassVar[str] = "sendrecv"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"})}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": 512}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    SHARDED_SINGLE_PROGRAM_ZERO: ClassVar[bool] = True
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"participating_programs"})

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PeerExchangeContract:
        """Return the value-preserving peer-exchange contract."""
        if kwargs.get("participating_programs", 2) not in {1, 2}:
            raise ValueError("peer exchange supports one or two participating programs")
        return PeerExchangeContract(input_operand="src", output_operand="dst")

    @classmethod
    def partition_tile_batching_contract(cls, kwargs: Mapping[str, Any]) -> PartitionTileBatchingContract:
        """Allow one exchange to span every contiguous physical tile."""
        _ = kwargs
        return PartitionTileBatchingContract(operands=("src", "dst"))

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an SBUF source."""
        role = _operand_role(kwargs["src"])
        if role is not None and role != "sbuf":
            raise TypeError(f"NKISendRecv(src=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> Any:
        """Return the zero peer contribution for one-program execution."""
        return np.zeros_like(kwargs["src"])
