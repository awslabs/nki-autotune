"""Peer SBUF exchange operation: maps to ``nisa.sendrecv``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKISendRecv(NKIOp):
    """Exchange one SBUF tile with the peer logical NeuronCore."""

    NAME: ClassVar[str] = "sendrecv"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"})}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": 512}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    SINGLE_PROGRAM_ZERO: ClassVar[bool] = True

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an SBUF source."""
        role = _operand_role(kwargs["src"])
        if role is not None and role != "sbuf":
            raise TypeError(f"NKISendRecv(src=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> Any:
        """Return the zero peer contribution used by one-program simulation."""
        return np.zeros_like(kwargs["src"])
