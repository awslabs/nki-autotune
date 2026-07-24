"""Tensor-Engine transpose op for ``nisa.nc_transpose``.

Swaps the partition and free axes of a tensor. Takes a ``(P, F)``
operand and produces an ``(F, P)`` output. Executes on Tensor Engine
by default (Vector Engine is a 32×32 fallback); the caller pays TE
cycles — contrast with :class:`NKIDMATranspose` which runs on the
DMA engine and leaves TE free for matmul.
"""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKITranspose(NKIOp):
    """Transpose ``data(P, F) -> dst(F, P)`` on Tensor Engine."""

    NAME: ClassVar[str] = "nc_transpose"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"data": ("P", "F"), "dst": ("F", "P")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    """Tensor Engine caps the input at 128×128; Vector Engine at 32×32.
    We target Tensor Engine, so both axes are capped at 128."""
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": 128}
    OUTPUT_ROLE: ClassVar[str] = "psum"
    OUTPUT_LOCATION: ClassVar[str] = "psum"

    def _check_roles(self, **kwargs: Any) -> None:
        """``data`` must be SBUF-resident."""
        role = _operand_role(kwargs["data"])
        if role is not None and role != "sbuf":
            raise TypeError(f"NKITranspose(data=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate and return ``data.T``."""
        return np.array(kwargs["data"]).T
