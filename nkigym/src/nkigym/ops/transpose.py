"""Transpose op: nisa.nc_transpose.

nc_transpose(P, F) -> output(F, P).
Tensor Engine: reads from SBUF, writes to PSUM. Max tile 128x128.
"""

from typing import ClassVar

import numpy as np

from nkigym.ops.base import NKIOp

TRANSPOSE_BLOCK = 128


class NKITranspose(NKIOp):
    """Transpose: data(P, F) -> output(F, P).

    Tensor Engine nc_transpose reads from SBUF, writes to PSUM,
    then tensor_copy moves the result to SBUF. Max tile is 128x128.

    Attributes:
        NAME: ``"nc_transpose"``.
        OPERAND_AXES: data is ``(P, F)``.
        OUTPUT_AXES: output is ``(F, P)``.
    """

    NAME: ClassVar[str] = "nc_transpose"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("F", "P")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": TRANSPOSE_BLOCK, "F": TRANSPOSE_BLOCK}
    ISA_LOC: ClassVar[str] = "psum"
    PSUM_DTYPE: ClassVar[str | None] = None
    INPUT_LOCS: ClassVar[dict[str, str]] = {"data": "sbuf"}

    def __call__(self, data: np.ndarray, **_: object) -> np.ndarray:
        """CPU simulation: data.T.

        Args:
            data: Array of shape (P, F).

        Returns:
            Transposed array of shape (F, P).
        """
        return data.T

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Format nisa.nc_transpose(dst, data)."""
        return f"nisa.nc_transpose({dst_expr}, {operand_exprs['data']})"
