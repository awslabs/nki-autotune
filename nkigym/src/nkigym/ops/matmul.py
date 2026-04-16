"""Matrix multiplication op: nisa.nc_matmul.

stationary(K, M).T @ moving(K, N) -> output(M, N).
Accumulates into PSUM in fp32 regardless of input dtype.
"""

from typing import ClassVar

import numpy as np

from nkigym.ops.base import NKIOp

MATMUL_FREE_MAX = 512


class NKIMatmul(NKIOp):
    """Matrix multiply: stationary.T @ moving -> output.

    Attributes:
        NAME: ``"nc_matmul"``.
        OPERAND_AXES: stationary is ``(K, M)``, moving is ``(K, N)``.
        OUTPUT_AXES: output is ``(M, N)``.
    """

    NAME: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"stationary": ("K", "M"), "moving": ("K", "N")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("M", "N")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset({"K"})
    TILE_LIMITS: ClassVar[dict[str, int]] = {"K": 128, "M": 128, "N": MATMUL_FREE_MAX}
    ISA_LOC: ClassVar[str] = "psum"
    PSUM_DTYPE: ClassVar[str | None] = "float32"
    INPUT_LOCS: ClassVar[dict[str, str]] = {"stationary": "sbuf", "moving": "sbuf"}

    def __call__(self, stationary: np.ndarray, moving: np.ndarray, **_: object) -> np.ndarray:
        """CPU simulation: stationary.T @ moving.

        Args:
            stationary: Array of shape (K, M).
            moving: Array of shape (K, N).

        Returns:
            Result array of shape (M, N).
        """
        return stationary.T @ moving

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Format nisa.nc_matmul(dst, stationary, moving)."""
        return f"nisa.nc_matmul({dst_expr}, {operand_exprs['stationary']}, {operand_exprs['moving']})"
