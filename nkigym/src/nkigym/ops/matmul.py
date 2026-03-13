"""Matrix multiplication: dst = stationary.T @ moving."""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from nkigym.ir.tensor import TensorRef
from nkigym.ops.base import NKIOp, _render_ref


@dataclass(frozen=True)
class NKIMatmul(NKIOp):
    """Matrix multiply: ``nisa.nc_matmul(dst=ref, stationary=ref, moving=ref)``.

    Attributes:
        dst: PSUM destination reference.
        stationary: Stationary operand reference [K, M].
        moving: Moving operand reference [K, N].
    """

    dst: TensorRef
    stationary: TensorRef
    moving: TensorRef

    op_name: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"stationary": ("K", "M"), "moving": ("K", "N")}
    OUTPUT_AXES: ClassVar[tuple[str, str]] = ("M", "N")
    TILE_LIMITS: ClassVar[dict[str, int]] = {"K": 128, "M": 128, "N": 512}

    def mac_count(self) -> int:
        """Count MACs: K * M * N for stationary [K, M] @ moving [K, N].

        Returns:
            Number of multiply-accumulate operations.
        """
        k, m = self.stationary.shape
        _, n = self.moving.shape
        return k * m * n

    def simulate(self, env: dict[str, Any]) -> None:
        """Accumulate stationary.T @ moving into dst.

        Args:
            env: Mutable variable environment.
        """
        stat = env[self.stationary.name][self.stationary.to_slices()]
        mov = env[self.moving.name][self.moving.to_slices()]
        env[self.dst.name][self.dst.to_slices()] += np.matmul(stat.T, mov)

    def render(self) -> str:
        """Render as NKI matmul statement.

        Returns:
            NKI source line.
        """
        dst = _render_ref(self.dst)
        stat = _render_ref(self.stationary)
        mov = _render_ref(self.moving)
        return f"nisa.nc_matmul(dst={dst}, stationary={stat}, moving={mov})"
