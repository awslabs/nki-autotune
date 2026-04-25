"""Matrix multiplication op: ``nisa.nc_matmul``.

``stationary(K, M).T @ moving(K, N) -> output(M, N)``. Accumulates into
PSUM at fp32 regardless of input dtype.
"""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp

MATMUL_FREE_MAX = 512


class NKIMatmul(NKIOp):
    """Matrix multiply: ``stationary.T @ moving -> output``."""

    NAME: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"stationary": ("K", "M"), "moving": ("K", "N")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("M", "N")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset({"K"})
    TILE_LIMITS: ClassVar[dict[str, int]] = {"K": 128, "M": 128, "N": MATMUL_FREE_MAX}

    def __call__(self, **kwargs: Any) -> Any:
        """CPU simulation: ``stationary.T @ moving``."""
        stationary: np.ndarray = kwargs["stationary"]
        moving: np.ndarray = kwargs["moving"]
        return stationary.T @ moving
