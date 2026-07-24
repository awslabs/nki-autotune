"""Matrix multiplication op for ``nisa.nc_matmul``.

``stationary(K, M).T @ moving(K, N) -> output(M, N)`` with fp32 PSUM
accumulation regardless of input dtype.
"""

from typing import Any, ClassVar, Literal

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, ReduceCombinator, _operand_role


class NKIMatmul(NKIOp):
    """Matrix multiply: ``stationary.T @ moving -> output``."""

    NAME: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {
        "stationary": ("K", "M"),
        "moving": ("K", "N"),
        "dst": ("M", "N"),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"stationary", "moving"})
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset({"dst"})
    RFACTOR_RECIPE: ClassVar[Literal["rmw", "slot"] | None] = "rmw"
    REDUCE_COMBINATOR: ClassVar[ReduceCombinator | None] = ReduceCombinator(combiner="add", identity=0.0)
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"K": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"K": 128, "M": 128, "N": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"K": 128, "M": 128, "N": 512}
    OUTPUT_ROLE: ClassVar[str] = "psum"
    OUTPUT_LOCATION: ClassVar[str] = "psum"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"

    def _check_roles(self, **kwargs: Any) -> None:
        """``stationary`` and ``moving`` must be SBUF-resident."""
        for slot in ("stationary", "moving"):
            role = _operand_role(kwargs[slot])
            if role is not None and role != "sbuf":
                raise TypeError(f"NKIMatmul({slot}=<role={role}>) expects sbuf; did you forget to load?")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate and return ``stationary.T @ moving`` at fp32."""
        return kwargs["stationary"].astype(np.float32).T @ kwargs["moving"].astype(np.float32)
