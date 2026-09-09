"""K=1 matmul used to broadcast one free-axis row."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import AxisRole, BilinearReductionContract, NKIOp, ReduceCombinator, _operand_role


class NKIMatmulBroadcast(NKIOp):
    """Broadcast ``moving(1, N)`` across ``M`` rows using a ones stationary tile."""

    NAME: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "stationary": ("K", "M"),
        "moving": ("K", "N"),
        "dst": ("M", "N"),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"stationary", "moving"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"K": 1}
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {
        "stationary": frozenset({"sbuf"}),
        "moving": frozenset({"sbuf"}),
    }
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset({"dst"})
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"K": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"K": 1, "M": 128, "N": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"K": 1, "M": 128, "N": 512}
    OUTPUT_LOCATION: ClassVar[str] = "psum"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> BilinearReductionContract:
        """Return the K=1 matrix-product contract."""
        _ = kwargs
        return BilinearReductionContract(
            left_operand="stationary",
            right_operand="moving",
            output_operand="dst",
            reduction_axis="K",
            combinator=ReduceCombinator(combiner="add", identity=0.0),
        )

    @classmethod
    def rmw_operands(cls, kwargs: Mapping[str, Any]) -> frozenset[str]:
        """Treat an explicit non-accumulating broadcast as overwrite."""
        return frozenset() if kwargs.get("accumulate") is False else cls.RMW_OPERANDS

    def _check_roles(self, **kwargs: Any) -> None:
        """Require SBUF inputs."""
        for operand in self.INPUT_OPERANDS:
            if (role := _operand_role(kwargs[operand])) is not None and role != "sbuf":
                raise TypeError(f"NKIMatmulBroadcast({operand}=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the fp32 matrix product."""
        return kwargs["stationary"].astype(np.float32).T @ kwargs["moving"].astype(np.float32)


__all__ = ["NKIMatmulBroadcast"]
