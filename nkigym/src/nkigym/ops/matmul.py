"""Matrix multiplication op for ``nisa.nc_matmul``.

``stationary(K, M).T @ moving(K, N) -> output(M, N)`` with fp32 PSUM
accumulation regardless of input dtype.
"""

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import numpy as np

from nkigym.ops.base import AxisRole, BilinearReductionContract, NKIOp, ReduceCombinator, _operand_role


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

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> BilinearReductionContract:
        """Return the matrix product's bilinear sum-reduction contract."""
        _ = kwargs
        return BilinearReductionContract(
            left_operand="stationary",
            right_operand="moving",
            output_operand="dst",
            reduction_axis="K",
            combinator=ReduceCombinator(combiner="add", identity=0.0),
        )

    @classmethod
    def first_write_overwrites(cls, operand: str, kwargs: Mapping[str, Any]) -> bool:
        """Return the configured PSUM first-write behavior."""
        return operand == "dst" and kwargs.get("accumulate") is not True

    @classmethod
    def rmw_operands(cls, kwargs: Mapping[str, Any]) -> frozenset[str]:
        """Treat an explicit first matmul as a write-only destination."""
        return frozenset() if kwargs.get("accumulate") is False else cls.RMW_OPERANDS

    @classmethod
    def with_first_write_overwrite(cls, operand: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Return an explicit first-matmul configuration."""
        if operand != "dst" or kwargs.get("accumulate") is True:
            raise ValueError(f"NKIMatmul.{operand} does not support first-write overwrite")
        return {**kwargs, "accumulate": False}

    def _check_roles(self, **kwargs: Any) -> None:
        """``stationary`` and ``moving`` must be SBUF-resident."""
        for slot in ("stationary", "moving"):
            role = _operand_role(kwargs[slot])
            if role is not None and role != "sbuf":
                raise TypeError(f"NKIMatmul({slot}=<role={role}>) expects sbuf; did you forget to load?")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate and return ``stationary.T @ moving`` at fp32."""
        return kwargs["stationary"].astype(np.float32).T @ kwargs["moving"].astype(np.float32)
