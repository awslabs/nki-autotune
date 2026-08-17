"""Fused tensor-scalar map and free-axis reduction.

This mirrors ``nisa.tensor_scalar_reduce``. The instruction writes both the
mapped tile and one per-partition reduction result.
"""

from collections.abc import Mapping
from numbers import Real
from typing import Any, ClassVar, Literal

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, ReductionContract, _operand_role, reduction_combinator

_OPS: dict[str, Any] = {"add": np.add, "multiply": np.multiply, "subtract": np.subtract}
_REDUCE_FNS: dict[str, Any] = {"add": np.sum, "max": np.max, "maximum": np.max}


class NKITensorScalarReduce(NKIOp):
    """Apply one tensor-scalar operation and reduce its mapped result."""

    NAME: ClassVar[str] = "tensor_scalar_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("P", "F"),
        "operand0": ("P",),
        "dst": ("P", "F"),
        "reduce_res": ("P",),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "operand0"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {
        "data": frozenset({"sbuf", "psum"}),
        "operand0": frozenset({"sbuf"}),
    }
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"operand0": "float32"}
    RFACTOR_RECIPE: ClassVar[Literal["rmw", "slot"] | None] = "slot"
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    PREFERRED_TILE_SIZE: ClassVar[dict[str, int]] = {"F": 512}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> ReductionContract:
        """Return the affine map followed by the configured reduction."""
        operator = str(kwargs["op0"])
        operand = kwargs.get("operand0")
        reverse = bool(kwargs.get("reverse0", False))
        map_operator = operator
        scale = 1.0
        bias = 0.0
        if isinstance(operand, Real):
            scalar = float(operand)
            if operator == "multiply":
                map_operator = "copy"
                scale = scalar
            elif operator == "add":
                map_operator = "copy"
                bias = scalar
            elif operator == "subtract":
                map_operator = "copy"
                if reverse:
                    scale = -1.0
                    bias = scalar
                else:
                    bias = -scalar
        return ReductionContract(
            input_operand="data",
            output_operand="reduce_res",
            reduction_axis="F",
            combinator=reduction_combinator(str(kwargs["reduce_op"])),
            map_operator=map_operator,
            scale=scale,
            bias=bias,
            mapped_output_operand="dst",
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require on-chip data and allow either an SBUF or literal scalar."""
        data_role = _operand_role(kwargs["data"])
        if data_role is not None and data_role not in {"sbuf", "psum"}:
            raise TypeError(f"NKITensorScalarReduce(data=<role={data_role}>) expects sbuf or psum")
        operand_role = _operand_role(kwargs.get("operand0"))
        if operand_role is not None and operand_role != "sbuf":
            raise TypeError(f"NKITensorScalarReduce(operand0=<role={operand_role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> Any:
        """Return the reduced mapped tile for CPU simulation."""
        data = np.asarray(kwargs["data"])
        operand = kwargs["operand0"]
        broadcast = operand[..., np.newaxis] if isinstance(operand, np.ndarray) else operand
        operands = (broadcast, data) if kwargs.get("reverse0", False) else (data, broadcast)
        mapped = _OPS[str(kwargs["op0"])](*operands)
        reduced = _REDUCE_FNS[str(kwargs["reduce_op"])](mapped, axis=1)
        return np.asarray(reduced, dtype=np.float32)


__all__ = ["NKITensorScalarReduce"]
