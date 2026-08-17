"""Fused activation and free-axis reduction."""

from collections.abc import Mapping
from numbers import Real
from typing import Any, ClassVar, Literal

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, ReductionContract, _operand_role, reduction_combinator

VE_PARTITION_MAX = 128
VE_FREE_MAX = 512

_ACT_FNS: dict[str, Any] = {
    "square": np.square,
    "exp": np.exp,
    "copy": lambda x: x,
    "reciprocal": lambda x: 1.0 / x,
    "tanh": np.tanh,
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "sqrt": np.sqrt,
}
_RED_FNS: dict[str, Any] = {"add": np.sum, "max": np.max}


class NKIActivationReduce(NKIOp):
    """Apply an activation and reduce the result along the free axis."""

    NAME: ClassVar[str] = "activation_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("P", "F"),
        "bias": ("P",),
        "dst": ("P", "F"),
        "reduce_res": ("P",),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "bias"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {
        "data": frozenset({"sbuf", "psum"}),
        "bias": frozenset({"sbuf", "psum"}),
    }
    RFACTOR_RECIPE: ClassVar[Literal["rmw", "slot"] | None] = "slot"
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    PREFERRED_TILE_SIZE: ClassVar[dict[str, int]] = {"F": 512}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> ReductionContract:
        """Return the configured mapped free-axis reduction."""
        return ReductionContract(
            input_operand="data",
            output_operand="reduce_res",
            reduction_axis="F",
            combinator=reduction_combinator(str(kwargs["reduce_op"])),
            map_operator=str(kwargs["op"]),
            scale=float(kwargs.get("scale", 1.0)),
            bias=float(kwargs["bias"]) if isinstance(kwargs.get("bias"), Real) else 0.0,
            bias_operand="bias",
            mapped_output_operand="dst",
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require on-chip data and optional broadcast bias."""
        data_role = _operand_role(kwargs["data"])
        if data_role is not None and data_role not in {"sbuf", "psum"}:
            raise TypeError(f"NKIActivationReduce(data=<role={data_role}>) expects sbuf or psum")
        bias_role = _operand_role(kwargs.get("bias"))
        if bias_role is not None and bias_role not in {"sbuf", "psum"}:
            raise TypeError(f"NKIActivationReduce(bias=<role={bias_role}>) expects sbuf or psum")

    def _run(self, **kwargs: Any) -> Any:
        """Return ``reduce_op(op(data), axis=F)`` for CPU simulation."""
        allowed = {"data", "op", "reduce_op", "scale", "bias"}
        extra = set(kwargs) - allowed
        if extra:
            raise TypeError(
                f"NKIActivationReduce received unexpected kwargs: {sorted(extra)}. "
                f"Only {sorted(allowed)} are supported; use a separate NKIActivation "
                f"for post-reduction operations."
            )
        data: np.ndarray = kwargs["data"]
        op_name: str = kwargs["op"]
        reduce_op: str = kwargs["reduce_op"]
        scale = kwargs.get("scale", 1.0)
        bias = kwargs.get("bias", 0.0)
        if isinstance(scale, np.ndarray):
            scale = scale[..., np.newaxis]
        if isinstance(bias, np.ndarray):
            bias = bias[..., np.newaxis]
        activated = _ACT_FNS[op_name](data.astype(np.float32) * scale + bias)
        return _RED_FNS[reduce_op](activated, axis=1).astype(np.float32)
