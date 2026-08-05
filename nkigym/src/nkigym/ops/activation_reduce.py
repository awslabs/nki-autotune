"""Fused activation + free-axis reduce op: mirrors ``nisa.activation_reduce``.

Math: ``reduce_op(op(data), axis=F)``. Output is the per-row reduction vector.
The fully activated tile is an internal byproduct the ISA call discards.

OPERAND_AXES reflects the valid ISA signature: ``data`` is the input
tile, ``dst`` receives the activated scratch (discarded by downstream
consumers), and ``reduce_res`` receives the per-row reduction result.
"""

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
    """Fused activation + free-axis reduce — mirrors ``nisa.activation_reduce``.

    Math: ``reduce_op(op(data), axis=F)``. Output is the ``(P,)`` per-row
    reduction vector. The fully activated ``(P, F)`` tile is an internal
    byproduct the gadget discards.

    Kwargs mirror the valid subset of ``nisa.activation_reduce``:

    * ``op``: activation applied per-element before the reduce.
    * ``reduce_op``: reduction operator along the free axis.

    Fused closures (e.g. rmsnorm's ``rsqrt(sum(x²)·scale + bias)``) must
    be spelled out in the DSL as a separate ``NKIActivation(op="rsqrt",
    scale=..., bias=...)`` call on the reduction output — not as
    ``post_op``/``scale``/``bias`` kwargs on this op.

    Lowering (Pattern 2): one ``nisa.activation_reduce`` per F-tile
    writes to a distinct slot of the op-local ``slot_vec`` buffer;
    after the F loop exits, one ``nisa.tensor_reduce(axis=2)`` folds
    ``slot_vec`` into the op's ``(P, 1)`` output.

    Future work: a hoist transform that pulls ``activation_reduce`` out
    of the F loop, combined with DCE on the closing ``tensor_reduce``
    (trivial when ``num_f_tiles == 1``), will reach Pattern 1 (one
    full-F ``activation_reduce`` call, no slot vector) monotonically.
    """

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
    """The F axis is a reduction axis — the op iterates over all F tiles
    before its output is complete. Render emits an F-loop memset prologue
    (on the reduce accumulator) and places downstream consumers outside
    this F-loop, symmetric to how matmul's K dim is handled."""
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
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
        """CPU simulation: allocate and return ``reduce_op(op(data), axis=F)``.

        Mirrors the valid ISA signature: configuration kwargs are
        ``{op, reduce_op}``; the operand kwarg is ``{data}``. Extra
        kwargs raise ``TypeError`` to keep the DSL honest. The activated
        ``(P, F)`` tile is an internal byproduct; the op's primary output
        is the ``(P,)`` per-row reduction vector.
        """
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
