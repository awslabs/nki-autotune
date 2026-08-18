"""Select one dynamic range and reduce the selected tile."""

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, ReductionContract, _operand_role, reduction_combinator
from nkigym.ops.range_select import _COMPARISONS, NKIRangeSelect


class NKIRangeSelectReduce(NKIOp):
    """Select elements and compute their maximum in one ISA instruction."""

    NAME: ClassVar[str] = "range_select"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "on_true_tile": ("P", "F"),
        "bound0": ("P",),
        "bound1": ("P",),
        "dst": ("P", "F"),
        "reduce_res": ("P",),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"on_true_tile", "bound0", "bound1"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {
        "on_true_tile": frozenset({"sbuf", "psum"}),
        "bound0": frozenset({"sbuf", "psum"}),
        "bound1": frozenset({"sbuf", "psum"}),
    }
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {"on_true_tile": frozenset({"float32"})}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"bound0": "float32", "bound1": "float32"}
    RFACTOR_RECIPE: ClassVar[Literal["rmw", "slot"] | None] = "slot"
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    PREFERRED_TILE_SIZE: ClassVar[dict[str, int]] = {"F": 512}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"width"})
    SPLIT_OFFSET_KWARGS: ClassVar[dict[str, tuple[str, str]]] = {"F": ("range_start", "dst")}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, width: int, **kwargs: Any) -> None:
        """Configure the complete implicit index width."""
        super().__init__(width=width, **kwargs)

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> ReductionContract:
        """Return the masked maximum reduction contract."""
        return ReductionContract(
            input_operand="on_true_tile",
            output_operand="reduce_res",
            reduction_axis="F",
            combinator=reduction_combinator(str(kwargs["reduce_op"])),
            map_operator="range_select",
            mapped_output_operand="dst",
            mapped_op_cls=NKIRangeSelect,
            mapped_input_operands=("on_true_tile", "bound0", "bound1"),
            mapped_excluded_kwargs=frozenset({"reduce_cmd", "reduce_op"}),
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require local data and bound tiles."""
        for slot in ("on_true_tile", "bound0", "bound1"):
            role = _operand_role(kwargs[slot])
            if role is not None and role not in {"sbuf", "psum"}:
                raise TypeError(f"NKIRangeSelectReduce({slot}=<role={role}>) expects sbuf or psum")
        if np.asarray(kwargs["on_true_tile"]).shape[1] != int(kwargs["width"]):
            raise ValueError("NKIRangeSelectReduce width must match the complete implicit index domain")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Apply the range predicate and return its partition-wise maximum."""
        data = np.asarray(kwargs["on_true_tile"])
        indices = np.arange(data.shape[1], dtype=np.float32)[None, :] + int(kwargs.get("range_start", 0))
        bound0, bound1 = (np.asarray(kwargs[name]).reshape(-1, 1) for name in ("bound0", "bound1"))
        lower = _COMPARISONS[str(kwargs["comp_op0"])](indices, bound0)
        upper = _COMPARISONS[str(kwargs["comp_op1"])](indices, bound1)
        selected = np.where(lower & upper, data, kwargs.get("on_false_value", np.finfo(np.float32).min))
        return np.max(selected, axis=1)


__all__ = ["NKIRangeSelectReduce"]
