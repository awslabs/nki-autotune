"""Free-axis reduction op: maps to ``nisa.tensor_reduce``."""

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, ReductionContract, _operand_role, reduction_combinator

_REDUCE_FNS: dict[str, Any] = {"add": np.sum, "max": np.max, "maximum": np.max, "multiply": np.prod}


class NKITensorReduce(NKIOp):
    """Reduce ``data`` along an axis into ``dst``.

    kwargs:
        axis: ``int`` — the axis of ``data`` to reduce over.
        op: ``"add"`` or ``"max"``.
    operands:
        data: source tensor.
        dst: destination tensor — shape equals ``data.shape`` with ``axis`` removed.
    """

    NAME: ClassVar[str] = "tensor_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "dst": ("P",)}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"sbuf", "psum"})}
    RFACTOR_RECIPE: ClassVar[Literal["rmw", "slot"] | None] = "slot"
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    PREFERRED_TILE_SIZE: ClassVar[dict[str, int]] = {"F": 512}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> ReductionContract:
        """Return the configured free-axis reduction."""
        return ReductionContract(
            input_operand="data",
            output_operand="dst",
            reduction_axis="F",
            combinator=reduction_combinator(str(kwargs["op"])),
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """``data`` must be SBUF-resident."""
        role = _operand_role(kwargs["data"])
        if role is not None and role not in {"sbuf", "psum"}:
            raise TypeError(f"NKITensorReduce(data=<role={role}>) expects sbuf or psum")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate and return the numpy reduction along axis."""
        result = _REDUCE_FNS[kwargs["op"]](kwargs["data"], axis=kwargs["axis"])
        return np.asarray(result)
