"""Vector-engine ``nisa.tensor_scalar_cumulative`` operation."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, _operand_role

_OPS = {"add": np.add, "multiply": np.multiply, "subtract": np.subtract}


class NKITensorScalarCumulative(NKIOp):
    """Apply scalar arithmetic followed by a free-axis cumulative reduction."""

    NAME: ClassVar[str] = "tensor_scalar_cumulative"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf", "psum"})}
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.SEQUENTIAL}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"F"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    PREFERRED_TILE_SIZE: ClassVar[dict[str, int]] = {"F": 2048}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one on-chip input tensor."""
        role = _operand_role(kwargs["src"])
        if role is not None and role not in {"sbuf", "psum"}:
            raise TypeError(f"NKITensorScalarCumulative expects an on-chip input, got role={role}")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Simulate scalar arithmetic followed by a cumulative reduction."""
        source = np.asarray(kwargs["src"], dtype=np.float32)
        transformed = _OPS[str(kwargs["op0"])](source, np.float32(kwargs["imm0"]))
        return _OPS[str(kwargs["op1"])].accumulate(transformed, axis=-1)


__all__ = ["NKITensorScalarCumulative"]
