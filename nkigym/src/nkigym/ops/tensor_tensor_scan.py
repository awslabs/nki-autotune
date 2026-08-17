"""Vector-engine ``nisa.tensor_tensor_scan`` operation."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, _operand_role

_OPS = {"add": np.add, "multiply": np.multiply, "subtract": np.subtract}


class NKITensorTensorScan(NKIOp):
    """Apply a programmable binary recurrence along the free axis."""

    NAME: ClassVar[str] = "tensor_tensor_scan"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data0": ("P", "F"), "data1": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data0", "data1"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {
        "data0": frozenset({"sbuf", "psum"}),
        "data1": frozenset({"sbuf", "psum"}),
    }
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.SEQUENTIAL}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"F"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    @classmethod
    def accepts_input_locations(cls, locations: Mapping[str, str]) -> bool:
        """Reject the unsupported two-PSUM input combination."""
        roles = (locations.get("data0"), locations.get("data1"))
        return super().accepts_input_locations(locations) and roles != ("psum", "psum")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require on-chip inputs and reject two PSUM operands."""
        roles = tuple(_operand_role(kwargs[name]) for name in ("data0", "data1"))
        if any(role is not None and role not in {"sbuf", "psum"} for role in roles):
            raise TypeError(f"NKITensorTensorScan expects on-chip inputs, got roles={roles}")
        if roles == ("psum", "psum"):
            raise TypeError("NKITensorTensorScan inputs cannot both reside in psum")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Simulate the configured recurrence in fp32."""
        data0 = np.asarray(kwargs["data0"], dtype=np.float32)
        data1 = np.asarray(kwargs["data1"], dtype=np.float32)
        flat0 = data0.reshape(data0.shape[0], -1)
        flat1 = data1.reshape(data1.shape[0], -1)
        initial = np.broadcast_to(np.asarray(kwargs["initial"], dtype=np.float32), (data0.shape[0],))
        result = np.empty_like(flat0)
        op0 = _OPS[str(kwargs["op0"])]
        op1 = _OPS[str(kwargs["op1"])]
        previous = initial
        for index in range(flat0.shape[1]):
            first_args = (previous, flat0[:, index]) if kwargs.get("reverse0", False) else (flat0[:, index], previous)
            first = op0(*first_args)
            second_args = (flat1[:, index], first) if kwargs.get("reverse1", False) else (first, flat1[:, index])
            previous = op1(*second_args)
            result[:, index] = previous
        return result.reshape(data0.shape)


__all__ = ["NKITensorTensorScan"]
