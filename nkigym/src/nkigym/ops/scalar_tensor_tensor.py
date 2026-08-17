"""Fused scalar-tensor-tensor op: maps to ``nisa.scalar_tensor_tensor``."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, PointwiseSequenceContract, _operand_role

_OPS: dict[str, Any] = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply, "maximum": np.maximum}


class NKIScalarTensorTensor(NKIOp):
    """Compute ``dst = (data <op0> operand0) <op1> operand1``.

    ``operand0`` may be a compile-time scalar or a per-partition vector.
    ``operand1`` has the same shape as ``data``. This is the native Vector
    Engine recurrence instruction used by online accumulation.
    """

    NAME: ClassVar[str] = "scalar_tensor_tensor"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("P", "F"),
        "operand0": ("P",),
        "operand1": ("P", "F"),
        "dst": ("P", "F"),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "operand0", "operand1"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {
        operand: frozenset({"sbuf", "psum"}) for operand in INPUT_OPERANDS
    }
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseSequenceContract:
        """Return the configured two-step pointwise contract."""
        return PointwiseSequenceContract(
            operators=(str(kwargs["op0"]), str(kwargs["op1"])),
            input_operands=("data", "operand0", "operand1"),
            output_operand="dst",
            broadcast_operands=frozenset({"operand0"}),
            reverse=(bool(kwargs.get("reverse0", False)), bool(kwargs.get("reverse1", False))),
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Validate the ISA's SBUF/PSUM operand constraints."""
        data_role = _operand_role(kwargs["data"])
        operand0_role = _operand_role(kwargs["operand0"])
        operand1_role = _operand_role(kwargs["operand1"])
        if data_role is not None and data_role not in {"sbuf", "psum"}:
            raise TypeError(f"NKIScalarTensorTensor(data=<role={data_role}>) expects sbuf or psum")
        if operand0_role is not None and operand0_role not in {"sbuf", "psum"}:
            raise TypeError(f"NKIScalarTensorTensor(operand0=<role={operand0_role}>) expects sbuf or psum")
        if operand1_role is not None and operand1_role not in {"sbuf", "psum"}:
            raise TypeError(f"NKIScalarTensorTensor(operand1=<role={operand1_role}>) expects sbuf or psum")
        if data_role == "psum" and operand1_role == "psum":
            raise TypeError("NKIScalarTensorTensor data and operand1 cannot both reside in psum")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation for the two configured binary operations."""
        data = kwargs["data"]
        operand0 = kwargs["operand0"]
        operand1 = kwargs["operand1"]
        broadcast0 = (
            operand0[..., np.newaxis]
            if isinstance(operand0, np.ndarray) and operand0.ndim + 1 == data.ndim
            else operand0
        )
        first_args = (broadcast0, data) if kwargs.get("reverse0", False) else (data, broadcast0)
        intermediate = _OPS[str(kwargs["op0"])](*first_args)
        second_args = (operand1, intermediate) if kwargs.get("reverse1", False) else (intermediate, operand1)
        return _OPS[str(kwargs["op1"])](*second_args)
