"""Tensor-scalar ``nisa.tensor_scalar`` operation.

Applies ``output = data <op> operand0`` where ``operand0`` is either a
compile-time scalar or a per-partition ``(P,)`` vector broadcast along
the free axis. The rmsnorm+matmul example uses the per-partition vector
form to multiply ``(d0, d1)`` lhs tiles by the 1D rsqrt result.
"""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role

VE_PARTITION_MAX = 128
VE_FREE_MAX = 512

_OPS: dict[str, Any] = {
    "multiply": np.multiply,
    "add": np.add,
    "subtract": np.subtract,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "divide": np.divide,
    "greater_equal": lambda left, right: np.greater_equal(left, right).astype(np.float32),
    "less": lambda left, right: np.less(left, right).astype(np.float32),
}


class NKITensorScalar(NKIOp):
    """Elementwise ``output = data <op> operand0`` with broadcast operand.

    ``operand0`` may be a compile-time literal (not captured as a
    tensor input) or a 1D ``(P,)`` vector that broadcasts across the
    free axis of ``data``.
    """

    NAME: ClassVar[str] = "tensor_scalar"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "operand0": ("P",), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "operand0"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {
        "data": frozenset({"sbuf", "psum"}),
        "operand0": frozenset({"sbuf", "psum"}),
    }
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"operand0": "float32"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract:
        """Return the configured binary pointwise operation."""
        return PointwiseContract(
            operator=str(kwargs["op0"]),
            input_operands=("data", "operand0"),
            output_operand="dst",
            broadcast_operands=frozenset({"operand0"}),
            reverse=bool(kwargs.get("reverse0", False)),
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Tensor inputs may be SBUF- or PSUM-resident."""
        data_role = _operand_role(kwargs["data"])
        if data_role is not None and data_role not in {"sbuf", "psum"}:
            raise TypeError(f"NKITensorScalar(data=<role={data_role}>) expects sbuf or psum")
        operand0_role = _operand_role(kwargs.get("operand0"))
        if operand0_role is not None and operand0_role not in {"sbuf", "psum"}:
            raise TypeError(f"NKITensorScalar(operand0=<role={operand0_role}>) expects sbuf or psum")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: broadcast ``operand0`` across F, apply ``op``, return the result."""
        data = kwargs["data"]
        operand0 = kwargs["operand0"]
        broadcast = (
            operand0[..., np.newaxis]
            if isinstance(operand0, np.ndarray) and operand0.ndim + 1 == data.ndim
            else operand0
        )
        operands = (broadcast, data) if kwargs.get("reverse0", False) else (data, broadcast)
        return _OPS[kwargs["op0"]](*operands)
