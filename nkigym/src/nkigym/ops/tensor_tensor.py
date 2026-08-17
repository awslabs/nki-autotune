"""Elementwise tensor-tensor op: maps to ``nisa.tensor_tensor``.

Applies ``dst = data1 <op> data2`` over two same-shape ``(P, F)`` SBUF tensors.
RFactor's ``"rmw"`` write-back block uses this as the running combine
``out_sbuf = out_sbuf + B_rf[ko]`` by binding both ``data1`` and ``dst`` to
``out_sbuf``.
"""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role

_OPS: dict[str, Any] = {
    "add": np.add,
    "subtract": np.subtract,
    "multiply": np.multiply,
    "max": np.maximum,
    "maximum": np.maximum,
}


class NKITensorTensor(NKIOp):
    """Elementwise ``dst = data1 <op> data2`` over two ``(P, F)`` tensors."""

    NAME: ClassVar[str] = "tensor_tensor"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data1": ("P", "F"), "data2": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data1", "data2"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"data1": frozenset({"sbuf"}), "data2": frozenset({"sbuf"})}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract:
        """Return the configured binary pointwise operation."""
        return PointwiseContract(
            operator="maximum" if kwargs["op"] == "max" else str(kwargs["op"]),
            input_operands=("data1", "data2"),
            output_operand="dst",
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """``data1`` and ``data2`` must both be SBUF-resident."""
        for slot in ("data1", "data2"):
            role = _operand_role(kwargs[slot])
            if role is not None and role != "sbuf":
                raise TypeError(f"NKITensorTensor({slot}=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate and return ``data1 <op> data2`` elementwise."""
        return _OPS[kwargs["op"]](kwargs["data1"], kwargs["data2"])
