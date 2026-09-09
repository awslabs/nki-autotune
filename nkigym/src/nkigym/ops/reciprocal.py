"""Standalone vector-engine reciprocal."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchValue, emit_activation
from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role


def emit_activation_or_reciprocal(
    source: TorchValue, operation: str, name: str, scale: float, body: list[str], imports: set[str]
) -> TorchValue:
    """Emit the native reciprocal op or a regular activation."""
    if operation != "reciprocal":
        return emit_activation(source, operation, name, scale, body, imports)
    target = TorchValue(name, source.shape, source.transposed, storage_dtype="float32")
    imports.add("NKIReciprocal")
    body.append(f"{target.name} = NKIReciprocal()(data={source.name})")
    return target


class NKIReciprocal(NKIOp):
    """Compute an elementwise fp32 reciprocal using ``nisa.reciprocal``."""

    NAME: ClassVar[str] = "reciprocal"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"sbuf", "psum"})}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract:
        """Return the reciprocal pointwise contract."""
        _ = kwargs
        return PointwiseContract(operator="reciprocal", input_operands=("data",), output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one on-chip input."""
        role = _operand_role(kwargs["data"])
        if role is not None and role not in {"sbuf", "psum"}:
            raise TypeError(f"NKIReciprocal(data=<role={role}>) expects sbuf or psum")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the fp32 reciprocal."""
        return np.reciprocal(np.asarray(kwargs["data"], dtype=np.float32))


__all__ = ["NKIReciprocal"]
