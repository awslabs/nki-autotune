"""Float32 activation scaling and compensated scalar division emission."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role


def emit_tensor_scalar_or_divide(
    source: TorchValue, operation: str, operand: str, reverse: bool, name: str, body: list[str], imports: set[str]
) -> TorchValue:
    """Emit a regular tensor-scalar op or compensated constant division."""
    target = TorchValue(name, source.shape, source.transposed, storage_dtype="float32")
    if operation != "divide":
        target = TorchValue(name, source.shape, source.transposed)
        reverse_argument = ", reverse0=True" if reverse and operation == "subtract" else ""
        imports.add("NKITensorScalar")
        body.append(
            f'{target.name} = NKITensorScalar(op0="{operation}"{reverse_argument})'
            f"(data={source.name}, operand0={operand})"
        )
        return target
    if reverse:
        raise ValueError("Torch scalar-over-tensor division is unsupported")
    divisor = float(operand)
    if divisor == 0.0:
        raise ValueError("Torch tensor division requires a nonzero scalar")
    scale = 1.0 / divisor
    approximate = TorchValue(f"{name}_approximate", source.shape, source.transposed, storage_dtype="float32")
    residual = TorchValue(f"{name}_residual", source.shape, source.transposed, storage_dtype="float32")
    imports.update(("NKIFloat32Scale", "NKIScalarTensorTensor"))
    body.extend(
        (
            f"{approximate.name} = NKIFloat32Scale(scale={scale!r})(data={source.name})",
            f'{residual.name} = NKIScalarTensorTensor(op0="multiply", op1="subtract", reverse1=True)'
            f"(data={approximate.name}, operand0={divisor!r}, operand1={source.name})",
            f'{target.name} = NKIScalarTensorTensor(op0="multiply", op1="add")'
            f"(data={residual.name}, operand0={scale!r}, operand1={approximate.name})",
        )
    )
    return target


class NKIFloat32Scale(NKIOp):
    """Scale one on-chip tensor into a float32 activation destination."""

    NAME: ClassVar[str] = "activation"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"sbuf", "psum"})}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"bfloat16", "float16", "float32"})}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"

    def __init__(self, scale: float) -> None:
        """Configure a native copy activation with one scalar scale."""
        super().__init__(op="copy", scale=scale)

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract:
        """Return the activation-scale pointwise contract."""
        return PointwiseContract(
            operator="copy", input_operands=("data",), output_operand="dst", scale=float(kwargs["scale"])
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one on-chip input."""
        role = _operand_role(kwargs["data"])
        if role is not None and role not in {"sbuf", "psum"}:
            raise TypeError(f"NKIFloat32Scale(data=<role={role}>) expects sbuf or psum")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the float32 scaled values."""
        return np.asarray(kwargs["data"], dtype=np.float32) * float(kwargs["scale"])


__all__ = ["NKIFloat32Scale"]
