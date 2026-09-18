"""Standalone vector-engine reciprocal."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchValue, emit_activation
from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role
from nkigym.ops.reinterpret_uint32 import emit_binary32_sqrt


def align_matmul(
    stationary: TorchValue, moving: TorchValue, body: list[str], imports: set[str]
) -> tuple[TorchValue, TorchValue]:
    """Promote either input when its peer requires the FP32 matmul format."""
    operands = [stationary, moving]
    fp32 = {"float32", "tfloat32"}
    if any(value.storage_dtype in fp32 for value in operands):
        for index, value in enumerate(operands):
            if value.storage_dtype not in fp32:
                target = TorchValue(
                    f"{value.name}_float32_{len(body)}", value.shape, value.transposed, storage_dtype="float32"
                )
                imports.add("NKIFloat32Cast")
                body.append(f"{target.name} = NKIFloat32Cast()(data={value.name})")
                operands[index] = target
    return operands[0], operands[1]


def matmul_target(
    name: str, shape: tuple[int, ...], stationary: TorchValue, moving: TorchValue
) -> tuple[TorchValue, bool]:
    """Choose the drain dtype required by one matmul input pair."""
    fp32 = not (moving.storage_dtype or "").startswith("float8")
    storage_dtype = "float32" if fp32 else "bfloat16"
    return TorchValue(name, shape, storage_dtype=storage_dtype), fp32


def emit_activation_or_reciprocal(
    source: TorchValue, operation: str, name: str, scale: float, body: list[str], imports: set[str]
) -> TorchValue:
    """Emit the native reciprocal op or a regular activation."""
    if operation == "gelu":
        return _emit_exact_gelu(source, name, body, imports)
    if operation == "sqrt":
        if scale != 1.0:
            source = emit_activation(source, "copy", f"{name}_scaled", scale, body, imports)
        return emit_binary32_sqrt(source, name, body, imports)
    if operation != "reciprocal":
        return emit_activation(source, operation, name, scale, body, imports)
    target = TorchValue(name, source.shape, source.transposed, storage_dtype="float32")
    imports.add("NKIReciprocal")
    body.append(f"{target.name} = NKIReciprocal()(data={source.name})")
    return target


def _emit_exact_gelu(source: TorchValue, name: str, body: list[str], imports: set[str]) -> TorchValue:
    """Emit the erf definition of GELU in float32 before physical rounding."""
    fp32 = TorchValue(f"{name}_fp32", source.shape, source.transposed, storage_dtype="float32")
    scaled = TorchValue(f"{name}_scaled", source.shape, source.transposed, storage_dtype="float32")
    transformed = TorchValue(f"{name}_erf", source.shape, source.transposed, storage_dtype="float32")
    shifted = TorchValue(f"{name}_shifted", source.shape, source.transposed, storage_dtype="float32")
    product = TorchValue(f"{name}_product", source.shape, source.transposed, storage_dtype="float32")
    target = TorchValue(name, source.shape, source.transposed, storage_dtype="float32")
    imports.update(("NKIActivation", "NKIFloat32Cast", "NKITensorScalar", "NKITensorTensor"))
    body.extend(
        (
            f"{fp32.name} = NKIFloat32Cast()(data={source.name})",
            f'{scaled.name} = NKIActivation(op="copy", scale=0.7071067811865476)(data={fp32.name})',
            f'{transformed.name} = NKIActivation(op="erf")(data={scaled.name})',
            f'{shifted.name} = NKITensorScalar(op0="add")(data={transformed.name}, operand0=1.0)',
            f'{product.name} = NKITensorTensor(op="multiply")(data1={fp32.name}, data2={shifted.name})',
            f'{target.name} = NKIActivation(op="copy", scale=0.5)(data={product.name})',
        )
    )
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
