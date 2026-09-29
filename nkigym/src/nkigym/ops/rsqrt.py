"""Float32 reciprocal square roots on the GpSIMD engine."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from torch.fx import Node

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchValue, emit_activation
from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role
from nkigym.ops.folded_load import PACKING_MARKER
from nkigym.ops.partition_slice_load import prepare_packed_binary


class NKIRsqrt(NKIOp):
    """Evaluate reciprocal square root through one GpSIMD instruction."""

    NAME: ClassVar[str] = "tensor_scalar"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"sbuf"})}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"float32"})}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self) -> None:
        """Select the GpSIMD unary reciprocal-square-root instruction."""
        super().__init__(op0="rsqrt", operand0=0.0, engine="gpsimd")

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract:
        """Describe the unary mathematical operation."""
        _ = kwargs
        return PointwiseContract(operator="rsqrt", input_operands=("data",), output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require float32 input in SBUF."""
        if _operand_role(kwargs["data"]) not in {None, "sbuf"}:
            raise TypeError("GpSIMD reciprocal square root requires SBUF input")
        if np.asarray(kwargs["data"]).dtype != np.float32:
            raise TypeError("GpSIMD reciprocal square root requires float32 input")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the reciprocal square root with float32 arithmetic."""
        return np.reciprocal(np.sqrt(np.asarray(kwargs["data"], dtype=np.float32)))


def emit_native_sqrt(
    source: TorchValue, operation: str, name: str, scale: float, body: list[str], imports: set[str]
) -> TorchValue:
    """Compute a native square root without Scalar Engine range restrictions."""
    if operation != "sqrt":
        raise ValueError("native square-root lowering requires sqrt")
    if scale != 1.0:
        source = emit_activation(source, "copy", f"{name}_scaled", scale, body, imports)
    emit = TorchArithmetic(name, body, imports)
    data = source.name if source.storage_dtype == "float32" else emit.cast("NKIFloat32Cast", source.name)
    inverse = emit.emit("NKIRsqrt", f"data={data}")
    root = emit.emit("NKIReciprocal", f"data={inverse}")
    return TorchValue(root, source.shape, source.transposed, storage_dtype="float32")


def centered_variance_input(source: TorchValue, node: Node, emit: TorchArithmetic) -> tuple[TorchValue, TorchValue]:
    """Shift scaled rows by their first value before accumulating their variance."""
    logical = node.meta[PACKING_MARKER]
    rows, parts = int(np.prod(logical[:-1])), source.shape[0] // int(np.prod(logical[:-1]))
    maxima = emit.emit("NKITensorScalarReduce", f"data={source.name}, operand0=0.0", "op0='abs', reduce_op='max'")
    maxima = emit.emit("NKIVectorDMATranspose", f"src={maxima}")
    maxima = emit.emit(
        "NKIGroupedTileReduce", f"data={maxima}", f"partitions=1, groups={rows}, chunks={parts}, op='max'"
    )
    bits = emit.cast("NKIReinterpretUInt32", maxima)
    exponent = emit.cast("NKIFloat32Cast", emit.binary("right_shift", bits, 23))
    shift = emit.binary("maximum", emit.binary("subtract", exponent, 167), 0.0)
    inverse_bits = emit.binary("left_shift", emit.cast("NKIUInt32Cast", emit.binary("add", shift, 127)), 23)
    inverse = emit.cast("NKIReinterpretFloat32", inverse_bits)
    scale_bits = emit.binary("left_shift", emit.cast("NKIUInt32Cast", emit.scalar("subtract", shift, 127.0, True)), 23)
    scale = TorchValue(
        emit.cast("NKIReinterpretFloat32", scale_bits), (rows, 1), transposed=True, storage_dtype="float32"
    )
    _left, broadcast = prepare_packed_binary(source, scale, node, emit)
    if not isinstance(broadcast, TorchValue):
        raise TypeError("variance scaling requires a row tensor")
    scaled = emit.scalar("multiply", source.name, broadcast.name)
    first = emit.emit("NKIDMATranspose", f"src={emit.slice(scaled, 0, 1)}")
    first = emit.emit("NKIStridedTensorCopy", f"src={first}", f"pattern={((parts, rows),)!r}, offset=0")
    anchor = TorchValue(first, (rows, 1), transposed=True, storage_dtype="float32")
    _left, broadcast = prepare_packed_binary(source, anchor, node, emit)
    if not isinstance(broadcast, TorchValue):
        raise TypeError("variance centering requires a row tensor")
    centered = emit.scalar("subtract", scaled, broadcast.name)
    return (
        TorchValue(centered, source.shape, storage_dtype="float32"),
        TorchValue(inverse, (rows, 1), transposed=True, storage_dtype="float32"),
    )


__all__ = ["NKIRsqrt", "emit_native_sqrt"]
