"""Elementwise tensor-tensor op: maps to ``nisa.tensor_tensor``.

Applies ``dst = data1 <op> data2`` over two same-shape ``(P, F)`` tensors.
RFactor uses this as a running combine with one input aliased to ``dst``.
A PSUM partial occupies ``data1`` and the SBUF accumulator occupies ``data2``.
"""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from torch.fx import Node

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import NKIOp, PartitionTileBatchingContract, PointwiseContract, _operand_role
from nkigym.ops.folded_load import PACKING_MARKER, _operation
from nkigym.ops.grouped_query_reduce import packed_maximum
from nkigym.ops.partition_slice_load import prepare_packed_binary
from nkigym.ops.rsqrt import centered_variance_input
from nkigym.ops.vector_dma_transpose import emit_packed_reduction

_OPS: dict[str, Any] = {
    "add": np.add,
    "subtract": np.subtract,
    "multiply": np.multiply,
    "max": np.maximum,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "greater": lambda left, right: np.greater(left, right).astype(np.float32),
    "greater_equal": lambda left, right: np.greater_equal(left, right).astype(np.float32),
    "equal": lambda left, right: np.equal(left, right).astype(np.float32),
    "not_equal": lambda left, right: np.not_equal(left, right).astype(np.float32),
}


class NKITensorTensor(NKIOp):
    """Elementwise ``dst = data1 <op> data2`` over two ``(P, F)`` tensors."""

    NAME: ClassVar[str] = "tensor_tensor"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data1": ("P", "F"), "data2": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data1", "data2"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {
        "data1": frozenset({"sbuf", "psum"}),
        "data2": frozenset({"sbuf"}),
    }
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {
        operand: frozenset({"bfloat16", "float16", "float32"}) for operand in ("data1", "data2")
    }
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    TENSORIZE_MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"F": 1}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    INPLACE_OPERANDS: ClassVar[dict[str, frozenset[str]]] = {"dst": frozenset({"data1", "data2"})}

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract:
        """Return the configured binary pointwise operation."""
        return PointwiseContract(
            operator="maximum" if kwargs["op"] == "max" else str(kwargs["op"]),
            input_operands=("data1", "data2"),
            output_operand="dst",
        )

    @classmethod
    def partition_tile_batching_contract(cls, kwargs: Mapping[str, Any]) -> PartitionTileBatchingContract | None:
        """Allow elementwise binary operations to span contiguous physical tiles."""
        return PartitionTileBatchingContract(operands=("data1", "data2", "dst")) if kwargs.get("op") in _OPS else None

    def _check_roles(self, **kwargs: Any) -> None:
        """Allow a PSUM first input while requiring an SBUF second input."""
        data1_role = _operand_role(kwargs["data1"])
        if data1_role is not None and data1_role not in {"sbuf", "psum"}:
            raise TypeError(f"NKITensorTensor(data1=<role={data1_role}>) expects sbuf or psum")
        data2_role = _operand_role(kwargs["data2"])
        if data2_role is not None and data2_role != "sbuf":
            raise TypeError(f"NKITensorTensor(data2=<role={data2_role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate and return ``data1 <op> data2`` elementwise."""
        return _OPS[kwargs["op"]](kwargs["data1"], kwargs["data2"])


def emit_packed_variance(source: TorchValue, node: Node, emit: TorchArithmetic) -> TorchValue:
    """Compute centered population variance in a packed native layout."""
    logical = node.meta[PACKING_MARKER]
    rows, width = int(np.prod(logical[:-1])), int(logical[-1])
    source, inverse = centered_variance_input(source, node, emit)
    partial = emit.emit("NKIActivationReduce", f"data={source.name}", "op='copy', reduce_op='add'")
    mean = emit_packed_reduction(partial, rows, source.shape[0], emit, "add")
    mean = TorchValue(
        emit.scalar("multiply", mean.name, 1.0 / width), mean.shape, mean.transposed, storage_dtype="float32"
    )
    _data, broadcast = prepare_packed_binary(source, mean, node, emit)
    if not isinstance(broadcast, TorchValue):
        raise TypeError("packed variance requires a tensor mean")
    centered = emit.scalar("subtract", source.name, broadcast.name)
    partial = emit.emit(
        "NKIActivationReduce", f"data={centered}", f"op='square', reduce_op='add', scale={width ** -0.5!r}"
    )
    variance = emit_packed_reduction(partial, rows, source.shape[0], emit, "add")
    result = emit.binary("multiply", emit.binary("multiply", variance.name, inverse.name), inverse.name)
    return TorchValue(result, variance.shape, variance.transposed, storage_dtype="float32")


def packed_reduction(node: Node) -> bool:
    """Recognize reductions whose lowerings retain the original row domain."""
    operation, native = _operation(node), node.meta.get("arithmetic") == "native"
    dimension, correction = node.kwargs.get("dim"), node.kwargs.get("correction")
    return packed_maximum(node) or (
        isinstance(dimension, int)
        and dimension == -1
        and node.kwargs.get("keepdim") is True
        and (
            operation == "mean"
            and (native or not getattr(node.target, "__name__", "").startswith("wrapped_"))
            or operation == "var"
            and native
            and isinstance(correction, (int, float))
            and correction == 0
        )
    )
