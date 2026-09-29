"""Folded batch/tile DMA load."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
import torch
from torch.fx import Node

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.ops.base import CopyContract, NKIOp, _operand_role
from nkigym.ops.grouped_tensor_scalar_reduce import packed_maximum
from nkigym.ops.vector_dma_transpose import full_partition_packing, packed_shape, repeat_partition_rows


class NKIFoldedLoad(NKIOp):
    """Load independent two-dimensional tiles from one packed HBM tensor."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "G", "T", "F"), "dst": ("P", "G", "T", "F")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("P",), ("G", "T", "F")) for slot in OPERAND_AXES
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "T": "tiles"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "PGTF"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "G": 1, "T": 1, "F": None}
    TENSORIZE_MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"T": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "tiles"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, tiles: int) -> None:
        """Configure packed batch and tile extents."""
        if groups < 1 or tiles < 1:
            raise ValueError("folded load extents must be positive")
        super().__init__(groups=groups, tiles=tiles)

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> CopyContract:
        """Return the value-preserving load contract."""
        _ = kwargs
        return CopyContract(input_operand="src", output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one HBM parameter source."""
        if (role := _operand_role(kwargs["src"])) is not None and role != "param":
            raise TypeError(f"NKIFoldedLoad(src=<role={role}>) expects param")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return a copy of the packed source."""
        return np.array(kwargs["src"], copy=True)


def grouped_context_input(array: np.ndarray, shape: tuple[int, ...], transform: tuple[object, ...]) -> np.ndarray:
    """Pack one batched context-attention input by group and query tile."""
    kind, raw_dimensions = str(transform[1]), transform[2:]
    if len(raw_dimensions) != 7 or not all(isinstance(value, int) for value in raw_dimensions):
        raise ValueError(f"grouped context-attention layout requires seven integer dimensions, got {raw_dimensions}")
    groups, queries, tiles, _reduction, partitions, width, _output_width = tuple(
        value for value in raw_dimensions if isinstance(value, int)
    )
    if kind == "q_t":
        result = array.transpose(2, 0, 1)
    elif kind in {"q", "k"}:
        result = array.transpose(1, 0, 2)
    elif kind == "v":
        result = array.reshape(groups, tiles, width // 128, 128, -1).transpose(3, 0, 1, 2, 4)
    elif kind in {"lower", "upper"}:
        result = array.reshape(groups, queries, partitions).transpose(2, 0, 1)
    else:
        raise ValueError(f"unknown grouped context-attention input layout {kind!r}")
    return result.reshape(shape)


__all__ = ["NKIFoldedLoad", "grouped_context_input"]


PACKING_MARKER = "nkigym_partition_pack"
_BINARY = frozenset({"add", "sub", "subtract", "mul", "multiply", "maximum", "minimum"})
_CASTS = frozenset({"float", "to", "astype", "view_as"})


def _operation(node: Node) -> str:
    """Return one normalized FX operation name."""
    return str(getattr(node.target, "__name__", node.target)).removeprefix("wrapped_")


def _shape(node: Node) -> tuple[int, ...]:
    """Return the propagated tensor shape, if available."""
    metadata = node.meta.get("tensor_meta", node.meta.get("example_value"))
    shape = tuple(int(value) for value in getattr(metadata, "shape", ()))
    while len(shape) > 2 and shape[0] == 1:
        shape = shape[1:]
    return shape


def _candidate_shape(node: Node) -> tuple[int, ...] | None:
    """Recognize shape-preserving arithmetic on a short matrix."""
    shape = _shape(node)
    native = node.meta.get("arithmetic") == "native"
    operations = _BINARY | _CASTS | {"square", "imul", "iadd", "truediv", "clamp"}
    if native and all(packed_maximum(user) for user in node.users):
        operations |= {"abs", "absolute"}
    if node.op not in {"call_function", "call_method"} or _operation(node) not in operations:
        return None
    broadcasts = {shape, (), (*shape[:-1], 1), (shape[-1],), (1, shape[-1])} if len(shape) >= 2 else {shape}
    if len(shape) < 2 or any(_shape(argument) not in broadcasts for argument in node.all_input_nodes):
        return None
    for argument in node.all_input_nodes:
        metadata = argument.meta.get("tensor_meta", argument.meta.get("example_value"))
        dtype = getattr(metadata, "dtype", None)
        if _shape(argument) != shape and dtype is not None and str(dtype) not in {"torch.float32", "float32"}:
            return None
    return shape if packed_shape(shape, native) is not None else None


def packing_operation(node: Node) -> bool:
    """Count arithmetic operations whose packing amortizes layout changes."""
    return _operation(node) in _BINARY | {"imul", "iadd", "truediv"}


def packing_component(component: set[Node]) -> bool:
    """Amortize reference packing at a reduction boundary with enough arithmetic."""
    native = all(node.meta.get("arithmetic") == "native" for node in component)
    maximum = any(packed_maximum(user) for node in component for user in node.users)
    reduction = any(_operation(user) == "mean" for node in component for user in node.users)
    if native and not reduction and not full_partition_packing(_shape(next(iter(component))), maximum):
        return False
    uniform = all(_shape(arg) == _shape(node) for node in component for arg in node.all_input_nodes)
    return (native or reduction or uniform) and sum(packing_operation(node) for node in component) + maximum >= 2


def direct_hbm_user(node: Node) -> bool:
    """Recognize operations whose lowering reads the placeholder directly."""
    return bool(node.meta.get(PACKING_MARKER)) or node.target in {torch.cumsum, torch.topk, torch.max}


def emit_loaded_feature(source: str, shape: tuple[int, int], parts: int, emit: TorchArithmetic) -> str:
    """Load one feature vector once and repeat its partition chunks."""
    width = shape[1]
    configuration = f"partitions={parts}, width={width}, pattern=[[0, {width}]], channel_multiplier=0"
    chunks = emit.emit("NKIIota", "", configuration)
    chunks = emit.emit("NKIPartitionSliceLoad", f"src={source}, dst={chunks}", f"start=0, rows={parts}")
    return repeat_partition_rows(chunks, shape, parts, emit)
