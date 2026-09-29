"""Load an HBM vector into a contiguous SBUF partition window."""

from typing import Any, ClassVar

import numpy as np
from torch.fx import Node

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.folded_load import PACKING_MARKER, emit_loaded_feature
from nkigym.ops.vector_dma_transpose import emit_packed_feature
from nkigym.ops.vector_dma_transpose import packed_shape as choose_packed_shape


class NKIPartitionSliceLoad(NKIOp):
    """Copy one HBM vector into an existing partition window using one DMA."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("S", "F"), "dst": ("P", "F")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("S", "F"),),
        "dst": (("P",), ("F",)),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {"src": (("S",), ("F",))}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {
        "src": frozenset({"shared_hbm"}),
        "dst": frozenset({"sbuf"}),
    }
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {
        "src": frozenset({"float16", "bfloat16", "float32"}),
        "dst": frozenset({"float32"}),
    }
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset({"dst"})
    RETURN_RMW_OPERAND: ClassVar[str | None] = "dst"
    SYNTHESIZE_RMW_INITIALIZER: ClassVar[bool] = False
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"S": "rows"}
    PARTITION_SLICES: ClassVar[dict[str, tuple[str, str]]] = {"dst": ("start", "rows")}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"S", "P", "F"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "S": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "S": 128, "F": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"start", "rows"})

    def __init__(self, start: int, rows: int) -> None:
        """Configure one contiguous partition window."""
        if start < 0 or not 1 <= rows <= 128 or start + rows > 128:
            raise ValueError("partition load window must fit within 128 partitions")
        super().__init__(start=start, rows=rows)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require a matching HBM vector and floating-point SBUF destination."""
        source, target = np.asarray(kwargs["src"]), np.asarray(kwargs["dst"])
        if _operand_role(kwargs["src"]) not in {None, "param", "shared_hbm", "stored"} or source.ndim != 1:
            raise TypeError("partition load requires one HBM vector")
        if _operand_role(kwargs["dst"]) not in {None, "sbuf"} or target.ndim != 2 or target.dtype != np.float32:
            raise TypeError("partition load requires a rank-two float32 SBUF destination")
        if str(source.dtype) not in {"bfloat16", "float16", "float32"}:
            raise TypeError("partition load requires floating-point source values")
        if source.size != kwargs["rows"] * target.shape[1] or kwargs["start"] + kwargs["rows"] > target.shape[0]:
            raise ValueError("partition load source and destination window sizes must match")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Copy the source into the selected window, retaining the other rows."""
        start, rows = int(kwargs["start"]), int(kwargs["rows"])
        target = kwargs["dst"]
        target[start : start + rows] = np.asarray(kwargs["src"], dtype=np.float32).reshape(rows, -1)
        return target


def prepare_packed_binary(
    left: TorchValue | float, right: TorchValue | float, node: Node, emit: TorchArithmetic
) -> tuple[TorchValue | float, TorchValue | float]:
    """Expand row or feature broadcasts to match one packed matrix."""
    shape = node.meta.get(PACKING_MARKER)
    if not isinstance(shape, tuple):
        return left, right
    rows, width = int(np.prod(shape[:-1])), shape[-1]
    packed_shape = choose_packed_shape(shape, node.meta.get("arithmetic") == "native")
    assert packed_shape is not None
    partitions, free, stages = *packed_shape, packed_shape[0] // rows
    operands = [left, right]
    for index, value in enumerate(operands):
        if not isinstance(value, TorchValue) or value.shape == packed_shape:
            continue
        if value.shape in {(width,), (1, width)} and value.hbm_source is None:
            name = emit_packed_feature(value, packed_shape, rows, emit)
            operands[index] = TorchValue(name, packed_shape, storage_dtype="float32")
        elif value.shape in {(width,), (1, width)} and value.hbm_source is not None:
            if rows > 2 + (partitions + 31) // 32:
                name = emit_loaded_feature(value.hbm_source, packed_shape, stages, emit)
            else:
                configuration = f"partitions={partitions}, width={free}, pattern=[[0, {free}]], channel_multiplier=0"
                name = emit.emit("NKIIota", "", configuration)
                for start in range(0, partitions, stages):
                    name = emit.emit(
                        "NKIPartitionSliceLoad", f"src={value.hbm_source}, dst={name}", f"start={start}, rows={stages}"
                    )
            operands[index] = TorchValue(name, packed_shape, storage_dtype="float32")
        elif value.shape in {(rows,), (rows, 1)}:
            if value.transposed:
                value = TorchValue(emit.emit("NKIDMATranspose", f"src={value.name}"), (rows, 1))
            result = emit.emit(
                "NKIIota", "", f"partitions={partitions}, width=1, pattern=[[0, 1]], channel_multiplier=0"
            )
            for start in range(0, partitions, 32):
                count = min(32, partitions - start)
                source_start = (start // stages // 32) * 32
                mask = [(start + lane) // stages - source_start if lane < count else 255 for lane in range(32)]
                result = emit.emit(
                    "NKIStreamShuffle",
                    f"src={value.name}, dst={result}",
                    f"source_start={source_start}, source_rows={min(32, rows - source_start)}, "
                    f"destination_start={start}, destination_rows={count}, start=0, width=1, shuffle_mask={mask}",
                )
            operands[index] = TorchValue(result, (partitions, 1), storage_dtype="float32")
    return operands[0], operands[1]
