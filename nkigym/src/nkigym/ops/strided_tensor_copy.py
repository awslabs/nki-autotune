"""Bit-exact strided selection through one Vector Engine tensor copy."""

from collections.abc import Mapping
from math import prod
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import NKIOp, SliceContract, _operand_role
from nkigym.ops.vector_dma_transpose import packed_sum_geometry, restore_partial_rows


class NKIStridedTensorCopy(NKIOp):
    """Copy a strided floating-point or uint32 view without numeric conversion."""

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "N")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"})}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {
        "src": frozenset({"float32", "uint32", "bfloat16", "float16"})
    }
    STRIDED_COPY_INPUTS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"N": "width"}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"F", "N"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1, "N": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None, "N": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"pattern", "offset", "width"})

    def __init__(self, pattern: tuple[tuple[int, int], ...], offset: int) -> None:
        """Select a fixed strided view and the bit-preserving Vector Engine."""
        if not pattern or len(pattern) > 4 or offset < 0 or any(s < 0 or n < 1 for s, n in pattern):
            raise ValueError("strided tensor copy requires one to four valid free-axis dimensions")
        super().__init__(pattern=pattern, offset=offset, width=prod(n for _, n in pattern), engine="vector")

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> SliceContract:
        """Describe the ordered source positions copied without arithmetic."""
        return SliceContract("src", "dst", 1, int(kwargs["offset"]), int(kwargs["width"]), kwargs["pattern"])

    def _check_roles(self, **kwargs: Any) -> None:
        """Require supported SBUF storage and a view within its free axis."""
        source = np.asarray(kwargs["src"])
        if _operand_role(kwargs["src"]) not in {None, "sbuf"} or source.ndim != 2:
            raise TypeError("NKIStridedTensorCopy requires a rank-two SBUF source")
        if str(source.dtype) not in self.INPUT_STORAGE_DTYPES["src"]:
            raise TypeError("NKIStridedTensorCopy requires supported floating-point or uint32 storage")
        if kwargs["engine"] != "vector":
            raise ValueError("NKIStridedTensorCopy requires the Vector Engine")
        last = kwargs["offset"] + sum(stride * (extent - 1) for stride, extent in kwargs["pattern"])
        if last >= source.shape[1]:
            raise ValueError("strided tensor copy exceeds the source free-axis extent")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Select source elements without changing their representations."""
        indices = np.array([kwargs["offset"]], dtype=np.int64)
        for stride, extent in kwargs["pattern"]:
            indices = (indices[:, None] + stride * np.arange(extent)).reshape(-1)
        return np.asarray(kwargs["src"])[:, indices].copy()


def emit_torch_sum(
    source: TorchValue, name: str, body: list[str], imports: set[str], original_shape: tuple[int, int] | None = None
) -> TorchValue:
    """Emit Torch's four-accumulator cascade and ordered vector/scalar tails."""
    rows, width = source.shape if original_shape is None else original_shape
    emit = TorchArithmetic(name, body, imports)
    data = source.name if source.storage_dtype == "float32" else emit.cast("NKIFloat32Cast", source.name)
    lanes = 8 if width >= 8 else 1
    units = width // (4 * lanes)
    step = 1 << max(4, (max(1, units) - 1).bit_length() // 4)

    def take(value: str, groups: int, size: int, pitch: int, offset: int) -> str:
        """Copy equal fixed-width pieces from contiguous groups."""
        return emit.emit(
            "NKIStridedTensorCopy", f"src={value}", f"pattern={((pitch, groups), (1, size))!r}, offset={offset}"
        )

    def fold(value: str, groups: int, items: int, size: int, offset: int) -> str:
        """Accumulate equally sized pieces in increasing address order."""
        total = emit.scalar("add", take(value, groups, size, items * size, offset), 0.0)
        for item in range(1, items):
            total = emit.binary("add", total, take(value, groups, size, items * size, offset + item * size))
        return total

    partials: list[str] = []
    value, count = data, units
    if source.shape[0] != rows:
        parts, groups = packed_sum_geometry(source.shape, (rows, width), step * 4 * lanes)
        value = fold(data, groups, step, 4 * lanes, 0)
        value = restore_partial_rows(value, (rows, parts), emit)
        count //= step
    while count:
        groups, tail = divmod(count, step)
        if tail:
            partials.append(fold(value, 1, tail, 4 * lanes, groups * step * 4 * lanes))
        value = fold(value, groups, step, 4 * lanes, 0) if groups else value
        count = groups
    zero = emit.emit(
        "NKIIota", "", f"partitions={rows}, width={4 * lanes}, pattern=[[0, {4 * lanes}]], channel_multiplier=0"
    )
    total = zero
    for partial in partials:
        total = emit.binary("add", total, partial)
    vector = take(total, 1, lanes, lanes, 0)
    for offset in range(units * 4 * lanes, width - width % lanes, lanes):
        vector = emit.binary("add", vector, take(data, 1, lanes, lanes, offset))
    for lane_group in range(1, 4):
        vector = emit.binary("add", vector, take(total, 1, lanes, lanes, lane_group * lanes))
    result = take(zero, 1, 1, 1, 0)
    for offset in range(width - width % lanes, width):
        result = emit.binary("add", result, take(data, 1, 1, 1, offset))
    if lanes == 1 or width % lanes:
        for lane in range(lanes):
            result = emit.binary("add", result, take(vector, 1, 1, 1, lane))
    else:
        scanned = emit.emit("NKITensorScalarCumulative", f"src={vector}", "op0='add', op1='add', imm0=0.0")
        result = take(scanned, 1, 1, 1, lanes - 1)
    imports.add("NKIActivationReduce")
    body.append(f'{name} = NKIActivationReduce(op="copy", reduce_op="add")(data={result})')
    return TorchValue(name, (rows,), storage_dtype="float32")
