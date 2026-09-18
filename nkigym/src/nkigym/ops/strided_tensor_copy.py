"""Bit-exact 32-bit strided selection through one Vector Engine tensor copy."""

from math import prod
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import NKIOp, _operand_role


class NKIStridedTensorCopy(NKIOp):
    """Copy a strided float32 or uint32 view without numeric conversion."""

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "N")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"})}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"float32", "uint32"})}
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

    def _check_roles(self, **kwargs: Any) -> None:
        """Require 32-bit SBUF storage and a view within its free axis."""
        source = np.asarray(kwargs["src"])
        if _operand_role(kwargs["src"]) not in {None, "sbuf"} or source.ndim != 2:
            raise TypeError("NKIStridedTensorCopy requires a rank-two SBUF source")
        if source.dtype not in (np.dtype("float32"), np.dtype("uint32")):
            raise TypeError("NKIStridedTensorCopy requires float32 or uint32 storage")
        if kwargs["engine"] != "vector":
            raise ValueError("NKIStridedTensorCopy requires the Vector Engine")
        last = kwargs["offset"] + sum(stride * (extent - 1) for stride, extent in kwargs["pattern"])
        if last >= source.shape[1]:
            raise ValueError("strided tensor copy exceeds the source free-axis extent")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Select source elements without changing their 32-bit representations."""
        indices = np.array([kwargs["offset"]], dtype=np.int64)
        for stride, extent in kwargs["pattern"]:
            indices = (indices[:, None] + stride * np.arange(extent)).reshape(-1)
        return np.asarray(kwargs["src"])[:, indices].copy()


def emit_torch_sum(source: TorchValue, name: str, body: list[str], imports: set[str]) -> TorchValue:
    """Emit Torch's four-accumulator cascade and ordered vector/scalar tails."""
    rows, width = source.shape
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
    for lane in range(lanes):
        result = emit.binary("add", result, take(vector, 1, 1, 1, lane))
    imports.add("NKIActivationReduce")
    body.append(f'{name} = NKIActivationReduce(op="copy", reduce_op="add")(data={result})')
    return TorchValue(name, (rows,), storage_dtype="float32")
