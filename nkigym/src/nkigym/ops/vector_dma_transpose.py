"""Transpose a partition vector into one SBUF row using DMA."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import CopyContract, NKIOp, _operand_role


class NKIVectorDMATranspose(NKIOp):
    """Copy a partition vector to the free axis without changing its values."""

    NAME: ClassVar[str] = "dma_transpose"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P",), "dst": ("F", "P")}
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {"src": (("P",), ("F",))}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"})}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {
        "src": frozenset({"bfloat16", "float16", "float32", "int32", "uint32"})
    }
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"F": 1}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": 1}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    OUTPUT_TILE_ALIGNMENT_BYTES: ClassVar[dict[str, int]] = {"dst": 32}

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> CopyContract:
        """Describe the value-preserving vector reshape."""
        return CopyContract(input_operand="src", output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an on-chip vector."""
        source = np.asarray(kwargs["src"])
        if _operand_role(kwargs["src"]) not in {None, "sbuf"} or source.ndim != 1:
            raise TypeError("vector DMA transpose requires one SBUF vector")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the vector as a single row without numeric conversion."""
        return np.asarray(kwargs["src"]).reshape(1, -1).copy()


def emit_packed_reduction(name: str, rows: int, partitions: int, emit: TorchArithmetic, op: str) -> TorchValue:
    """Merge consecutive partition partials with the given row reduction."""
    partials = emit.emit("NKIVectorDMATranspose", f"src={name}")
    total = emit.emit(
        "NKIGroupedTileReduce",
        f"data={partials}",
        f"partitions=1, groups={rows}, chunks={partitions // rows}, op={op!r}",
    )
    return TorchValue(total, (rows, 1), transposed=True, storage_dtype="float32")


def packed_shape(shape: tuple[int, ...], native: bool) -> tuple[int, int] | None:
    """Choose independent partitions, respecting complete reference sum groups."""
    rows, width = int(np.prod(shape[:-1])), shape[-1]
    if not 1 <= rows <= 128:
        return None
    if native:
        parts = next((part for part in range(min(128 // rows, width // 128), 1, -1) if width % part == 0), 1)
        valid = parts > 1
    else:
        lanes = 8 if width >= 8 else 1
        units = width // (4 * lanes)
        span = (1 << max(4, (max(1, units) - 1).bit_length() // 4)) * 4 * lanes
        groups = width // span if width % span == 0 else 0
        parts = next((part for part in range(min(128 // rows, groups), 1, -1) if groups % part == 0), 1)
        valid = parts >= 4
    return (rows * parts, width // parts) if valid else None


def full_partition_packing(shape: tuple[int, ...], allow_single_row: bool = False) -> bool:
    """Recognize full partition occupancy, optionally including one logical row."""
    rows, width = int(np.prod(shape[:-1])), shape[-1]
    eligible = rows >= (1 if allow_single_row else 2) and 128 % rows == 0
    return eligible and width % (128 // rows) == 0 and width // (128 // rows) >= 128


def packed_sum_geometry(source: tuple[int, ...], logical: tuple[int, int], span: int) -> tuple[int, int]:
    """Require complete ordered chunks and return partition groups and local chunks."""
    rows, width = logical
    if rows < 1 or source[0] % rows or int(np.prod(source)) != rows * width or source[1] % span:
        raise ValueError("packed reference sum requires equal element counts and complete ordered chunks")
    return source[0] // rows, source[1] // span


def restore_partial_rows(source: str, shape: tuple[int, int], emit: TorchArithmetic) -> str:
    """Restore consecutive partial groups to their original logical rows."""
    rows, parts = shape
    stored = emit.emit("NKIGroupedStore", f"src={source}", f"groups=1, rows={rows}, stages={parts}")
    return emit.emit("NKILoad", f"src={stored}")


def emit_packed_feature(source: TorchValue, shape: tuple[int, int], rows: int, emit: TorchArithmetic) -> str:
    """Broadcast one computed float32 feature row into packed partitions."""
    partitions, width = shape
    parts = partitions // rows
    data = emit.emit("NKILoad", f"src={source.name}") if source.is_hbm else source.name
    if source.storage_dtype != "float32":
        data = emit.cast("NKIFloat32Cast", data)
    if source.transposed or len(source.shape) == 1:
        operation = "NKIVectorDMATranspose" if len(source.shape) == 1 else "NKIDMATranspose"
        data = emit.emit(operation, f"src={data}")
    stored = emit.emit("NKIStore", f"src={data}")
    chunks = emit.emit("NKIGroupedLoad", f"src={stored}", f"groups=1, rows=1, stages={parts}")
    return repeat_partition_rows(chunks, shape, parts, emit)


def repeat_partition_rows(chunks: str, shape: tuple[int, int], parts: int, emit: TorchArithmetic) -> str:
    """Repeat consecutive source partitions without another HBM read."""
    partitions, width = shape
    result = emit.emit(
        "NKIIota", "", f"partitions={partitions}, width={width}, pattern=[[0, {width}]], channel_multiplier=0"
    )
    for start in range(0, partitions, 32):
        count = min(32, partitions - start)
        for source_start in range(0, parts, 32):
            source_rows = min(32, parts - source_start)
            positions = [(start + lane) % parts - source_start for lane in range(32)]
            mask = [index if lane < count and 0 <= index < source_rows else 255 for lane, index in enumerate(positions)]
            if any(index != 255 for index in mask):
                result = emit.emit(
                    "NKIStreamShuffle",
                    f"src={chunks}, dst={result}",
                    f"source_start={source_start}, source_rows={source_rows}, destination_start={start}, "
                    f"destination_rows={count}, start=0, width={width}, shuffle_mask={mask}",
                )
    return result
