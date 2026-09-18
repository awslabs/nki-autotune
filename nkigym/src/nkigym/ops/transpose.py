"""Tensor-Engine transpose op for ``nisa.nc_transpose``.

Swaps the partition and free axes of a tensor. Takes a ``(P, F)``
operand and produces an ``(F, P)`` output. Executes on Tensor Engine
by default (Vector Engine is a 32×32 fallback); the caller pays TE
cycles — contrast with :class:`NKIDMATranspose` which runs on the
DMA engine and leaves TE free for matmul.
"""

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.ops.base import NKIOp, PermutationContract, _operand_role
from nkigym.ops.register_load import ControlEmitter

_MATMUL_DTYPES = frozenset(
    {"float8_e4m3", "float8_e5m2", "bfloat16", "float16", "tfloat32", "float32", "float8_e4m3fn"}
)


class NKITranspose(NKIOp):
    """Transpose ``data(P, F) -> dst(F, P)`` on Tensor Engine."""

    NAME: ClassVar[str] = "nc_transpose"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"data": ("P", "F"), "dst": ("F", "P")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"sbuf"})}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {"data": _MATMUL_DTYPES}
    """Tensor Engine caps the input at 128×128; Vector Engine at 32×32.
    We target Tensor Engine, so both axes are capped at 128."""
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": 128}
    OUTPUT_ROLE: ClassVar[str] = "psum"
    OUTPUT_LOCATION: ClassVar[str] = "psum"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PermutationContract:
        """Return the two-axis transpose contract."""
        _ = kwargs
        return PermutationContract(input_operand="data", output_operand="dst", permutation=(1, 0))

    def _check_roles(self, **kwargs: Any) -> None:
        """``data`` must be SBUF-resident."""
        role = _operand_role(kwargs["data"])
        if role is not None and role != "sbuf":
            raise TypeError(f"NKITranspose(data=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate and return ``data.T``."""
        return np.array(kwargs["data"]).T


def emit_partition_sum(emit: TorchArithmetic, column: str) -> str:
    """Sum a column across partitions and return one scalar column."""
    row = emit.emit("NKIDMATranspose", f"src={column}")
    total = emit.emit("NKITensorReduce", f"data={row}", "op='add', axis=1")
    zero = emit.emit("NKIIota", "", "partitions=1, width=1, pattern=[[0, 1]], channel_multiplier=0")
    return emit.scalar("add", zero, total)


def emit_partition_row_sum(emit: ControlEmitter, source: str, width: int) -> str:
    """Sum a matrix across partitions, returning one row in bounded transpose tiles."""
    output = emit.emit("NKIIota", "", f"partitions=1, width={width}, pattern=[[0, {width}]], channel_multiplier=0")
    emit.imports.add("NKIInplaceTensorCopy")
    for start in range(0, width, 128):
        size = min(128, width - start)
        piece = emit.emit("NKITensorSlice", f"src={source}", f"start={start}, width={size}")
        transposed = emit.emit("NKITranspose", f"data={piece}")
        copied = emit.emit("NKITensorCopy", f"src={transposed}", "engine='vector'")
        total = emit.emit("NKITensorReduce", f"data={copied}", "op='add', axis=1")
        zero = emit.emit("NKIIota", "", f"partitions={size}, width=1, pattern=[[0, 1]], channel_multiplier=0")
        column = emit.scalar("add", zero, total)
        row = emit.emit("NKITranspose", f"data={column}")
        row = emit.emit("NKITensorCopy", f"src={row}", "engine='vector'")
        emit.line(
            f"{output} = NKIInplaceTensorCopy(groups=1, partitions=1, start={start}, width={size}, "
            f"engine='vector')(src={row}, dst={output})"
        )
    return output


def emit_merge_disjoint_rows(
    emit: ControlEmitter,
    destinations: tuple[str, str],
    values: tuple[str, str],
    inside: str,
    shape: tuple[int, int],
    index_cast: Literal["NKIUInt16Cast", "NKIUInt32Cast"],
) -> None:
    """Merge disjoint value/index intervals without arithmetic on unselected NaNs.

    At most one row contributes at each position. Indices must be integers
    exactly representable in float32, as guaranteed by the caller's width bound.
    """
    partitions, width = shape
    zeros = emit.emit(
        "NKIIota", "", f"partitions={partitions}, width={width}, pattern=[[0, {width}]], channel_multiplier=0"
    )
    selected_values = emit.select(inside, values[0], zeros)
    selected_indices = emit.select(inside, emit.cast("NKIFloat32Cast", values[1]), zeros)
    merged = (
        emit_partition_row_sum(emit, selected_values, width),
        emit.cast(index_cast, emit_partition_row_sum(emit, selected_indices, width)),
    )
    changed = emit.cast("NKIUInt32Cast", emit_partition_row_sum(emit, inside, width))
    emit.imports.add("NKITensorCopyPredicated")
    for destination, source in zip(destinations, merged, strict=True):
        emit.line(f"{destination} = NKITensorCopyPredicated()(src={source}, predicate={changed}, dst={destination})")
