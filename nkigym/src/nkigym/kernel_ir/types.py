"""Dim / tensor value types used by the KernelIR layer."""

from dataclasses import dataclass
from enum import Enum


class DimRole(Enum):
    """Loop-iteration dependency structure of a dimension."""

    PARALLEL = "parallel"
    ACCUMULATION = "accumulation"


@dataclass
class DimInfo:
    """Per-dimension analysis result.

    Attributes:
        dim_size: Total number of elements along this dimension.
        logical_tile_size: Iteration granularity.
        physical_tile_size: Buffer allocation granularity.
        role: ``DimRole`` for this dim.
    """

    dim_size: int
    logical_tile_size: int
    physical_tile_size: int
    role: DimRole


@dataclass
class TensorInfo:
    """Per-tensor analysis result."""

    dim_ids: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
