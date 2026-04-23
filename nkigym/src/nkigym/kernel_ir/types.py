"""Dim / tensor value types used across the KernelIR layer.

Split from ``ir.py`` so leaf modules (``fusion_group``, compute-skip
specs, rewrite patterns) can import these small dataclasses without
pulling the entire flat ``KernelIR`` class.
"""

from dataclasses import dataclass
from enum import Enum


class DimRole(Enum):
    """Loop-iteration dependency structure of a dimension."""

    PARALLEL = "parallel"
    SERIAL = "serial"
    ACCUMULATION = "accumulation"


@dataclass
class DimInfo:
    """Per-dimension analysis result.

    Attributes:
        dim_size: Total number of elements along this dimension.
        logical_tile_size: Iteration granularity.
        physical_tile_size: Buffer allocation granularity.
        role: ``DimRole`` for this dim's loop-iteration dependency.
    """

    dim_size: int
    logical_tile_size: int
    physical_tile_size: int
    role: DimRole

    @property
    def blocks_consumers(self) -> bool:
        """True iff downstream consumers must wait for this dim's loop to finish."""
        return self.role is DimRole.SERIAL

    @property
    def is_sequential(self) -> bool:
        """True iff iterations share buffer state (``SERIAL`` or ``ACCUMULATION``)."""
        return self.role is not DimRole.PARALLEL

    @property
    def num_ptiles(self) -> int:
        """Physical tiles per logical tile."""
        return self.logical_tile_size // self.physical_tile_size


@dataclass
class TensorInfo:
    """Per-tensor analysis result.

    Attributes:
        dim_ids: Concrete dimension IDs.
        shape: Full shape.
        dtype: Dtype string.
    """

    dim_ids: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
