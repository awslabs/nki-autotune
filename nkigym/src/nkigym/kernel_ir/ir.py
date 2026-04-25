"""KernelIR — flat kernel representation matching ``examples/matmul_lhsT_rhs.md``.

A ``KernelIR`` captures all state for lowering a matmul-shaped nkigym
program to NKI source:

* **Derived objective information** (immutable once built):
  ``func_name``, ``param_names``, ``return_name``, ``dimensions``,
  ``logical_tensors``, ``physical_buffers``.
* **Tunable IR knobs** (sampler-owned):
  ``ops``, ``edges``, ``dim_order``, ``ltiles_per_block``,
  ``buffer_scopes``, ``num_buffers``, ``emission_depth``.
"""

from dataclasses import dataclass, field
from enum import Enum

from nkigym.kernel_ir.types import DimInfo, TensorInfo


class BufferScope(Enum):
    """Per-buffer usage scope — determines buffer sizing.

    The label counts how many of the buffer's own dims span their full
    extent, counting outermost-first in ``dim_order``:

    * ``INNER`` — 0 dims full (all tile-sized).
    * ``MIDDLE`` — 1 dim full (outermost-in-``dim_order`` per-block,
      inner full for a 2D buffer).
    * ``OUTER`` — all dims full.
    """

    INNER = "inner"
    MIDDLE = "middle"
    OUTER = "outer"


@dataclass
class PhysicalBuffer:
    """SBUF buffer carrying a logical tensor.

    ``p_axis`` / ``f_axis`` name the dims laid across the partition
    and free axes. For 1D tensors ``f_axis`` is ``None``.
    """

    tile: tuple[int, int]
    dim_ids: tuple[str, ...]
    dtype: str
    p_axis: str
    f_axis: str | None


@dataclass
class NumBuffers:
    """Per-axis multi-buffering factors for one physical buffer.

    ``None`` on an axis ⇒ no rotation along that axis (that list level
    collapses in ``allocate_buffers``'s return shape).
    """

    num_p_buffers: int | None = None
    num_f_buffers: int | None = None


@dataclass
class Op:
    """One NKI op record in the ops list.

    Attributes:
        kind: Op class name — ``"NKILoad"``, ``"NKIMatmul"``, ``"NKIStore"``.
        inputs: Named inputs — role → tensor name. Tensor names resolve
            against ``logical_tensors`` (kernel params and producer outputs)
            or ``physical_buffers`` (SBUF-side aliases).
        outputs: Output names in declaration order.
        axis_map: Abstract axis label → dim id, e.g. ``{"K": "d0"}``.
        blocking_dims: Dim ids this op iterates over as inner loops
            (the K reduction axis for matmul).
    """

    kind: str
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    axis_map: dict[str, str] = field(default_factory=dict)
    blocking_dims: set[str] = field(default_factory=set)


@dataclass
class KernelIR:
    """Flat kernel IR matching the design-doc schema."""

    func_name: str
    param_names: list[str]
    return_name: str
    dimensions: dict[str, DimInfo]
    logical_tensors: dict[str, TensorInfo]
    physical_buffers: dict[str, PhysicalBuffer]
    ops: list[Op]
    edges: list[tuple[int, int]] = field(default_factory=list)
    dim_order: list[str] = field(default_factory=list)
    ltiles_per_block: dict[str, int] = field(default_factory=dict)
    buffer_scopes: dict[str, BufferScope] = field(default_factory=dict)
    num_buffers: dict[str, NumBuffers] = field(default_factory=dict)
    emission_depth: dict[str, int] = field(default_factory=dict)

    def num_blocks(self, dim_id: str) -> int:
        """Blocks along ``dim_id`` — ``num_ltile / ltiles_per_block``."""
        info = self.dimensions[dim_id]
        num_ltile = info.dim_size // info.logical_tile_size
        return num_ltile // self.ltiles_per_block[dim_id]

    def block_extent(self, dim_id: str) -> int:
        """Elements per block along ``dim_id`` — ``ltiles_per_block × ptile``."""
        info = self.dimensions[dim_id]
        return self.ltiles_per_block[dim_id] * info.physical_tile_size
