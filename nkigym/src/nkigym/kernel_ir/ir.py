"""KernelIR — schema per ``/home/ubuntu/nki-autotune/nkigym/src/nkigym/design.md``.

A KernelIR is a flat dataclass with:

* **Immutable objective information** (derived from the nkigym math function):
    ``func_name``, ``param_names``, ``return_name``, ``dimensions``,
    ``logical_tensors``, ``physical_buffers`` (each carrying ``p_axis`` / ``f_axis``).
* **Tunable IR knobs** (sampler-owned):
    ``ops`` (ordered list of :class:`Op` records, each with per-op attrs
    like ``NKITranspose.mode``), ``edges``, ``dim_order``,
    ``ltiles_per_block``, ``buffer_scopes``, ``num_buffers`` (per-axis:
    ``num_p_buffers`` / ``num_f_buffers``), ``emission_depth``.

Rewrites are external transformations that mutate ``ops`` / ``edges``
in place (or return a new IR); they are not fields on the IR itself.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nkigym.kernel_ir.types import DimInfo, TensorInfo


class BufferScope(Enum):
    """Per-buffer usage scope — determines sizing.

    The label counts how many of the buffer's own dims span their full
    extent, counting outermost-first in ``dim_order``:

    * ``INNER`` — 0 dims full (all tile-sized).
    * ``MIDDLE`` — 1..n-1 dims full (outermost-in-`dim_order` per-block,
      the inner one(s) full). For the 2D SBUF model this means
      "outermost-in-`dim_order` per-block, the other full".
    * ``OUTER`` — all dims full.
    """

    INNER = "inner"
    MIDDLE = "middle"
    OUTER = "outer"


@dataclass
class PhysicalBuffer:
    """SBUF / PSUM buffer carrying a logical tensor.

    ``p_axis`` and ``f_axis`` name the dims laid across the partition
    and free axes of each leaf, respectively. For 1D tensors,
    ``f_axis`` is ``None`` (the leaf has a free axis of width 1).
    """

    tile: tuple[int, int]
    dim_ids: tuple[str, ...]
    dtype: str
    p_axis: str
    f_axis: str | None


@dataclass
class NumBuffers:
    """Per-axis multi-buffering factors for one physical buffer.

    ``None`` on an axis ⇒ no rotation along that axis (the list level
    collapses in the return shape of ``allocate_buffers``).

    If both are ``None``, the whole buffer is "compiler-offload":
    codegen emits the allocation at the tightest enclosing loop of
    every use and does not multi-buffer explicitly.
    """

    num_p_buffers: int | None = None
    num_f_buffers: int | None = None

    @property
    def is_compiler_offload(self) -> bool:
        """True iff both axes are ``None`` — allocation hoisted per use-site."""
        return self.num_p_buffers is None and self.num_f_buffers is None


@dataclass
class Op:
    """One NKI op record in the ops list.

    Attributes:
        kind: Op class name, e.g. ``"NKILoad"``, ``"NKIMatmul"``,
            ``"NKITranspose"``.
        inputs: Named inputs — role → tensor name.
            Tensor names resolve to either ``logical_tensors`` (for
            kernel params and op outputs before DMA insertion) or
            ``physical_buffers`` (for SBUF-side inputs after DMA
            insertion).
        outputs: Named outputs in declaration order.
        kwargs: Op-specific scalar/static kwargs — each entry is
            either a Python literal (scalar) or a tensor name
            (resolved against the tensor catalog at codegen time).
        attrs: Per-op free-form attributes. Notable keys:
            * ``mode`` (on ``NKITranspose``) ∈ ``{"dma_transpose",
              "nc_transpose"}`` — controls which transpose backend
              codegen emits.
        axis_map: Logical axis name → dim id (e.g. ``{"K": "d0"}``).
        tile_sizes: Per-dim-id tile size override.
        blocking_dims: Set of dim ids this op blocks on
            (i.e. drives inner loops for).
    """

    kind: str
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)
    axis_map: dict[str, str] = field(default_factory=dict)
    tile_sizes: dict[str, int] = field(default_factory=dict)
    blocking_dims: set[str] = field(default_factory=set)


@dataclass
class KernelIR:
    """Flat kernel IR matching ``design.md``.

    See module docstring for the field taxonomy (immutable vs tunable).
    """

    func_name: str
    param_names: list[str]
    return_name: str
    dimensions: dict[str, DimInfo]
    logical_tensors: dict[str, TensorInfo]
    physical_buffers: dict[str, PhysicalBuffer]
    ops: list[Op]
    edges: list[tuple[int, int, str, str]] = field(default_factory=list)
    dim_order: list[str] = field(default_factory=list)
    ltiles_per_block: dict[str, int] = field(default_factory=dict)
    buffer_scopes: dict[str, BufferScope] = field(default_factory=dict)
    num_buffers: dict[str, NumBuffers] = field(default_factory=dict)
    emission_depth: dict[str, int] = field(default_factory=dict)

    def tensor_info(self, name: str) -> TensorInfo:
        """Return ``TensorInfo`` for any name in the logical or physical catalog.

        ``physical_buffers`` entries synthesize a ``TensorInfo`` view
        from their dim_ids / dtype / (shape derived from dim sizes).
        """
        if name in self.logical_tensors:
            return self.logical_tensors[name]
        if name in self.physical_buffers:
            pb = self.physical_buffers[name]
            shape = tuple(self.dimensions[d].dim_size for d in pb.dim_ids)
            return TensorInfo(pb.dim_ids, shape, pb.dtype)
        raise KeyError(f"tensor {name!r} not found")

    def has_tensor(self, name: str) -> bool:
        """Whether a tensor of this name exists anywhere in the catalog."""
        return name in self.logical_tensors or name in self.physical_buffers

    def num_blocks(self, dim_id: str) -> int:
        """Number of blocks for ``dim_id`` — ``num_ltile / ltiles_per_block``."""
        info = self.dimensions[dim_id]
        num_ltile = info.dim_size // info.logical_tile_size
        return num_ltile // self.ltiles_per_block[dim_id]

    def block_extent(self, dim_id: str) -> int:
        """Elements per block along ``dim_id`` — ``ltiles_per_block × ptile``."""
        info = self.dimensions[dim_id]
        return self.ltiles_per_block[dim_id] * info.physical_tile_size

    def producer_of(self, tensor_name: str) -> int | None:
        """Index of the op that writes ``tensor_name``, or ``None`` if not produced."""
        for i, op in enumerate(self.ops):
            if tensor_name in op.outputs:
                return i
        return None
