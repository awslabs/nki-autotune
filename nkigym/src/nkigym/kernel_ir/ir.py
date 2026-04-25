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
from typing import Any

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
        kind: Op class name — e.g. ``"NKILoad"``, ``"NKIMatmul"``,
            ``"NKITranspose"``, ``"NKIStore"``.
        inputs: Named inputs — role → tensor name.
        outputs: Output names in declaration order.
        axis_map: Abstract axis label → dim id, e.g. ``{"K": "d0"}``.
        blocking_dims: Dim ids this op iterates over as inner loops
            (e.g. the K reduction axis for matmul).
        attrs: Per-op free-form attributes — rewrite-owned. Notable key:
            * ``transpose`` (on ``NKILoad``) — when ``True`` the load
              emits ``load_block(..., transpose=True)`` and the
              destination sbuf's (p_axis, f_axis) are swapped relative
              to the source HBM tensor.
    """

    kind: str
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    axis_map: dict[str, str] = field(default_factory=dict)
    blocking_dims: set[str] = field(default_factory=set)
    attrs: dict[str, Any] = field(default_factory=dict)


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

    def __repr__(self) -> str:
        """Human-readable multi-line view of every field."""
        return _format_kernel_ir(self)


def _format_kernel_ir(ir: "KernelIR") -> str:
    """Render ``ir`` as a section-structured multi-line string."""
    lines: list[str] = [f"KernelIR(func_name={ir.func_name!r})"]

    params = ", ".join(ir.param_names)
    lines.append(f"  signature: ({params}) -> {ir.return_name}")
    lines.append(f"  dim_order: {ir.dim_order}")
    lines.append(f"  ltiles_per_block: {_format_dict(ir.ltiles_per_block)}")

    lines.append("  dimensions:")
    dim_rows = [
        (
            name,
            str(info.dim_size),
            f"ltile={info.logical_tile_size}",
            f"ptile={info.physical_tile_size}",
            info.role.value,
        )
        for name, info in ir.dimensions.items()
    ]
    lines.extend(_format_table(dim_rows, indent="    "))

    lines.append("  logical_tensors:")
    tensor_rows = [
        (f"{name}:", f"dims=({','.join(t.dim_ids)})", f"shape={t.shape}", t.dtype)
        for name, t in ir.logical_tensors.items()
    ]
    lines.extend(_format_table(tensor_rows, indent="    "))

    lines.append("  physical_buffers:")
    buf_rows = []
    for name, buf in ir.physical_buffers.items():
        f_axis = buf.f_axis if buf.f_axis is not None else "-"
        buf_rows.append(
            (
                f"{name}:",
                f"tile={buf.tile}",
                f"dims=({','.join(buf.dim_ids)})",
                buf.dtype,
                f"p={buf.p_axis}",
                f"f={f_axis}",
            )
        )
    lines.extend(_format_table(buf_rows, indent="    "))

    lines.append("  buffer_scopes:")
    scope_rows = [(f"{name}:", scope.value) for name, scope in ir.buffer_scopes.items()]
    lines.extend(_format_table(scope_rows, indent="    "))

    lines.append("  num_buffers:")
    nb_rows = [
        (
            f"{name}:",
            f"p={nb.num_p_buffers if nb.num_p_buffers is not None else '-'}",
            f"f={nb.num_f_buffers if nb.num_f_buffers is not None else '-'}",
        )
        for name, nb in ir.num_buffers.items()
    ]
    lines.extend(_format_table(nb_rows, indent="    "))

    lines.append("  emission_depth:")
    depth_rows = [(f"{name}:", str(depth)) for name, depth in ir.emission_depth.items()]
    lines.extend(_format_table(depth_rows, indent="    "))

    lines.append("  ops:")
    for i, op in enumerate(ir.ops):
        inputs = ", ".join(f"{k}={v}" for k, v in op.inputs.items())
        outputs = ", ".join(op.outputs)
        extras: list[str] = []
        if op.blocking_dims:
            extras.append(f"blocking={{{','.join(sorted(op.blocking_dims))}}}")
        if op.axis_map:
            axis_str = ",".join(f"{k}={v}" for k, v in op.axis_map.items())
            extras.append(f"axes={{{axis_str}}}")
        if op.attrs:
            extras.append(f"attrs={op.attrs}")
        extras_str = f"  [{'  '.join(extras)}]" if extras else ""
        lines.append(f"    {i}. {op.kind}({inputs}) -> {outputs}{extras_str}")

    lines.append("  edges:")
    if ir.edges:
        edge_str = ", ".join(f"{p}->{c}" for p, c in ir.edges)
        lines.append(f"    {edge_str}")

    return "\n".join(lines)


def _format_dict(d: dict[str, int]) -> str:
    """Compact ``{k:v, k:v}`` rendering preserving insertion order."""
    return "{" + ", ".join(f"{k}:{v}" for k, v in d.items()) + "}"


def _format_table(rows: list[tuple[str, ...]], indent: str) -> list[str]:
    """Left-align each column across ``rows``; join with two-space gaps."""
    if not rows:
        return []
    width = max(len(r) for r in rows)
    col_widths = [max(len(r[c]) for r in rows if c < len(r)) for c in range(width)]
    out = []
    for row in rows:
        cells = [cell.ljust(col_widths[i]) for i, cell in enumerate(row)]
        out.append(indent + "  ".join(cells).rstrip())
    return out
