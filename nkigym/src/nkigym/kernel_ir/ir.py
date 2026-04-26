"""KernelIR — flat kernel representation matching ``examples/matmul_lhsT_rhs.md``.

A ``KernelIR`` captures all state for lowering a matmul-shaped nkigym
program to NKI source:

* **Derived objective information** (immutable once built):
  ``func_name``, ``param_names``, ``return_name``, ``dimensions``,
  ``logical_tensors``, ``physical_buffers``.
* **Tunable IR knobs** (sampler-owned):
  ``ops``, ``edges``, ``loop_order``, ``ltiles_per_block``,
  ``buffer_scopes``.

``loop_order`` carries **2N entries** — ``{d}.block`` + ``{d}.tile`` per
dim. ``{d}.block`` must precede ``{d}.tile`` for every dim.
``buffer_scopes`` is a per-dim extent map — ``{buffer: {dim: DimScope}}``.
Allocation depths and rotation are derived mechanically; they are not
carried in the IR.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from nkigym.kernel_ir.types import DimInfo, TensorInfo


class DimScope(Enum):
    """Per-buffer per-dim extent.

    * ``PER_TILE`` — buffer holds one tile along this dim.
    * ``PER_BLOCK`` — buffer holds ``ltiles_per_block[d]`` tiles along
      this dim.
    * ``FULL`` — buffer holds every tile along this dim
      (``num_ltile[d]`` tiles).
    """

    PER_TILE = "per_tile"
    PER_BLOCK = "per_block"
    FULL = "full"


BufferLoc = Literal["sbuf", "psum", "hbm"]
"""Physical memory pool a buffer lives in.

* ``"sbuf"`` — state buffer (24 MiB on Trn2); the default for every
  on-chip tile.
* ``"psum"`` — partial-sum buffer (8 banks × 128 partitions × 512 cols
  × fp32 = 2 MiB). Written by ``nc_matmul`` / ``nc_transpose``; drained
  separately to SBUF.
* ``"hbm"`` — device HBM; only the kernel output lives here."""


@dataclass
class PhysicalBuffer:
    """Physical buffer carrying a logical tensor.

    ``dim_ids`` names the dims the buffer is structurally laid out over
    (partition + free axis). For 2D tensors the first id is the
    partition axis, the second is the free axis. ``tile`` carries the
    per-tile shape on those axes. ``loc`` names the memory pool.
    """

    tile: tuple[int, int]
    dim_ids: tuple[str, ...]
    dtype: str
    loc: BufferLoc = "sbuf"

    @property
    def p_axis(self) -> str:
        """Partition axis — first entry in ``dim_ids``."""
        return self.dim_ids[0]

    @property
    def f_axis(self) -> str | None:
        """Free axis — second entry in ``dim_ids`` when the buffer is 2D."""
        return self.dim_ids[1] if len(self.dim_ids) >= 2 else None


@dataclass
class Op:
    """One NKI op record in the ops list.

    Attributes:
        kind: Op class name — e.g. ``"NKILoad"``, ``"NKIMatmul"``,
            ``"NKIStore"``.
        inputs: Named inputs — role → tensor name.
        outputs: Output names in declaration order.
        axis_map: Abstract axis label → dim id, e.g. ``{"K": "d0"}``.
        blocking_dims: Dim ids this op reduces over (e.g. K for matmul).
        kwargs: Scalar-literal keyword arguments captured from the
            nkigym call site. Tensor-valued kwargs live in ``inputs``.
        attrs: Per-op free-form attributes — rewrite-owned.
    """

    kind: str
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    axis_map: dict[str, str] = field(default_factory=dict)
    blocking_dims: set[str] = field(default_factory=set)
    kwargs: dict[str, Any] = field(default_factory=dict)
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
    loop_order: list[str] = field(default_factory=list)
    ltiles_per_block: dict[str, int] = field(default_factory=dict)
    buffer_scopes: dict[str, dict[str, DimScope]] = field(default_factory=dict)

    def num_ltile(self, dim_id: str) -> int:
        """Logical-tile count along ``dim_id`` — ``dim_size / logical_tile_size``."""
        info = self.dimensions[dim_id]
        return info.dim_size // info.logical_tile_size

    def num_blocks(self, dim_id: str) -> int:
        """Blocks along ``dim_id`` — ``num_ltile / ltiles_per_block``."""
        return self.num_ltile(dim_id) // self.ltiles_per_block[dim_id]

    def block_extent(self, dim_id: str) -> int:
        """Elements per block along ``dim_id`` — ``ltiles_per_block × ptile``."""
        info = self.dimensions[dim_id]
        return self.ltiles_per_block[dim_id] * info.physical_tile_size

    def loop_depth(self, entry: str) -> int:
        """Depth (1-indexed) at which ``entry`` opens in ``loop_order``."""
        return self.loop_order.index(entry) + 1

    def __repr__(self) -> str:
        """Human-readable multi-line view of every field."""
        return _format_kernel_ir(self)


def _format_kernel_ir(ir: "KernelIR") -> str:
    """Render ``ir`` as a section-structured multi-line string."""
    lines: list[str] = [f"KernelIR(func_name={ir.func_name!r})"]

    params = ", ".join(ir.param_names)
    lines.append(f"  signature: ({params}) -> {ir.return_name}")
    lines.append(f"  loop_order: {ir.loop_order}")
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
        buf_rows.append(
            (f"{name}:", f"loc={buf.loc}", f"tile={buf.tile}", f"dims=({','.join(buf.dim_ids)})", buf.dtype)
        )
    lines.extend(_format_table(buf_rows, indent="    "))

    lines.append("  buffer_scopes:")
    scope_rows: list[tuple[str, ...]] = []
    for name, scope_map in ir.buffer_scopes.items():
        parts = ",".join(f"{d}={s.value}" for d, s in scope_map.items())
        scope_rows.append((f"{name}:", "{" + parts + "}"))
    lines.extend(_format_table(scope_rows, indent="    "))

    lines.append("  ops:")
    for i, op in enumerate(ir.ops):
        inputs = ", ".join(f"{k}={v}" for k, v in op.inputs.items())
        outputs = ", ".join(op.outputs)
        extras: list[str] = []
        if op.kwargs:
            kw_str = ",".join(f"{k}={v!r}" for k, v in op.kwargs.items())
            extras.append(f"kwargs={{{kw_str}}}")
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
