"""SBUF buffer model: list-of-lists of 2D physical tiles.

Every SBUF buffer is a nested Python list
``sbuf_X[p_list_slot][f_list_slot]`` where each leaf is a 2D
``nl.ndarray`` of shape ``(p.logical, f.logical)``. Each axis
carries the four factors that make up ``num_tiles``:
``multi_buffer × num_blocks × ltiles_per_block × ptiles_per_ltile``.

How those factors materialize differs by axis:

* Partition axis: every factor becomes a Python-list level. The
  leaf's P dim is always ``phys_P = 128`` (the full hardware
  partition block), so there is no ptile slice inside the leaf.
* Free axis: ``multi_buffer × num_blocks × ltiles_per_block``
  becomes Python-list levels; ``ptiles_per_ltile`` materializes
  as a free-axis slice inside the leaf (leaf F size is
  ``phys_F × ptiles_per_ltile``).

1D logical tensors are lifted to 2D with ``f.phys = 1`` and every
free-axis factor set to 1, so every access produces a genuine 2D
memref — no 4D reshape tricks, no affine-select 4D AP rejection.
"""

from dataclasses import dataclass

from nkigym.kernel_ir import KernelIR, get_tpb


@dataclass(frozen=True)
class SbufAxis:
    """Per-axis slot structure for an SBUF buffer.

    ``num_tiles = multi_buffer * num_blocks * ltiles_per_block *
    ptiles_per_ltile``. All four factors materialize as Python-
    list levels on the partition axis. On the free axis the first
    three are list levels and ``ptiles_per_ltile`` is an inner
    free-axis slice. ``logical`` gives the per-leaf size on this
    axis; ``list_slots`` gives the Python-list count on this axis.
    """

    phys: int
    ptiles_per_ltile: int
    ltiles_per_block: int
    num_blocks: int
    multi_buffer: int
    leaf_includes_ptile: bool

    @property
    def list_slots(self) -> int:
        """Length of the Python-list level on this axis."""
        outer = self.multi_buffer * self.num_blocks * self.ltiles_per_block
        return outer * (1 if self.leaf_includes_ptile else self.ptiles_per_ltile)

    @property
    def logical(self) -> int:
        """Leaf ndarray's size on this axis."""
        return self.phys * (self.ptiles_per_ltile if self.leaf_includes_ptile else 1)

    @property
    def num_tiles(self) -> int:
        """Total physical tiles on this axis across all loop levels."""
        return self.list_slots * (self.ptiles_per_ltile if self.leaf_includes_ptile else 1)


@dataclass(frozen=True)
class AxisAccess:
    """How a caller wants to iterate one axis of an SBUF buffer at a given scope.

    ``block``/``ltile``: ``None`` means the loop is not in scope
    (span every slot on that factor, start from 0); a string means
    the loop var is bound to that expression (span one slot, offset
    by the expression). Buffer slot stays at 0 until multi-buffer
    pipelining is wired up. For the free axis, ``ptile`` is either
    ``None`` (cover every physical tile inside the leaf) or a
    string (slice one leaf tile indexed by the expression).
    """

    block: str | None = None
    ltile: str | None = None
    ptile: str | None = None


@dataclass(frozen=True)
class SbufBuffer:
    """SBUF buffer backed by a nested Python list of 2D ``nl.ndarray`` leaves.

    Outer shape: ``[p.list_slots][f.list_slots]``. Each leaf is
    ``(p.logical, f.logical)``. 1D logical tensors are lifted to
    2D with ``f.phys = 1``, ``f.ptiles_per_ltile = 1``.
    """

    name: str
    dtype: str
    p: SbufAxis
    f: SbufAxis

    def alloc_line(self) -> str:
        """Emit the Python list-of-list allocation for this buffer."""
        leaf = f"nl.ndarray(({self.p.logical}, {self.f.logical}), dtype=nl.{self.dtype}, buffer=nl.sbuf)"
        inner = f"[{leaf} for _ in range({self.f.list_slots})]"
        outer = f"[{inner} for _ in range({self.p.list_slots})]"
        return f"sbuf_{self.name} = {outer}"

    def get_tile(self, p: AxisAccess, f: AxisAccess) -> str:
        """Return an access string for one physical 2D tile.

        Requires every list-factor loop to be in scope on both axes
        (``block``/``ltile`` bound, plus ``ptile`` bound on the
        partition axis where it is also a list level). On the free
        axis the leaf slice picks either one physical tile
        (``f.ptile`` bound) or the whole leaf (``f.ptile is None``).
        """
        p_idx = _list_index(self.p, p.block, p.ltile, p.ptile)
        f_idx = _list_index(self.f, f.block, f.ltile, None)
        f_slice = _f_ptile_slice(self.f, f.ptile) if f.ptile is not None else f"0:{self.f.logical}"
        return f"sbuf_{self.name}[{p_idx}][{f_idx}][0:{self.p.logical}, {f_slice}]"

    def range(self, p: AxisAccess, f: AxisAccess) -> tuple[str, int, str, int]:
        """Return ``(p_start, p_count, f_start, f_count)`` for a gadget sub-block.

        An axis factor (``block``/``ltile``/``ptile``) that is
        ``None`` contributes its full count to the range; a bound
        factor contributes one slot offset by the bound expression.
        ``ptile`` only participates on axes where the factor is a
        list level (the partition axis); on the free axis the
        ptile lives inside the leaf and gadgets always span it in
        full.
        """
        p_start, p_count = _list_range(self.p, p.block, p.ltile, p.ptile)
        f_start, f_count = _list_range(self.f, f.block, f.ltile, None)
        return p_start, p_count, f_start, f_count


def _list_index(axis: SbufAxis, block: str | None, ltile: str | None, ptile: str | None) -> str:
    """Collapse the list-level indices into a flat expression.

    List-level factors on ``axis`` are (outer→inner):
    ``multi_buffer``, ``num_blocks``, ``ltiles_per_block``, and —
    when ``axis.leaf_includes_ptile`` is ``False`` —
    ``ptiles_per_ltile``. Each bound argument contributes a term
    scaled by the product of strides inner to it; unbound
    arguments default to 0 and are allowed only when the factor
    collapses to 1. ``ptile`` must be ``None`` on axes whose leaf
    already carries the ptile slice.
    """
    strides, factors, idx_sources = _list_strides(axis, block, ltile, ptile)
    terms: list[str] = []
    for idx, stride, size in zip(idx_sources, strides, factors):
        if size == 1:
            continue
        if idx is None:
            raise ValueError(f"list factor of size {size} must be bound to a loop-var expression")
        terms.append(idx if stride == 1 else f"{idx} * {stride}")
    return " + ".join(terms) if terms else "0"


def _f_ptile_slice(axis: SbufAxis, f_ptile: str) -> str:
    """Return the ``start:end`` free-axis slice for one physical tile inside the leaf."""
    return f"{f_ptile} * {axis.phys}:{f_ptile} * {axis.phys} + {axis.phys}"


def _list_range(axis: SbufAxis, block: str | None, ltile: str | None, ptile: str | None) -> tuple[str, int]:
    """Return ``(start_expr, count)`` for a contiguous list-slot range on one axis.

    Each list-level factor is either bound (one slot, offset by
    the bound expression) or unbound (full span starting at 0).
    An unbound factor forbids binding any factor inner to it —
    the contiguous region must be a clean rectangular sub-list.
    """
    strides, factors, idx_sources = _list_strides(axis, block, ltile, ptile)
    terms: list[str] = []
    count = 1
    unbound_seen = False
    for idx, stride, size in zip(idx_sources, strides, factors):
        if size == 1:
            continue
        if idx is None:
            unbound_seen = True
            count *= size
            continue
        if unbound_seen:
            raise ValueError("cannot bind an inner factor while an outer factor is unbound — sub-list not contiguous")
        terms.append(idx if stride == 1 else f"{idx} * {stride}")
    start = " + ".join(terms) if terms else "0"
    return start, count


def _list_strides(
    axis: SbufAxis, block: str | None, ltile: str | None, ptile: str | None
) -> tuple[list[int], list[int], list[str | None]]:
    """Return per-factor ``(strides, factors, idx_sources)`` for list-level indexing on an axis.

    Factors are listed outer→inner: ``multi_buffer``,
    ``num_blocks``, ``ltiles_per_block``, ``ptiles_per_ltile``
    (the last included only when the leaf does not absorb it).
    Multi-buffer is always "0" today (single-buffered).
    """
    factors_list = [axis.multi_buffer, axis.num_blocks, axis.ltiles_per_block]
    idx_sources: list[str | None] = ["0", block, ltile]
    if not axis.leaf_includes_ptile:
        factors_list.append(axis.ptiles_per_ltile)
        idx_sources.append(ptile)
    elif ptile is not None:
        raise ValueError("ptile cannot be bound on an axis whose leaf already carries the ptile slice")
    strides: list[int] = []
    running = 1
    for size in reversed(factors_list):
        strides.append(running)
        running *= size
    strides.reverse()
    return strides, factors_list, idx_sources


def build_sbuf_buffer(ir: KernelIR, tensor_name: str, dtype: str) -> SbufBuffer:
    """Construct the ``SbufBuffer`` for a logical tensor.

    Decomposes ``num_tiles`` on each axis into its four factors.
    Partition axis materializes all four as list levels (leaf P
    is the fixed 128-lane partition). Free axis lifts
    ``ptiles_per_ltile`` into the leaf (leaf F is
    ``phys_F × ptiles_per_ltile``). 1D logical tensors are lifted
    to 2D with ``f.phys = 1``. ``dtype`` is resolved by the caller
    (may be promoted to float32 for FLOAT32_KWARGS).
    """
    tinfo = ir.dim_analysis.tensors[tensor_name]
    dim_ids = tinfo.dim_ids
    if len(dim_ids) not in (1, 2):
        raise ValueError(f"Tensor {tensor_name} has {len(dim_ids)} dims, expected 1 or 2")
    p_axis = _build_axis(ir, tensor_name, dim_ids[0], leaf_includes_ptile=False)
    if len(dim_ids) == 2:
        f_axis = _build_axis(ir, tensor_name, dim_ids[1], leaf_includes_ptile=True)
    else:
        f_axis = SbufAxis(
            phys=1, ptiles_per_ltile=1, ltiles_per_block=1, num_blocks=1, multi_buffer=1, leaf_includes_ptile=True
        )
    return SbufBuffer(name=tensor_name, dtype=dtype, p=p_axis, f=f_axis)


def _build_axis(ir: KernelIR, tensor_name: str, dim_id: str, leaf_includes_ptile: bool) -> SbufAxis:
    """Factor this dim's ``num_tiles`` into the four SbufAxis fields."""
    da = ir.dim_analysis
    di = da.dims[dim_id]
    phys = di.physical_tile_size
    ptiles_per_ltile = _max_op_tile(ir, tensor_name, dim_id) // phys
    tier = ir._effective_tier.get(("sbuf", tensor_name, dim_id), "per_tile")
    tpb = get_tpb(ir, dim_id)
    multi_buffer = ir._effective_degree.get(("sbuf", tensor_name, dim_id), 1)
    ltiles_per_block = tpb if tier in ("per_block", "full") else 1
    num_blocks = di.dim_size // (tpb * di.logical_tile_size) if tier == "full" else 1
    return SbufAxis(
        phys=phys,
        ptiles_per_ltile=ptiles_per_ltile,
        ltiles_per_block=ltiles_per_block,
        num_blocks=num_blocks,
        multi_buffer=multi_buffer,
        leaf_includes_ptile=leaf_includes_ptile,
    )


def _max_op_tile(ir: KernelIR, tensor_name: str, dim_id: str) -> int:
    """Widest op tile size seen on ``dim_id`` across all ops touching ``tensor_name``."""
    da = ir.dim_analysis
    touching = ir.op_graph.ops_touching(tensor_name)
    max_tile = da.dims[dim_id].logical_tile_size
    for op_idx in touching:
        op_tile = da.op_tile_sizes[op_idx].get(dim_id)
        if op_tile is not None:
            max_tile = max(max_tile, op_tile)
    return max_tile
