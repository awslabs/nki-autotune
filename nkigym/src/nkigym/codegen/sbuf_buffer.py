"""SBUF buffer model: list-of-lists of 2D physical tiles."""

from dataclasses import dataclass

from nkigym.kernel_ir import KernelIR


@dataclass(frozen=True)
class SbufAxis:
    """Per-axis slot structure for an SBUF buffer."""

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
        """Total physical tiles on this axis."""
        return self.list_slots * (self.ptiles_per_ltile if self.leaf_includes_ptile else 1)


@dataclass(frozen=True)
class AxisAccess:
    """How a caller wants to iterate one axis of an SBUF buffer at a given scope."""

    block: str | None = None
    ltile: str | None = None
    ptile: str | None = None


@dataclass(frozen=True)
class SbufBuffer:
    """SBUF buffer backed by a nested Python list of 2D ``nl.ndarray`` leaves."""

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
        """Return an access string for one physical 2D tile."""
        p_idx = _list_index(self.p, p.block, p.ltile, p.ptile)
        f_idx = _list_index(self.f, f.block, f.ltile, None)
        f_slice = _f_ptile_slice(self.f, f.ptile) if f.ptile is not None else f"0:{self.f.logical}"
        return f"sbuf_{self.name}[{p_idx}][{f_idx}][0:{self.p.logical}, {f_slice}]"

    def range(self, p: AxisAccess, f: AxisAccess) -> tuple[str, int, str, int]:
        """Return ``(p_start, p_count, f_start, f_count)`` for a gadget sub-block."""
        p_start, p_count = _list_range(self.p, p.block, p.ltile, p.ptile)
        f_start, f_count = _list_range(self.f, f.block, f.ltile, None)
        return p_start, p_count, f_start, f_count


def _list_index(axis: SbufAxis, block: str | None, ltile: str | None, ptile: str | None) -> str:
    """Collapse the list-level indices into a flat expression."""
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
    """Return ``(start_expr, count)`` for a contiguous list-slot range on one axis."""
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
            raise ValueError("cannot bind an inner factor while an outer factor is unbound")
        terms.append(idx if stride == 1 else f"{idx} * {stride}")
    start = " + ".join(terms) if terms else "0"
    return start, count


def _list_strides(
    axis: SbufAxis, block: str | None, ltile: str | None, ptile: str | None
) -> tuple[list[int], list[int], list[str | None]]:
    """Return per-factor ``(strides, factors, idx_sources)`` for list-level indexing."""
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
    """Construct the ``SbufBuffer`` for a logical tensor."""
    context = ir.context
    tinfo = context.logical_tensors[tensor_name]
    dim_ids = tinfo.dim_ids
    if len(dim_ids) not in (1, 2):
        raise ValueError(f"Tensor {tensor_name} has {len(dim_ids)} dims, expected 1 or 2")
    effective_tier = _effective_tier_map(ir)
    effective_degree = _effective_degree_map(ir)
    p_axis = _build_axis(ir, tensor_name, dim_ids[0], False, effective_tier, effective_degree)
    if len(dim_ids) == 2:
        f_axis = _build_axis(ir, tensor_name, dim_ids[1], True, effective_tier, effective_degree)
    else:
        f_axis = SbufAxis(
            phys=1, ptiles_per_ltile=1, ltiles_per_block=1, num_blocks=1, multi_buffer=1, leaf_includes_ptile=True
        )
    return SbufBuffer(name=tensor_name, dtype=dtype, p=p_axis, f=f_axis)


def _effective_tier_map(ir: KernelIR) -> dict[tuple[str, str, str], str]:
    """Cross-group widest tier per ``(kind, tensor, dim)``."""
    ranks = {"per_tile": 0, "per_block": 1, "full": 2}
    result: dict[tuple[str, str, str], str] = {}
    for group in ir.graph.groups:
        for key, tier in group.tensor_placements.items():
            prev = result.get(key)
            if prev is None or ranks[tier] > ranks[prev]:
                result[key] = tier
    return result


def _effective_degree_map(ir: KernelIR) -> dict[tuple[str, str, str], int]:
    """Cross-group max degree per ``(kind, tensor, dim)``."""
    result: dict[tuple[str, str, str], int] = {}
    for group in ir.graph.groups:
        for key, deg in group.buffer_degrees.items():
            prev = result.get(key, 0)
            if deg > prev:
                result[key] = deg
    return result


def _build_axis(
    ir: KernelIR,
    tensor_name: str,
    dim_id: str,
    leaf_includes_ptile: bool,
    effective_tier: dict[tuple[str, str, str], str],
    effective_degree: dict[tuple[str, str, str], int],
) -> SbufAxis:
    """Factor this dim's ``num_tiles`` into the four SbufAxis fields."""
    context = ir.context
    di = context.dimensions[dim_id]
    phys = di.physical_tile_size
    ptiles_per_ltile = _max_op_tile(ir, tensor_name, dim_id) // phys
    tier = effective_tier.get(("sbuf", tensor_name, dim_id), "per_tile")
    tpb = context.ltiles_per_block.get(dim_id, 1)
    multi_buffer = effective_degree.get(("sbuf", tensor_name, dim_id), 1)
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
    context = ir.context
    max_tile = context.dimensions[dim_id].logical_tile_size
    for group in ir.graph.groups:
        for op in group.ops:
            touched = [*context.op_inputs.get(op, {}).values(), *context.op_outputs.get(op, [])]
            if tensor_name not in touched:
                continue
            op_tile = context.op_tile_sizes.get(op, {}).get(dim_id)
            if op_tile is not None:
                max_tile = max(max_tile, op_tile)
    return max_tile
