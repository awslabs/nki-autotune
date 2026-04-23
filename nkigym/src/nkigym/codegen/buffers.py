"""Tensor buffer allocation: on-chip SBUF and PSUM buffers."""

from nkigym.codegen.matmul_block_detect import is_matmul_block_candidate
from nkigym.codegen.sbuf_buffer import SbufBuffer, buffer_ident, build_sbuf_buffer
from nkigym.kernel_ir import KernelIR
from nkigym.kernel_ir.context.context import KernelContext, TensorInfo
from nkigym.kernel_ir.graph.graph import KernelGraph
from nkigym.ops.base import NKIOp

__all__ = [
    "build_sbuf_buffer",
    "build_tensor_to_groups",
    "find_psum_tensors_needing_sbuf",
    "prime_sbuf_cache",
    "producer_op",
    "producer_op_tiles",
    "psum_tile_count",
    "psum_tile_shape",
    "psum_tile_slice",
    "render_psum_allocations",
    "render_sbuf_buffers",
    "sbuf_buffer",
    "sbuf_dtype",
]

_SBUF_BUFFER_CACHE: dict[int, dict[str, SbufBuffer]] = {}


def render_sbuf_buffers(
    ir: KernelIR, staged: set[str], tensor_to_groups: dict[str, set[int]]
) -> dict[int, dict[int, list[str]]]:
    """Emit SBUF declarations keyed by ``{group_idx: {depth: [lines]}}``.

    Alloc depth follows the tensor's tightest-tier dim:
    a tensor that is ``per_block`` on some dim in the owning
    group gets its allocation pushed inside that dim's block
    loop so the compiler sees fresh-per-iteration buffers —
    matching the ``nki_matmul_fully_optimized_`` pattern when
    a matmul-block gadget dispatches on the same group.
    """
    groups = ir.graph.groups
    by_group: dict[int, dict[int, list[str]]] = {gi: {} for gi in range(len(groups))}
    order = ir.graph.toposort_groups()
    group_rank = {gi: rank for rank, gi in enumerate(order)}
    persistent = _persistent_sbuf_tensors(ir, staged, tensor_to_groups)
    for name, _tinfo in persistent:
        first_gi = min(tensor_to_groups[name], key=lambda gi: group_rank[gi])
        depth = _alloc_depth(ir, name, first_gi)
        by_group[first_gi].setdefault(depth, []).append(sbuf_buffer(ir, name).alloc_line())
    for group_idx, names in _per_group_sbuf_tensors(ir, staged, tensor_to_groups).items():
        for name, _tinfo in names:
            depth = _alloc_depth(ir, name, group_idx)
            by_group[group_idx].setdefault(depth, []).append(sbuf_buffer(ir, name).alloc_line())
    return by_group


def _alloc_depth(ir: KernelIR, tensor_name: str, group_idx: int) -> int:
    """Return the slot depth at which ``tensor_name`` should be allocated in ``group_idx``.

    Kernel-top (``0``) if every placement dim in this group is
    ``full``; otherwise innermost ``block_depth(pos) + 1`` across
    all ``per_block``/``per_tile`` dims — i.e. the buffer is
    declared fresh each iteration of the tightest block loop it
    depends on.
    """
    placements = ir.graph.groups[group_idx].tensor_placements
    dim_order = ir.graph.groups[group_idx].dim_order
    tinfo = ir.context.logical_tensors[tensor_name]
    depth = 0
    for dim_id in tinfo.dim_ids:
        if dim_id not in dim_order:
            continue
        tier = placements.get(("sbuf", tensor_name, dim_id), "per_tile")
        if tier == "full":
            continue
        pos = dim_order.index(dim_id)
        depth = max(depth, 2 * pos + 1)
    return depth


def render_psum_allocations(ir: KernelIR, op_to_group: dict[int, int]) -> dict[int, list[str]]:
    """Return ``{group_idx: [alloc_lines]}`` for every PSUM tensor.

    Skips ops dispatched to the matmul_block gadget: matmul_block
    allocates its own PSUM scratch internally, so emitting a
    module-level PSUM declaration would be dead code.
    """
    context = ir.context
    psum_dtype_map = _build_psum_dtype_map(ir)
    by_group: dict[int, list[str]] = {}
    for name, tinfo in context.logical_tensors.items():
        producer = producer_op(ir, name)
        if producer is None or type(producer).ISA_LOC != "psum":
            continue
        gi = op_to_group[id(producer)]
        if is_matmul_block_candidate(ir, producer, gi):
            continue
        psum_dtype = psum_dtype_map.get(name, tinfo.dtype)
        by_group.setdefault(gi, []).append(_psum_line(ir, name, tinfo, psum_dtype))
    return by_group


def sbuf_buffer(ir: KernelIR, name: str) -> SbufBuffer:
    """Return the ``SbufBuffer`` for ``name``, building once per IR."""
    cache = _SBUF_BUFFER_CACHE.get(id(ir))
    if cache is None:
        cache = prime_sbuf_cache(ir)
    return cache[name]


def prime_sbuf_cache(ir: KernelIR) -> dict[str, SbufBuffer]:
    """Build every tensor's ``SbufBuffer`` and cache it under ``id(ir)``."""
    f32_tensors = _float32_promoted_tensors(ir)
    cache = {
        name: build_sbuf_buffer(ir, name, "float32" if name in f32_tensors else tinfo.dtype)
        for name, tinfo in ir.context.logical_tensors.items()
    }
    _SBUF_BUFFER_CACHE[id(ir)] = cache
    return cache


def sbuf_dtype(ir: KernelIR, name: str, tinfo: TensorInfo) -> str:
    """Return the SBUF dtype for one tensor, promoted to float32 when a FLOAT32_KWARGS role consumes it."""
    return "float32" if name in _float32_promoted_tensors(ir) else tinfo.dtype


def _float32_promoted_tensors(ir: KernelIR) -> set[str]:
    """Tensors consumed via a ``FLOAT32_KWARGS`` role anywhere in the graph."""
    context = ir.context
    promoted: set[str] = set()
    for op in context.op_inputs:
        f32_roles = type(op).FLOAT32_KWARGS
        if not f32_roles:
            continue
        kwargs = context.op_kwargs.get(op, {})
        for role in f32_roles:
            raw = kwargs.get(role)
            if raw is None:
                continue
            value = raw[1:-1] if raw.startswith("'") and raw.endswith("'") else raw
            if value in context.logical_tensors:
                promoted.add(value)
    return promoted


def find_psum_tensors_needing_sbuf(ir: KernelIR) -> set[str]:
    """PSUM-produced tensors that need an SBUF staging buffer."""
    context = ir.context
    graph = ir.graph
    result: set[str] = set()
    for group in graph.groups:
        for op in group.ops:
            input_locs = type(op).INPUT_LOCS
            for role, tname in context.op_inputs.get(op, {}).items():
                if tname not in context.logical_tensors:
                    continue
                prod = producer_op(ir, tname)
                if prod is None or type(prod).ISA_LOC != "psum":
                    continue
                if input_locs.get(role) == "sbuf":
                    result.add(tname)
    ret_producer = producer_op(ir, context.return_name)
    if ret_producer is not None and type(ret_producer).ISA_LOC == "psum":
        result.add(context.return_name)
    return result


def producer_op(ir: KernelIR, tensor_name: str) -> NKIOp | None:
    """Return the op producing ``tensor_name`` or None."""
    result: NKIOp | None = None
    for group in ir.graph.groups:
        for op in group.ops:
            if tensor_name in ir.context.op_outputs.get(op, []):
                result = op
                break
        if result is not None:
            break
    return result


def producer_op_tiles(ir: KernelIR, tensor_name: str) -> dict[str, int]:
    """Producing op's tile sizes for a tensor, or ``{}`` for kernel inputs."""
    prod = producer_op(ir, tensor_name)
    return dict(ir.context.op_tile_sizes.get(prod, {})) if prod is not None else {}


def psum_tile_shape(ir: KernelIR, tensor_name: str, tinfo: TensorInfo) -> tuple[int, ...]:
    """PSUM tile shape — the producer op's own output tile size."""
    context = ir.context
    dim_ids = tinfo.dim_ids
    if len(dim_ids) not in (1, 2):
        raise ValueError(f"Tensor {tensor_name} has {len(dim_ids)} dims")
    op_tiles = producer_op_tiles(ir, tensor_name)
    return tuple(op_tiles.get(d, context.dimensions[d].physical_tile_size) for d in dim_ids)


def psum_tile_slice(ir: KernelIR, tensor_name: str, tinfo: TensorInfo) -> str:
    """Bare ``[0:P, 0:F]`` (or ``[0:P]``) slice for one PSUM tile."""
    shape = psum_tile_shape(ir, tensor_name, tinfo)
    if len(shape) == 2:
        expr = f"[0:{shape[0]}, 0:{shape[1]}]"
    elif len(shape) == 1:
        expr = f"[0:{shape[0]}]"
    else:
        raise ValueError(f"PSUM tile must be 1D or 2D, got {len(shape)}D")
    return expr


def psum_tile_count(ir: KernelIR, tensor_name: str, tinfo: TensorInfo) -> int:
    """Number of distinct PSUM tiles held simultaneously."""
    dim_ids = tinfo.dim_ids
    if len(dim_ids) not in (1, 2):
        raise ValueError(f"Tensor {tensor_name} has {len(dim_ids)} dims")
    max_deg_map = _max_degree_map(ir)
    count = 1
    for d in dim_ids:
        count *= max_deg_map.get(("psum", tensor_name, d), 1)
    return count


def _max_degree_map(ir: KernelIR) -> dict[tuple[str, str, str], int]:
    """Cross-group widest degree per ``(kind, tensor, dim)``."""
    result: dict[tuple[str, str, str], int] = {}
    for group in ir.graph.groups:
        for key, deg in group.buffer_degrees.items():
            prev = result.get(key, 0)
            if deg > prev:
                result[key] = deg
    return result


def _persistent_sbuf_tensors(
    ir: KernelIR, staged: set[str], tensor_to_groups: dict[str, set[int]]
) -> list[tuple[str, TensorInfo]]:
    """SBUF-needing tensors touched by 2+ groups, in IR order."""
    return [
        (name, tinfo)
        for name, tinfo in ir.context.logical_tensors.items()
        if _needs_sbuf(ir, name, staged) and len(tensor_to_groups.get(name, ())) >= 2
    ]


def _per_group_sbuf_tensors(
    ir: KernelIR, staged: set[str], tensor_to_groups: dict[str, set[int]]
) -> dict[int, list[tuple[str, TensorInfo]]]:
    """SBUF-needing tensors touched by exactly one fusion group, keyed by that group."""
    result: dict[int, list[tuple[str, TensorInfo]]] = {gi: [] for gi in range(len(ir.graph.groups))}
    for name, tinfo in ir.context.logical_tensors.items():
        if not _needs_sbuf(ir, name, staged):
            continue
        groups = tensor_to_groups.get(name, set())
        if len(groups) == 1:
            result[next(iter(groups))].append((name, tinfo))
    return result


def _needs_sbuf(ir: KernelIR, name: str, staged: set[str]) -> bool:
    """True iff ``name`` needs an SBUF buffer.

    Excluded from SBUF allocation:

    * Kernel-input ``param_names`` — raw HBM inputs whose SBUF
      counterparts are ``<name>_sbuf`` tensors produced by the
      Load op inserted by ``insert_dma_nodes``.
    * Tensors produced by an op with ``ISA_LOC == "hbm"`` (e.g.
      ``NKIStore``'s ``<name>_hbm`` output). These live in HBM,
      not SBUF.
    * PSUM-produced tensors that no consumer stages through SBUF.
    """
    is_param = name in ir.context.param_names
    prod = producer_op(ir, name)
    isa_loc = type(prod).ISA_LOC if prod is not None else None
    is_hbm_producer = isa_loc == "hbm"
    is_unstaged_psum = isa_loc == "psum" and name not in staged
    return not (is_param or is_hbm_producer or is_unstaged_psum)


def _psum_line(ir: KernelIR, name: str, tinfo: TensorInfo, dtype: str) -> str:
    """Emit a PSUM allocation."""
    shape = psum_tile_shape(ir, name, tinfo)
    shape_str = ", ".join(str(s) for s in shape)
    count = psum_tile_count(ir, name, tinfo)
    tile_expr = f"nl.ndarray(({shape_str}), dtype=nl.{dtype}, buffer=nl.psum)"
    rhs = tile_expr if count == 1 else f"[{tile_expr} for _ in range({count})]"
    return f"psum_{buffer_ident(name)} = {rhs}"


def _build_psum_dtype_map(ir: KernelIR) -> dict[str, str]:
    """Map tensor names to PSUM dtype overrides."""
    result: dict[str, str] = {}
    for op in ir.context.op_outputs:
        dtype = type(op).PSUM_DTYPE
        if dtype is None:
            continue
        for tname in ir.context.op_outputs.get(op, []):
            result[tname] = dtype
    return result


def build_tensor_to_groups(ir: KernelIR) -> dict[str, set[int]]:
    """Map tensor names → set of group indices whose ops touch them."""
    context = ir.context
    result: dict[str, set[int]] = {}
    for gi, group in enumerate(ir.graph.groups):
        for op in group.ops:
            names = [*context.op_inputs.get(op, {}).values(), *context.op_outputs.get(op, [])]
            for name in names:
                if name in context.logical_tensors:
                    result.setdefault(name, set()).add(gi)
    return result


_ = (KernelContext, KernelGraph)
