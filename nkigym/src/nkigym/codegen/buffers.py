"""Tensor buffer allocation: on-chip SBUF and PSUM buffers.

Persistent SBUF tensors (touched by 2+ fusion groups) declare at
the top of the first group that touches them. Per-FG SBUF tensors
declare at the top of their owning group. PSUM tensors are per-FG
(PSUM never crosses groups); placement is the compiler's automatic
allocator. SBUF is modeled as a nested Python list ``[NP][NF]`` of
2D ``nl.ndarray`` tiles (see ``sbuf_buffer.SbufBuffer``); PSUM is
a flat 2D ``nl.ndarray``, or a Python list of them when the
producer writes >1 tile.
"""

from nkigym.codegen.sbuf_buffer import SbufBuffer, build_sbuf_buffer
from nkigym.kernel_ir import KernelIR
from nkigym.kernel_ir.dim_analysis import TensorInfo
from nkigym.kernel_ir.ir import build_tensor_to_groups

__all__ = [
    "build_sbuf_buffer",
    "build_tensor_to_groups",
    "find_psum_tensors_needing_sbuf",
    "prime_sbuf_cache",
    "producer_op_tiles",
    "psum_tile_count",
    "psum_tile_shape",
    "psum_tile_slice",
    "render_psum_allocations",
    "render_sbuf_buffers",
    "sbuf_buffer",
    "sbuf_dtype",
]


def render_sbuf_buffers(ir: KernelIR, staged: set[str], tensor_to_groups: dict[str, set[int]]) -> dict[int, list[str]]:
    """Emit SBUF declarations keyed by the fusion group at whose top they appear.

    Persistent tensors (touched by 2+ groups) declare first, at
    the top of the earliest group that touches them. Per-FG
    tensors stack after the persistent range inside their
    owning group. The compiler's automatic allocator handles
    actual byte placement.
    """
    by_group: dict[int, list[str]] = {gi: [] for gi in range(len(ir.fusion_groups))}
    order = ir.op_graph.toposort_groups([g.op_indices for g in ir.fusion_groups])
    group_rank = {gi: rank for rank, gi in enumerate(order)}
    persistent = _persistent_sbuf_tensors(ir, staged, tensor_to_groups)
    for name, _tinfo in persistent:
        first_gi = min(tensor_to_groups[name], key=lambda gi: group_rank[gi])
        by_group[first_gi].append(sbuf_buffer(ir, name).alloc_line())
    for group_idx, names in _per_group_sbuf_tensors(ir, staged, tensor_to_groups).items():
        for name, _tinfo in names:
            by_group[group_idx].append(sbuf_buffer(ir, name).alloc_line())
    return by_group


def render_psum_allocations(ir: KernelIR, op_to_group: dict[int, int], elided: dict[int, str]) -> dict[int, list[str]]:
    """Return ``{group_idx: [alloc_lines]}`` for every PSUM tensor.

    Every PSUM buffer is declared at the top of its fusion group
    (depth 0), alongside the per-FG SBUF declarations. Per-iteration
    zeroing still fires at the blocking depth via the memset that
    ``render_nki_ops`` emits. PSUM tensors within a group pack into
    ``PSUM_BANK_SIZE``-aligned byte offsets starting at 0; banks
    between groups are reclaimed by the compiler because PSUM never
    crosses group boundaries. Tensors produced by ops in ``elided``
    get no PSUM allocation — their fused HBM ``dma_transpose`` load
    writes straight into SBUF.
    """
    da = ir.dim_analysis
    graph = ir.op_graph
    tensor_to_psum_dtype = _build_psum_dtype_map(ir)
    by_group: dict[int, list[str]] = {}
    for name, tinfo in da.tensors.items():
        if graph.producer_isa_loc(name) != "psum":
            continue
        producer = graph.producer_op(name)
        if producer is None or producer in elided:
            continue
        group_idx = op_to_group[producer]
        psum_dtype = tensor_to_psum_dtype.get(name, tinfo.dtype)
        by_group.setdefault(group_idx, []).append(_psum_line(ir, name, tinfo, psum_dtype))
    return by_group


_SBUF_BUFFER_CACHE: dict[int, dict[str, SbufBuffer]] = {}


def sbuf_buffer(ir: KernelIR, name: str) -> SbufBuffer:
    """Return the ``SbufBuffer`` for ``name``, building once per IR.

    The renderer calls this for every tensor access; without the
    cache each call would re-walk the op graph for dtype promotion
    and the per-axis factor decomposition. ``prime_sbuf_cache`` is
    called from ``render_ir`` to populate the cache for every
    tensor with one shared promotion pass.
    """
    cache = _SBUF_BUFFER_CACHE.get(id(ir))
    if cache is None:
        cache = prime_sbuf_cache(ir)
    return cache[name]


def prime_sbuf_cache(ir: KernelIR) -> dict[str, SbufBuffer]:
    """Build every tensor's ``SbufBuffer`` and cache it under ``id(ir)``.

    One shared pass computes the FLOAT32 promotion set; each buffer
    then dispatches with a set-membership check rather than walking
    the op graph per call.
    """
    f32_tensors = _float32_promoted_tensors(ir)
    cache = {
        name: build_sbuf_buffer(ir, name, "float32" if name in f32_tensors else tinfo.dtype)
        for name, tinfo in ir.dim_analysis.tensors.items()
    }
    _SBUF_BUFFER_CACHE[id(ir)] = cache
    return cache


def sbuf_dtype(ir: KernelIR, name: str, tinfo: TensorInfo) -> str:
    """Return the SBUF dtype for one tensor, promoted to float32 when any FLOAT32_KWARGS role consumes it.

    NKI hardware requires ``operand0``/``operand1`` of
    ``nisa.tensor_scalar`` and ``scale`` of ``nisa.activation`` /
    ``nisa.activation_reduce`` to be float32 regardless of the
    data tile's dtype; without promotion the MLIR verifier rejects
    the kernel.
    """
    return "float32" if name in _float32_promoted_tensors(ir) else tinfo.dtype


def _float32_promoted_tensors(ir: KernelIR) -> set[str]:
    """Return the set of tensor names that any op consumes via a ``FLOAT32_KWARGS`` role."""
    graph = ir.op_graph
    promoted: set[str] = set()
    for op_idx, op_cls in enumerate(graph.op_classes):
        f32_roles = op_cls.FLOAT32_KWARGS
        if not f32_roles:
            continue
        all_kwargs = graph.op_all_kwargs[op_idx]
        for role in f32_roles:
            raw = all_kwargs.get(role)
            if raw is None:
                continue
            value = raw[1:-1] if raw.startswith("'") and raw.endswith("'") else raw
            if value in ir.dim_analysis.tensors:
                promoted.add(value)
    return promoted


def find_psum_tensors_needing_sbuf(ir: KernelIR) -> set[str]:
    """Find PSUM tensors that need an SBUF staging buffer.

    A PSUM tensor needs staging when:
    1. A consumer op requires SBUF for the operand that reads
       this tensor (``INPUT_LOCS[role] == "sbuf"``).
    2. The tensor is the return tensor (needs dma_copy to HBM,
       which reads from SBUF).
    """
    da = ir.dim_analysis
    graph = ir.op_graph
    result: set[str] = set()
    for consumer_idx, (inputs, _outputs) in enumerate(graph.op_tensors):
        input_locs = graph.op_classes[consumer_idx].INPUT_LOCS
        for role, tensor_name in inputs.items():
            if tensor_name not in da.tensors:
                continue
            if graph.producer_isa_loc(tensor_name) != "psum":
                continue
            if input_locs.get(role) == "sbuf":
                result.add(tensor_name)
    if graph.producer_isa_loc(da.return_name) == "psum":
        result.add(da.return_name)
    return result


def producer_op_tiles(ir: KernelIR, tensor_name: str) -> dict[str, int]:
    """Producing op's tile sizes for a tensor, or ``{}`` for kernel inputs."""
    producer = ir.op_graph.producer_op(tensor_name)
    return ir.dim_analysis.op_tile_sizes[producer] if producer is not None else {}


def psum_tile_shape(ir: KernelIR, tensor_name: str, tinfo: TensorInfo) -> tuple[int, ...]:
    """PSUM tile shape — the producer op's own output tile size.

    Falls back to the dim's physical tile size when the producer
    has no ``TILE_LIMITS`` for that axis.
    """
    da = ir.dim_analysis
    dim_ids = tinfo.dim_ids
    if len(dim_ids) not in (1, 2):
        raise ValueError(f"Tensor {tensor_name} has {len(dim_ids)} dims, expected 1 or 2")
    op_tiles = producer_op_tiles(ir, tensor_name)
    return tuple(op_tiles.get(d, da.dims[d].physical_tile_size) for d in dim_ids)


def psum_tile_slice(ir: KernelIR, tensor_name: str, tinfo: TensorInfo) -> str:
    """Bare ``[0:P, 0:F]`` (or ``[0:P]``) slice covering one PSUM tile at the producer's tile size."""
    shape = psum_tile_shape(ir, tensor_name, tinfo)
    if len(shape) == 2:
        expr = f"[0:{shape[0]}, 0:{shape[1]}]"
    elif len(shape) == 1:
        expr = f"[0:{shape[0]}]"
    else:
        raise ValueError(f"PSUM tile must be 1D or 2D, got {len(shape)}D")
    return expr


def psum_tile_count(ir: KernelIR, tensor_name: str, tinfo: TensorInfo) -> int:
    """Number of distinct PSUM tiles held simultaneously — product of per-dim PSUM buffer degrees.

    Baseline (``buffer_degrees[("psum", t, d)] == 1`` everywhere)
    gives count ``1``: a single PSUM ``nl.ndarray``. Multi-buffering
    (degree > 1 on a dim) emits a Python list sized to the product
    of degrees.
    """
    dim_ids = tinfo.dim_ids
    if len(dim_ids) not in (1, 2):
        raise ValueError(f"Tensor {tensor_name} has {len(dim_ids)} dims, expected 1 or 2")
    count = 1
    for d in dim_ids:
        count *= ir._effective_degree.get(("psum", tensor_name, d), 1)
    return count


def _persistent_sbuf_tensors(
    ir: KernelIR, staged: set[str], tensor_to_groups: dict[str, set[int]]
) -> list[tuple[str, TensorInfo]]:
    """SBUF-needing tensors touched by 2+ fusion groups, in IR order."""
    return [
        (name, tinfo)
        for name, tinfo in ir.dim_analysis.tensors.items()
        if _needs_sbuf(ir, name, staged) and len(tensor_to_groups.get(name, ())) >= 2
    ]


def _per_group_sbuf_tensors(
    ir: KernelIR, staged: set[str], tensor_to_groups: dict[str, set[int]]
) -> dict[int, list[tuple[str, TensorInfo]]]:
    """SBUF-needing tensors touched by exactly one fusion group, keyed by that group."""
    result: dict[int, list[tuple[str, TensorInfo]]] = {gi: [] for gi in range(len(ir.fusion_groups))}
    for name, tinfo in ir.dim_analysis.tensors.items():
        if not _needs_sbuf(ir, name, staged):
            continue
        groups = tensor_to_groups.get(name, set())
        if len(groups) == 1:
            result[next(iter(groups))].append((name, tinfo))
    return result


def _needs_sbuf(ir: KernelIR, name: str, staged: set[str]) -> bool:
    """True iff ``name`` needs an SBUF buffer (non-PSUM, or a staged PSUM tensor)."""
    is_psum = ir.op_graph.producer_isa_loc(name) == "psum"
    return (not is_psum) or (name in staged)


def _psum_line(ir: KernelIR, name: str, tinfo: TensorInfo, dtype: str) -> str:
    """Emit a PSUM allocation.

    Single-tile → one ``nl.ndarray``. Multi-tile → Python list of
    ``nl.ndarray``.
    """
    shape = psum_tile_shape(ir, name, tinfo)
    shape_str = ", ".join(str(s) for s in shape)
    count = psum_tile_count(ir, name, tinfo)
    tile_expr = f"nl.ndarray(({shape_str}), dtype=nl.{dtype}, buffer=nl.psum)"
    rhs = tile_expr if count == 1 else f"[{tile_expr} for _ in range({count})]"
    return f"psum_{name} = {rhs}"


def _build_psum_dtype_map(ir: KernelIR) -> dict[str, str]:
    """Map tensor names to PSUM dtype overrides.

    Only tensors produced by ops with ``PSUM_DTYPE`` set (e.g.
    nc_matmul → float32) get an override.
    """
    result: dict[str, str] = {}
    for op_idx, op_cls in enumerate(ir.op_graph.op_classes):
        if op_cls.PSUM_DTYPE is None:
            continue
        _, outputs = ir.op_graph.op_tensors[op_idx]
        for tensor_name in outputs:
            result[tensor_name] = op_cls.PSUM_DTYPE
    return result
