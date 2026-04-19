"""Tensor buffer allocation: on-chip SBUF and PSUM buffers.

SBUF and PSUM buffers are declared with explicit
``address=(base_partition, byte_offset)`` arguments and placed at
the tightest Python scope that still covers their live range:

* Persistent SBUF tensors (touched by 2+ fusion groups) declare at
  kernel top and occupy low addresses that stay pinned for the
  whole kernel.
* Per-FG SBUF tensors (touched by exactly one fusion group) declare
  at the top of that group's loop nest; each group's per-FG region
  starts right after the persistent range, and the range is reused
  by other groups' per-FG tensors.
* PSUM tensors are always per-FG (PSUM never crosses groups) and
  declare at the depth where their producer's outermost blocking
  loop opens. PSUM addresses are packed into ``PSUM_BANK_SIZE``
  banks within each group.

SBUF is a multi-D ``nl.ndarray``. PSUM is a flat 2D ``nl.ndarray``
(or a Python list of them when the producer writes >1 tile) —
multi-D PSUM ndarrays tripped spurious reshape failures in the
NKI simulator.
"""

from nkigym.kernel_ir import KernelIR, get_tpb
from nkigym.kernel_ir.dim_analysis import TensorInfo
from nkigym.kernel_ir.ir import build_tensor_to_groups

__all__ = ["build_tensor_to_groups", "render_sbuf_buffers"]

PSUM_BANK_SIZE = 2048
"""Bytes per PSUM bank; each allocation is aligned up to this boundary."""

_DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "uint32": 4,
    "int16": 2,
    "uint16": 2,
    "int8": 1,
    "uint8": 1,
    "float8_e4m3": 1,
    "float8_e5m2": 1,
}


def render_sbuf_buffers(ir: KernelIR, staged: set[str], tensor_to_groups: dict[str, set[int]]) -> dict[int, list[str]]:
    """Emit SBUF declarations keyed by the fusion group at whose top they appear.

    Each SBUF tensor is declared at depth 0 of the first fusion
    group (in toposort order) that touches it. Persistent tensors
    (touched by 2+ groups) occupy low byte offsets that stay
    pinned for the rest of the kernel because the NKI compiler
    honors the explicit address verbatim. Per-FG tensors (touched
    by exactly one group) stack after the persistent range inside
    that group — per-FG regions overlap across groups so the
    compiler reuses the same addresses for tensors local to
    different groups.
    """
    by_group: dict[int, list[str]] = {gi: [] for gi in range(len(ir.fusion_groups))}
    order = ir.op_graph.toposort_groups([g.op_indices for g in ir.fusion_groups])
    group_rank = {gi: rank for rank, gi in enumerate(order)}
    persistent_offset = 0
    persistent_end = sum(
        _sbuf_bytes(ir, name, tinfo) for name, tinfo in _persistent_sbuf_tensors(ir, staged, tensor_to_groups)
    )
    for name, tinfo in _persistent_sbuf_tensors(ir, staged, tensor_to_groups):
        first_gi = min(tensor_to_groups[name], key=lambda gi: group_rank[gi])
        by_group[first_gi].append(_sbuf_line(ir, name, tinfo, pad="", offset=persistent_offset))
        persistent_offset += _sbuf_bytes(ir, name, tinfo)
    for group_idx, names in _per_group_sbuf_tensors(ir, staged, tensor_to_groups).items():
        offset = persistent_end
        for name, tinfo in names:
            by_group[group_idx].append(_sbuf_line(ir, name, tinfo, pad="", offset=offset))
            offset += _sbuf_bytes(ir, name, tinfo)
    return by_group


def render_psum_allocations(ir: KernelIR, op_to_group: dict[int, int]) -> dict[int, list[str]]:
    """Return ``{group_idx: [alloc_lines]}`` for every PSUM tensor.

    Every PSUM buffer is declared at the top of its fusion group
    (depth 0), alongside the per-FG SBUF declarations. Per-iteration
    zeroing still fires at the blocking depth via the memset that
    ``render_nki_ops`` emits. PSUM tensors within a group pack into
    ``PSUM_BANK_SIZE``-aligned byte offsets starting at 0; banks
    between groups are reclaimed by the compiler because PSUM never
    crosses group boundaries.
    """
    da = ir.dim_analysis
    graph = ir.op_graph
    tensor_to_psum_dtype = _build_psum_dtype_map(ir)
    by_group: dict[int, list[str]] = {}
    psum_offsets: dict[int, int] = {}
    for name, tinfo in da.tensors.items():
        if graph.producer_isa_loc(name) != "psum":
            continue
        producer = graph.producer_op(name)
        if producer is None:
            continue
        group_idx = op_to_group[producer]
        psum_dtype = tensor_to_psum_dtype.get(name, tinfo.dtype)
        offset = psum_offsets.get(group_idx, 0)
        by_group.setdefault(group_idx, []).append(_psum_line(ir, name, tinfo, psum_dtype, pad="", offset=offset))
        psum_offsets[group_idx] = offset + _psum_bytes(ir, name, tinfo, psum_dtype)
    return by_group


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


def _sbuf_bytes(ir: KernelIR, tensor_name: str, tinfo: TensorInfo) -> int:
    """Byte footprint of an SBUF tensor on one partition (product of free dims × dtype)."""
    shape = sbuf_shape(ir, tensor_name, tinfo)
    free_elems = 1
    for dim in shape[1:]:
        free_elems *= dim
    return free_elems * _dtype_bytes(sbuf_dtype(ir, tensor_name, tinfo))


def _psum_bank_step(shape: tuple[int, ...], dtype: str) -> int:
    """Per-tile byte footprint rounded up to a full PSUM bank."""
    free_elems = 1
    for dim in shape[1:]:
        free_elems *= dim
    tile_bytes = free_elems * _dtype_bytes(dtype)
    return ((tile_bytes + PSUM_BANK_SIZE - 1) // PSUM_BANK_SIZE) * PSUM_BANK_SIZE


def _psum_bytes(ir: KernelIR, tensor_name: str, tinfo: TensorInfo, dtype: str) -> int:
    """Byte footprint of one PSUM allocation, rounded up to a full bank per tile."""
    shape = psum_tile_shape(ir, tensor_name, tinfo)
    count = psum_tile_count(ir, tensor_name, tinfo)
    return _psum_bank_step(shape, dtype) * count


def _dtype_bytes(dtype: str) -> int:
    """Byte size of one element in ``dtype``."""
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"Unknown dtype {dtype!r}")
    return _DTYPE_BYTES[dtype]


def _sbuf_line(ir: KernelIR, name: str, tinfo: TensorInfo, pad: str, offset: int) -> str:
    """Emit a single SBUF ``nl.ndarray`` allocation line.

    ``offset`` is currently unused — the compiler's automatic
    allocator handles SBUF placement. Kept in the signature so
    callers can pass through their cursor without branching.
    """
    del offset
    shape = sbuf_shape(ir, name, tinfo)
    shape_str = ", ".join(str(s) for s in shape)
    dtype = sbuf_dtype(ir, name, tinfo)
    return f"{pad}sbuf_{name} = nl.ndarray(({shape_str}), dtype=nl.{dtype}, buffer=nl.sbuf)"


def sbuf_dtype(ir: KernelIR, name: str, tinfo: TensorInfo) -> str:
    """Promote the SBUF dtype to float32 if any consumer reads it through a FLOAT32_KWARGS role.

    NKI hardware requires ``operand0``/``operand1`` of
    ``nisa.tensor_scalar`` and ``scale`` of ``nisa.activation`` /
    ``nisa.activation_reduce`` to be float32 regardless of the
    data tile's dtype. If this tensor feeds any such role in any
    op, the backing SBUF buffer must be float32 or the compiler
    rejects the MLIR. Otherwise the logical dtype from
    ``dim_analysis`` stands.
    """
    graph = ir.op_graph
    promote = False
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
            if value == name:
                promote = True
    return "float32" if promote else tinfo.dtype


def _psum_line(ir: KernelIR, name: str, tinfo: TensorInfo, dtype: str, pad: str, offset: int) -> str:
    """Emit a PSUM allocation.

    Single-tile → one ``nl.ndarray``. Multi-tile → Python list of
    ``nl.ndarray``. ``offset`` is unused; the compiler's allocator
    handles PSUM bank placement.
    """
    del offset
    shape = psum_tile_shape(ir, name, tinfo)
    shape_str = ", ".join(str(s) for s in shape)
    count = psum_tile_count(ir, name, tinfo)
    tile_expr = f"nl.ndarray(({shape_str}), dtype=nl.{dtype}, buffer=nl.psum)"
    rhs = tile_expr if count == 1 else f"[{tile_expr} for _ in range({count})]"
    return f"{pad}psum_{name} = {rhs}"


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


def sbuf_shape(ir: KernelIR, tensor_name: str, tinfo: TensorInfo) -> tuple[int, ...]:
    """SBUF shape for a tensor.

    2D tensor → 4D: ``(phys_P, num_tiles_P, num_tiles_F, phys_F)``.
    1D tensor → 2D: ``(phys_P, num_tiles_P)``.
    """
    da = ir.dim_analysis
    dim_ids = tinfo.dim_ids

    if len(dim_ids) not in (1, 2):
        raise ValueError(f"Tensor {tensor_name} has {len(dim_ids)} dims, expected 1 or 2")

    shape: tuple[int, ...] = ()
    if len(dim_ids) == 2:
        d_p, d_f = dim_ids[0], dim_ids[1]
        tile_p = da.dims[d_p].physical_tile_size
        tile_f = da.dims[d_f].physical_tile_size
        num_tiles_p = num_tiles(ir, tensor_name, d_p)
        num_tiles_f = num_tiles(ir, tensor_name, d_f)
        shape = (tile_p, num_tiles_p, num_tiles_f, tile_f)
    elif len(dim_ids) == 1:
        d_p = dim_ids[0]
        tile_p = da.dims[d_p].physical_tile_size
        num_tiles_p = num_tiles(ir, tensor_name, d_p)
        shape = (tile_p, num_tiles_p)

    return shape


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
    gives count ``1``: a single PSUM ``nl.ndarray``. The ptile
    loop (emitted by codegen when the producer's op tile is
    narrower than the dim's logical tile) reuses that one PSUM
    across iterations and stages to SBUF inside the loop.
    Multi-buffering (degree > 1 on a dim) emits a Python list of
    size equal to the product of degrees so that concurrent
    iterations can alias different tiles.
    """
    dim_ids = tinfo.dim_ids
    if len(dim_ids) not in (1, 2):
        raise ValueError(f"Tensor {tensor_name} has {len(dim_ids)} dims, expected 1 or 2")
    count = 1
    for d in dim_ids:
        count *= _effective_buffer_degree(ir, "psum", tensor_name, d)
    return count


def num_tiles(ir: KernelIR, tensor_name: str, dim_id: str) -> int:
    """Derive num_tiles for one dimension from KernelIR fields.

    num_tiles = num_ptiles * tpb_factor * blocks_factor * buffer_degree

    | tier       | tpb_factor | blocks_factor              |
    |------------|------------|----------------------------|
    | per_tile   | 1          | 1                          |
    | per_block  | tpb        | 1                          |
    | full       | tpb        | num_blocks                 |
    """
    da = ir.dim_analysis
    di = da.dims[dim_id]

    max_op_tile = _max_op_tile_for_tensor(ir, tensor_name, dim_id)
    num_ptiles = max_op_tile // di.physical_tile_size

    tier = _effective_sbuf_tier(ir, tensor_name, dim_id)
    tpb = get_tpb(ir, dim_id)
    degree = _effective_buffer_degree(ir, "sbuf", tensor_name, dim_id)

    if tier == "per_tile":
        tpb_factor = 1
        blocks_factor = 1
    elif tier == "per_block":
        tpb_factor = tpb
        blocks_factor = 1
    elif tier == "full":
        tpb_factor = tpb
        blocks_factor = di.dim_size // (tpb * di.logical_tile_size)
    else:
        raise ValueError(f"Unknown tensor_placement tier: {tier!r}")

    return num_ptiles * tpb_factor * blocks_factor * degree


def _effective_sbuf_tier(ir: KernelIR, tensor_name: str, dim_id: str) -> str:
    """Return the widest SBUF tier this tensor sees across the groups that touch it.

    A single physical SBUF buffer backs all uses of a tensor. Its
    shape must fit the widest tier requested by any group that
    touches the tensor. Groups that don't have an entry for the
    tensor are ignored (they don't use this buffer).
    """
    return ir._effective_tier.get(("sbuf", tensor_name, dim_id), "per_tile")


def _effective_buffer_degree(ir: KernelIR, kind: str, tensor_name: str, dim_id: str) -> int:
    """Return the max multi-buffer degree this ``(kind, tensor, dim)`` sees across the groups that touch it.

    A single physical buffer backs all uses of the tensor within
    a group (PSUM) or across groups (SBUF). The buffer's slot
    count must fit the largest degree requested.
    """
    return ir._effective_degree.get((kind, tensor_name, dim_id), 1)


def _max_op_tile_for_tensor(ir: KernelIR, tensor_name: str, dim_id: str) -> int:
    """Find the largest op tile size on a dimension across all ops touching a tensor."""
    da = ir.dim_analysis
    ops = _ops_for_tensor(ir, tensor_name)
    max_tile = da.dims[dim_id].logical_tile_size
    for op_idx in ops:
        op_tile = da.op_tile_sizes[op_idx].get(dim_id)
        if op_tile is not None:
            max_tile = max(max_tile, op_tile)
    return max_tile


def producer_op_tiles(ir: KernelIR, tensor_name: str) -> dict[str, int]:
    """Producing op's tile sizes for a tensor, or ``{}`` for kernel inputs."""
    producer = ir.op_graph.producer_op(tensor_name)
    return ir.dim_analysis.op_tile_sizes[producer] if producer is not None else {}


def _ops_for_tensor(ir: KernelIR, tensor_name: str) -> list[int]:
    """Find all op indices that touch a tensor (as input or output)."""
    result = ir.op_graph.ops_touching(tensor_name)
    if not result:
        raise ValueError(f"No ops touch tensor {tensor_name!r}")
    return result
