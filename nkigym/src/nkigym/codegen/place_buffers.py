"""Buffer placement: compute tensor SBUF/PSUM shapes from writer footprint.

Per-tensor shape derivation (per-op tiling model):

- The writer's BufferAccess is the authority for tile size along each
  logical dim: ``tile[i] = writer_access.pattern[i].extent``. Non-RMW
  tensors have exactly one writer; RMW tensors (matmul dst) have one
  RMW writer whose extent matches all readers by construction. When a
  tensor has both an RMW and a non-RMW writer (e.g. memset + matmul RMW
  into the same PSUM), the RMW writer wins — it is the finalising op
  and its tile fully partitions the tensor.
- Slot count per dim = ``total_size[dim] // tile[dim]``.
- Cross-nest coverage: find the common prefix of ancestor iter-vars
  across every touching block. Prefix iter-vars reduce the required
  slot count by their extent product.
- Shape: ``(P_tile, num_P_slots, *middle_slots, F_tile * num_F_tiles)``.

Param tensors and return-HBM tensors are untouched — their shape comes
from input_specs / alloc declaration.
"""

from nkigym.ir.ir import BufferAccess, ForNode, IterVar, KernelIR, SBlock, Tensor


def place_buffers(module: KernelIR) -> None:
    """Mutate intermediate SBUF/PSUM tensor shapes in place."""
    for tensor in module.tensors.values():
        if tensor.origin == "param":
            continue
        if tensor.location == "hbm":
            continue
        _place_one(module, tensor)


def _place_one(module: KernelIR, tensor: Tensor) -> None:
    """Derive N-D SBUF/PSUM shape for one intermediate tensor.

    1D logical tensors (e.g. ``reduce_res``, ``operand0``) promote to 3D
    ``(P_tile, num_P_slots, 1)`` so NKI's ≥2D SBUF/PSUM requirement is
    met while the trailing ``1`` preserves the ``(P, 1)`` slice contract
    required by ``nisa.activation_reduce.reduce_res`` and
    ``nisa.tensor_scalar.operand0``.
    """
    accesses = _find_accesses(module, tensor.name)
    if not accesses:
        return
    writer_access = _find_writer_access(module, tensor.name)
    if writer_access is None or not writer_access.pattern:
        return

    tensor_axis_ids: list[int] = []
    for dim_name in tensor.dim_ids:
        try:
            tensor_axis_ids.append(module.axis_id_by_name(dim_name))
        except KeyError:
            tensor_axis_ids.append(-1)

    per_axis_tile: dict[int, int] = {}
    for axis_id, ar in zip(tensor_axis_ids, writer_access.pattern):
        per_axis_tile[axis_id] = ar.extent

    required = _required_tiles(module, accesses, per_axis_tile)
    p_axis = tensor_axis_ids[0]
    p_tile = per_axis_tile[p_axis]
    p_dim_name = tensor.dim_ids[0]
    num_p_slots = required.get(p_axis, 1) * tensor.buffer_degree.get(p_dim_name, 1)

    if len(tensor.dim_ids) == 1:
        tensor.shape = (p_tile, num_p_slots, 1)
        return

    f_axis = tensor_axis_ids[-1]
    f_dim_name = tensor.dim_ids[-1]
    middle_axes = tensor_axis_ids[1:-1]
    middle_dim_names = tensor.dim_ids[1:-1]
    f_tile = per_axis_tile[f_axis]

    middle_slots = [
        required.get(aid, 1) * tensor.buffer_degree.get(dn, 1) for aid, dn in zip(middle_axes, middle_dim_names)
    ]
    num_f_tiles = required.get(f_axis, 1) * tensor.buffer_degree.get(f_dim_name, 1)

    shape_parts: list[int] = [p_tile, num_p_slots, *middle_slots, f_tile * num_f_tiles]
    tensor.shape = tuple(shape_parts)


def _find_writer_access(module: KernelIR, tensor_name: str) -> BufferAccess | None:
    """Return the RMW writer's access if any exists; else the first non-RMW writer.

    Under per-op tile sizes, different writers to the same tensor can
    carry different tile widths (e.g. NKIMemset writes ``psum_acc`` full-F
    while NKIMatmul RMW writes it per-N-tile). The RMW writer is the
    finalising op — its tile fully partitions the tensor and every reader
    sub-slices compatibly. Pick the RMW writer when present; fall back to
    the first non-RMW writer (the tensor has exactly one in that case).

    NKIAlloc blocks produce a writes[output] BufferAccess with empty
    pattern — skip these, as they don't carry tile info. The real writer
    is the first op that populates the buffer (e.g. NKILoad, NKIMemset).
    """
    rmw_writer: BufferAccess | None = None
    first_non_rmw: BufferAccess | None = None

    def walk(node: ForNode | SBlock) -> None:
        nonlocal rmw_writer, first_non_rmw
        if rmw_writer is not None:
            return
        if isinstance(node, SBlock):
            for access in node.reads_writes.values():
                if access.tensor_name == tensor_name and access.pattern:
                    rmw_writer = access
                    return
            for access in node.writes.values():
                if access.tensor_name == tensor_name and access.pattern and first_non_rmw is None:
                    first_non_rmw = access
            return
        for child in node.children:
            walk(child)

    for root in module.body:
        walk(root)
    return rmw_writer if rmw_writer is not None else first_non_rmw


def _find_accesses(module: KernelIR, tensor_name: str) -> list[tuple[SBlock, tuple[IterVar, ...]]]:
    """Return ``(block, ancestor_iter_vars)`` for every SBlock touching ``tensor_name``.

    The ancestor tuple is ordered root-first so the common prefix across
    accesses is the LCA's ancestor chain.
    """
    results: list[tuple[SBlock, tuple[IterVar, ...]]] = []

    def walk(node: ForNode | SBlock, ancestors: tuple[IterVar, ...]) -> None:
        if isinstance(node, SBlock):
            touched = (
                {a.tensor_name for a in node.reads.values()}
                | {a.tensor_name for a in node.writes.values()}
                | {a.tensor_name for a in node.reads_writes.values()}
            )
            if tensor_name in touched:
                results.append((node, ancestors))
            return
        new_ancestors = ancestors + (node.iter_var,)
        for child in node.children:
            walk(child, new_ancestors)

    for root in module.body:
        walk(root, ())
    return results


def _required_tiles(
    module: KernelIR, accesses: list[tuple[SBlock, tuple[IterVar, ...]]], per_axis_tile: dict[int, int]
) -> dict[int, int]:
    """Per-axis required slot count based on common-prefix iter vars.

    ``slots_per_axis = (total_size // tile) // coverage_prefix_product``
    with a floor of 1.
    """
    common = _common_prefix(accesses)
    coverage: dict[int, int] = {}
    for iv in common:
        coverage[iv.axis_id] = coverage.get(iv.axis_id, 1) * iv.extent

    result: dict[int, int] = {}
    for axis_id, axis in module.axes.items():
        tile = per_axis_tile.get(axis_id)
        if tile is None:
            continue
        total_slots = axis.total_size // tile
        cov = coverage.get(axis_id, 1)
        result[axis_id] = max(1, total_slots // cov) if cov > 0 else total_slots
    return result


def _common_prefix(accesses: list[tuple[SBlock, tuple[IterVar, ...]]]) -> tuple[IterVar, ...]:
    """Return the longest common prefix of ancestor chains by iter-var identity."""
    if not accesses:
        return ()
    common = accesses[0][1]
    for _block, ancestors in accesses[1:]:
        new_len = 0
        for a, b in zip(common, ancestors):
            if a.var_id == b.var_id:
                new_len += 1
            else:
                break
        common = common[:new_len]
    return common
