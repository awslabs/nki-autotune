"""Buffer placement: compute tensor SBUF/PSUM shapes from iter-var LCA walk.

Per-tensor shape derivation:

- For each ``SBlock`` that touches this tensor, collect enclosing
  ``ForNode`` iter vars (from forest root down to the block).
- Find the common iter-var prefix across all accesses -- these iter
  vars are ancestors above the LCA. Per-dim coverage = product of
  extents of common-prefix iter vars on that dim.
- ``required_tiles[dim] = num_tiles[dim] / coverage[dim]``.
- ``num_slots[dim] = required_tiles[dim] * tensor.buffer_degree.get(dim, 1)``.
- Shape: ``(P_tile, num_P_slots, *middle_slots, F_tile * num_F_tiles)``.

N-D unified path: trivial dims (``num_slots == 1``) stay explicit per
Q5 decision. A single-tile 2D tensor emits ``(P_tile, 1, F_tile)``.

Param tensors and return-HBM tensors are untouched -- their shape comes
from input_specs / alloc declaration.
"""

from nkigym.codegen.ir import ForNode, IterVar, KernelModule, SBlock, Tensor


def place_buffers(module: KernelModule) -> None:
    """Mutate intermediate SBUF/PSUM tensor shapes in place.

    Args:
        module: KernelModule with iter-var-based body.
    """
    for tensor in module.tensors.values():
        if tensor.origin == "param":
            continue
        if tensor.location == "hbm":
            continue
        _place_one(module, tensor)


def _place_one(module: KernelModule, tensor: Tensor) -> None:
    """Derive N-D SBUF/PSUM shape for one intermediate tensor.

    Emits trivial dims explicitly -- a (P, F) tensor with num_P=1 still
    renders as ``(P_tile, 1, F_tile * num_F_tiles)``. 1D logical tensors
    (e.g. ``reduce_res``, ``operand0``) promote to 3D ``(P_tile,
    num_P_slots, 1)`` so NKI's ≥2D SBUF/PSUM requirement is met while the
    trailing ``1`` preserves the ``(P, 1)`` slice contract required by
    ``nisa.activation_reduce.reduce_res`` and ``nisa.tensor_scalar.operand0``.
    """
    accesses = _find_accesses(module, tensor.name)
    if not accesses:
        return
    required = _required_tiles(module, accesses)
    p_dim = tensor.dim_ids[0]

    p_tile = module.dims[p_dim].tile_size
    num_p_slots = required.get(p_dim, 1) * tensor.buffer_degree.get(p_dim, 1)

    if len(tensor.dim_ids) == 1:
        tensor.shape = (p_tile, num_p_slots, 1)
        return

    f_dim = tensor.dim_ids[-1]
    middle_dims = tensor.dim_ids[1:-1]
    f_tile = module.dims[f_dim].tile_size

    middle_slots = [required.get(d, 1) * tensor.buffer_degree.get(d, 1) for d in middle_dims]
    num_f_tiles = required.get(f_dim, 1) * tensor.buffer_degree.get(f_dim, 1)

    shape_parts: list[int] = [p_tile, num_p_slots, *middle_slots, f_tile * num_f_tiles]
    tensor.shape = tuple(shape_parts)


def _find_accesses(module: KernelModule, tensor_name: str) -> list[tuple[SBlock, tuple[IterVar, ...]]]:
    """Return ``(block, ancestor_iter_vars)`` for every SBlock touching ``tensor_name``.

    The ancestor tuple is ordered root-first so the common prefix across
    accesses is the LCA's ancestor chain.
    """
    results: list[tuple[SBlock, tuple[IterVar, ...]]] = []

    def walk(node: ForNode | SBlock, ancestors: tuple[IterVar, ...]) -> None:
        """Pre-order DFS collecting ancestor iter vars for every touching block."""
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


def _required_tiles(module: KernelModule, accesses: list[tuple[SBlock, tuple[IterVar, ...]]]) -> dict[str, int]:
    """Per-dim required tile count based on common-prefix iter vars.

    Find the common prefix (by iter-var identity) across all access
    chains -- those are the iter vars above the LCA. Product of their
    extents on each dim = covered portion; divide ``num_tiles`` by it.

    Dims absent from the common prefix get full ``num_tiles`` (cross-nest).
    """
    common = _common_prefix(accesses)
    coverage: dict[str, int] = {}
    for iv in common:
        coverage[iv.dim_id] = coverage.get(iv.dim_id, 1) * iv.extent

    result: dict[str, int] = {}
    for dim_id, dim in module.dims.items():
        cov = coverage.get(dim_id, 1)
        if cov > 0:
            result[dim_id] = max(1, dim.num_tiles // cov)
        else:
            result[dim_id] = dim.num_tiles
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
