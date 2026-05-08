"""``PlaceBuffers`` pass: derive buffer shapes and placement metadata from IR.

Runs before source emission. The pass walks the schedule tree to compute,
for every intermediate tensor:

* :func:`required_tiles` — the minimum number of tiles a tensor must hold
  along a given dim, derived from the LCA of its producer/consumer paths.
  For cross-nest tensors (LCA at the forest root) this is
  ``num_tiles(dim_id)``; for fully intra-nest tensors (LCA below all
  dim-iterating ancestors) this is ``1``.
* :func:`sbuf_shape` — the 3D SBUF allocation shape
  ``(p_tile, total_slots_P, num_f_tiles * f_tile)``, where
  ``total_slots_P`` multiplies :func:`required_tiles` by the multi-buffer
  degree.
* :func:`tensor_total_slots` — per-dim slot count
  (``required_tiles * buffer_degree``); consumed by body emitters to
  build slot-indexing expressions.

Shared with :mod:`nkigym.tune.multi_buffer`, which uses
:func:`required_tiles` for legality checks: a tensor's multi-buffer
degree may never exceed ``num_tiles(dim_id) / required_tiles(dim_id)``
without producing slot-modulo aliasing in the rendered kernel.
"""

from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, Tensor


def required_tiles(tensor: Tensor, dim_id: str, module: KernelModule) -> int:
    """Return the minimum tile count along ``dim_id`` that ``tensor`` must hold.

    Derived by walking the module body: find the LCA of ``tensor``'s
    producer and all consumers, then take
    ``num_tiles(dim_id) / product_of_dim_id_trips_above_lca``. For
    cross-loopnest tensors (LCA is the forest root) the product is 1
    and ``required_tiles`` equals ``num_tiles(dim_id)``. For fully
    intra-loopnest tensors (LCA below all ``dim_id``-iterating
    ancestors) the product equals ``num_tiles(dim_id)`` and
    ``required_tiles`` is ``1``.

    Parameter and return tensors — which live in HBM and are not
    tile-decomposed — return ``module.dims[dim_id].num_tiles`` unchanged.

    Args:
        tensor: Target tensor.
        dim_id: Dim along which to size the tile count.
        module: Enclosing :class:`KernelModule`.

    Returns:
        The minimum tile count along ``dim_id`` this tensor must hold.

    Raises:
        ValueError: Product of ancestor trip counts does not divide
            ``num_tiles(dim_id)`` (unsupported forest shape).
    """
    num_t = module.dims[dim_id].num_tiles
    if tensor.origin in ("param", "return"):
        return num_t
    paths = _find_access_paths(tensor.name, module)
    if not paths:
        return num_t
    lca = _lowest_common_ancestor(paths)
    prod = 1
    for node in lca:
        if isinstance(node, LoopNode) and node.dim_id == dim_id:
            prod *= node.trip_count
    if num_t % prod != 0:
        raise ValueError(
            f"Tensor {tensor.name!r} dim {dim_id!r}: ancestor trip product {prod} does not divide num_tiles {num_t}"
        )
    return num_t // prod


def sbuf_shape(tensor: Tensor, module: KernelModule) -> tuple[int, int, int]:
    """Compute 3D SBUF shape ``(p_tile, total_slots_P, num_f_tiles * f_tile)``.

    ``total_slots_P = required_tiles(P) * buffer_degree[P]``. Free axis
    still spans the full tile count for now — free-axis multi-buffer is
    out of scope.

    1D tensors collapse the free axis to a single element.

    Args:
        tensor: Target tensor.
        module: Enclosing :class:`KernelModule`.

    Returns:
        3D SBUF shape tuple.

    Raises:
        ValueError: Tensor has no declared dims.
    """
    if not tensor.dim_ids:
        raise ValueError(f"Tensor {tensor.name!r} has no dims")
    p_axis = tensor.dim_ids[0]
    p_info = module.dims[p_axis]
    p_required = required_tiles(tensor, p_axis, module)
    p_total = p_required * tensor.buffer_degree[p_axis]
    if len(tensor.dim_ids) == 1:
        return (p_info.tile_size, p_total, 1)
    f_axis = tensor.dim_ids[1]
    f_info = module.dims[f_axis]
    return (p_info.tile_size, p_total, f_info.num_tiles * f_info.tile_size)


def tensor_total_slots(tensor: Tensor, dim_id: str, module: KernelModule) -> int:
    """Per-dim total slot count for a tensor: ``required_tiles * buffer_degree``.

    Args:
        tensor: Target tensor.
        dim_id: Dim along which to compute the slot count.
        module: Enclosing :class:`KernelModule`.

    Returns:
        The total number of tile-slots allocated for ``tensor`` on ``dim_id``.
    """
    return required_tiles(tensor, dim_id, module) * tensor.buffer_degree[dim_id]


def _find_access_paths(tensor_name: str, module: KernelModule) -> list[list[LoopNode | BodyLeaf]]:
    """Return root-to-leaf node paths for every BodyLeaf that reads or writes ``tensor_name``.

    Each path is a list of ancestor nodes ending in the ``BodyLeaf``.
    Walks the module body directly — leaves are self-describing so we
    filter by each leaf's ``reads``/``writes`` sets.
    """
    paths: list[list[LoopNode | BodyLeaf]] = []

    def walk(node: LoopNode | BodyLeaf, stack: list[LoopNode | BodyLeaf]) -> None:
        """Pre-order walk; record paths whose leaves touch ``tensor_name``."""
        stack.append(node)
        if isinstance(node, BodyLeaf):
            if tensor_name in node.writes or tensor_name in node.reads.values():
                paths.append(list(stack))
        else:
            for child in node.children:
                walk(child, stack)
        stack.pop()

    for root in module.body:
        walk(root, [])
    return paths


def _lowest_common_ancestor(paths: list[list[LoopNode | BodyLeaf]]) -> list[LoopNode | BodyLeaf]:
    """Return the longest common prefix of root-to-leaf paths."""
    if not paths:
        return []
    common = paths[0]
    for p in paths[1:]:
        new_len = 0
        for a, b in zip(common, p):
            if a is b:
                new_len += 1
            else:
                break
        common = common[:new_len]
    return common
