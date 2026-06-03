"""Loop-var + trip-1 normalization for a block subtree.

After a transform (Split/Fuse) mutates a block's ForNode chain,
:func:`normalize_block` restores the two IR invariants:

* No trip-1 ForNodes — a trip-1 loop is removed, its children re-linked to
  its parent (the axis becomes loopless; its extent folds into the access
  tile width, already set by the transform).
* Dense position-in-dim names — each dim's surviving ForNodes are named
  ``i_d{dim}_{N}`` with N the loop's ordinal among that dim's loops,
  outer-to-inner.

iter_values and every region ``lo`` in the block are then RECOMPUTED (not
merely substituted) from the dim's surviving dense loops:

* The tile-space affine for a dim with loops ``l_0(t_0) … l_{k-1}(t_{k-1})``
  (outer-to-inner) is ``T = Σ_j l_j · Π(t_i for i > j)`` — the pure
  factorization, ranging over the dim's full loop space. A loopless dim has
  ``T = 0``.
* A block ``iter_value`` is exactly ``T`` (tile space, stride unit 1).
* A region axis ``lo`` is the bare ``T`` for the SBUF/PSUM partition axis
  (axis 0, width 128 — ``T`` is a tile index) and ``Mul(T, width)`` for every
  element-space axis. Keeping the offset un-flattened as ``Mul(affine,
  width)`` mirrors :func:`nkigym.ir.canonical_build._build_region`, so a
  normalize on canonical IR is the identity, a Split that only inserts loops
  and sets the access width renders byte-exact to the hand-written ladder,
  and the element stride stays recoverable by Fuse's outer-trip merge.
"""

from __future__ import annotations

from math import prod

from nkigym.ir.arith.expr import Const, Expr, Mul, from_affine, to_affine
from nkigym.ir.tree import PARTITION_DIM, BlockNode, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.transforms._tree_ops import _block_local_descendants, _replace_in_parent_children


def _enclosing_loops(tree: KernelTree, block_nid: int) -> list[tuple[str, int]]:
    """Return the ForNodes the block is nested under, as ``(loop_var, extent)`` outer-to-inner.

    A block moved by ComputeAt becomes nested inside the target's loop chain;
    the dims it shares with the target (its "covered" dims) are then driven by
    those ENCLOSING loops rather than by loops the block owns locally. This
    helper walks ``tree.ancestors(block_nid)`` (root-first = outer-first) and
    keeps the ForNodes strictly below the block's nearest BlockNode ancestor —
    i.e. the loops between the block and its enclosing BlockNode.

    For a top-level block (a direct child of the root BlockNode with no
    op-loops above it), the only ancestor is the empty root BlockNode, so the
    result is empty and every caller behaves exactly as before — Split / Fuse /
    Reorder are unaffected.
    """
    out: list[tuple[str, int]] = []
    for anc in tree.ancestors(block_nid):
        data = tree.data(anc)
        if isinstance(data, BlockNode):
            out = []
            continue
        if isinstance(data, ForNode):
            out.append((data.loop_var, data.extent))
    return out


def _iter_value_loopvars(block: BlockNode) -> set[str]:
    """Return the loop_vars that appear in the block's ``iter_values``.

    These are the loops genuinely driving the block's dims: a covered dim's
    iter_value (set by ``regen_and_rebind``) references the enclosing loop var
    it binds to, while an enclosing loop the block does not index never
    appears. Restricting the enclosing-loop gather to this set keeps unrelated
    ancestor loops (e.g. a consumer's free-axis loop the moved block reads in
    full) from being treated as tiling the moved block.
    """
    out: set[str] = set()
    for value in block.iter_values:
        for name in to_affine(value):
            if name is not None:
                out.add(name)
    return out


def normalize_block(tree: KernelTree, block_nid: int) -> None:
    """Drop trip-1 ForNodes and re-densify loop-var names in this block's subtree."""
    _drop_trip1(tree, block_nid)
    _rename_dense(tree, block_nid)
    _recompute_bindings(tree, block_nid)


def _drop_trip1(tree: KernelTree, block_nid: int) -> None:
    """Remove every trip-1 ForNode under the block, re-linking children to the parent."""
    trivial = [
        n
        for n in _block_local_descendants(tree, block_nid)
        if isinstance(tree.data(n), ForNode) and tree.data(n).extent == 1
    ]
    for nid in trivial:
        parent = tree.parent(nid)
        children = tree.children(nid)
        _replace_in_parent_children(tree, parent, [nid], children)
        tree.graph.remove_node(nid)


def _enclosing_dim_counts(tree: KernelTree, block_nid: int, block: BlockNode) -> dict[str, int]:
    """Count, per dim, the block's ENCLOSING loops that drive that dim.

    Same enclosing notion :func:`_dim_loops` uses: the ForNodes a ComputeAt move
    nested the block under (``_enclosing_loops``), restricted to the loop vars the
    block actually binds (``_iter_value_loopvars``). A covered/enclosing loop and a
    locally-regenerated residual loop on the SAME dim are distinct loops, so the
    local loops must continue the dim's ordinal AFTER these enclosing ones — else a
    residual would be renamed to the enclosing loop's name and collide. For a
    top-level block (Split / Fuse / Reorder) there are no enclosing op-loops, so
    every count is 0 and local ordinals start at 0 exactly as before.
    """
    old_to_dim = _loopvar_to_dim(tree, block_nid, block)
    bound = _iter_value_loopvars(block)
    out: dict[str, int] = {}
    for loop_var, _extent in _enclosing_loops(tree, block_nid):
        if loop_var not in bound:
            continue
        dim = old_to_dim[loop_var]
        out[dim] = out.get(dim, 0) + 1
    return out


def _rename_dense(tree: KernelTree, block_nid: int) -> None:
    """Rename each dim's surviving LOCAL ForNodes to dense ``i_d{dim}_{N}`` (outer-to-inner).

    A dim's ordinal CONTINUES across the block's enclosing same-dim loops: ``N``
    starts at the count of bound enclosing loops on the dim, so a residual loop
    regenerated below an enclosing target loop gets the next ordinal (``i_d1_1``
    after enclosing ``i_d1_0``) rather than re-using and colliding with it.
    Enclosing loops belong to the parent block's scope and are left untouched.
    """
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode)
    old_to_dim = _loopvar_to_dim(tree, block_nid, block)
    counters: dict[str, int] = dict(_enclosing_dim_counts(tree, block_nid, block))
    for nid in _block_local_descendants(tree, block_nid):
        data = tree.data(nid)
        if not isinstance(data, ForNode):
            continue
        dim = old_to_dim.get(data.loop_var)
        if dim is None:
            continue
        n = counters.get(dim, 0)
        counters[dim] = n + 1
        new_name = f"i_{dim}_{n}"
        if new_name != data.loop_var:
            tree.graph.nodes[nid]["data"] = ForNode(loop_var=new_name, extent=data.extent)


def _recompute_bindings(tree: KernelTree, block_nid: int) -> None:
    """Recompute iter_values + every region ``lo`` from the renamed dense loops.

    Each dim's iter_value and region offsets are rebuilt from the dim's
    surviving dense loops (their element strides), so the transform's loop
    edits + access-width edits are sufficient — the offsets never have to be
    set by the transform.
    """
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode)
    dim_loops = _dim_loops(tree, block_nid, block)
    tensor_axes = _tensor_to_axes(tree, block_nid)
    new_iter_values = tuple(_iter_value(iv.axis, dim_loops) for iv in block.iter_vars)
    new_block = BlockNode(
        iter_vars=block.iter_vars,
        iter_values=new_iter_values,
        reads=tuple(_recompute_region(tree, r, tensor_axes, block.axis_map, dim_loops) for r in block.reads),
        writes=tuple(_recompute_region(tree, w, tensor_axes, block.axis_map, dim_loops) for w in block.writes),
        alloc_buffers=block.alloc_buffers,
        annotations=dict(block.annotations),
        axis_map=block.axis_map,
    )
    tree.graph.nodes[block_nid]["data"] = new_block
    for nid in _block_local_descendants(tree, block_nid):
        data = tree.data(nid)
        if not isinstance(data, ISANode):
            continue
        op_axes = data.op_cls.OPERAND_AXES
        new_bindings = {
            slot: _recompute_region(tree, region, {region.tensor: op_axes[slot]}, block.axis_map, dim_loops)
            for slot, region in data.operand_bindings.items()
        }
        tree.graph.nodes[nid]["data"] = ISANode(
            op_cls=data.op_cls, operand_bindings=new_bindings, kwargs=dict(data.kwargs)
        )


def _dim_loops(tree: KernelTree, block_nid: int, block: BlockNode) -> dict[str, list[tuple[str, int]]]:
    """Map each concrete dim to its driving loops as ``(loop_var, extent)`` outer-to-inner.

    A dim's loop list is the block's ENCLOSING ForNodes on that dim (the loops
    a ComputeAt move nested the block under — outer) followed by the block's
    own surviving dense loops on that dim (inner). For a top-level block the
    enclosing list is empty, so the result is exactly the block-local loops as
    before.
    """
    old_to_dim = _loopvar_to_dim(tree, block_nid, block)
    bound = _iter_value_loopvars(block)
    out: dict[str, list[tuple[str, int]]] = {}
    for loop_var, extent in _enclosing_loops(tree, block_nid):
        if loop_var not in bound:
            continue
        out.setdefault(old_to_dim[loop_var], []).append((loop_var, extent))
    for nid in _block_local_descendants(tree, block_nid):
        data = tree.data(nid)
        if not isinstance(data, ForNode):
            continue
        dim = old_to_dim.get(data.loop_var) or _dim_from_loopvar(data.loop_var)
        out.setdefault(dim, []).append((data.loop_var, data.extent))
    return out


def _iter_value(dim: str, dim_loops: dict[str, list[tuple[str, int]]]) -> Expr:
    """Tile-space affine for ``dim`` (stride unit 1); ``Const(0)`` if the dim is loopless."""
    return _tile_space_affine(dim_loops.get(dim, []))


def _tile_space_affine(loops: list[tuple[str, int]]) -> Expr:
    """Build ``Σ_j loop_j · Π(extent_i for i inner to j)`` over ``loops`` (outer-to-inner).

    The innermost loop has stride 1; each outer loop strides by the product
    of all extents nested inside it. An empty ``loops`` yields ``Const(0)``.
    This is the pure factorization an :class:`IterVar` binds (its iter_value)
    and the base a region offset scales by its element width.
    """
    coeffs: dict[str | None, int] = {None: 0}
    for j, (loop_var, _extent) in enumerate(loops):
        inner_extents = [extent for _v, extent in loops[j + 1 :]]
        coeffs[loop_var] = prod(inner_extents)
    return from_affine(coeffs)


def _recompute_region(
    tree: KernelTree,
    region: BufferRegion,
    tensor_axes: dict[str, tuple[str, ...]],
    axis_map: dict[str, str],
    dim_loops: dict[str, list[tuple[str, int]]],
) -> BufferRegion:
    """Rebuild each axis ``lo`` of ``region`` from its dim's dense loops at the stored width.

    The width on each range is left untouched (the transform owns it). The
    SBUF/PSUM partition axis (axis 0, width 128) carries the bare tile-space
    affine — a tile index, stride unit 1. Every element-space axis carries
    that affine scaled by its element width, stored as ``Mul(affine, width)``
    so the offset structurally mirrors
    :func:`nkigym.ir.canonical_build._build_region`. A region whose tensor
    has no known operand axes is returned unchanged.
    """
    abstract_axes = tensor_axes.get(region.tensor)
    if abstract_axes is None:
        return region
    present = [a for a in abstract_axes if a in axis_map]
    location = _tensor_location(tree, region.tensor)
    new_ranges: list[tuple[Expr, Expr]] = []
    for axis_index, (_lo, width) in enumerate(region.ranges):
        assert isinstance(width, Const), f"region width must be Const; got {width!r}"
        dim = axis_map.get(present[axis_index]) if axis_index < len(present) else None
        affine = _tile_space_affine(dim_loops.get(dim, []) if dim is not None else [])
        is_partition = axis_index == 0 and location in ("sbuf", "psum") and width.value == PARTITION_DIM
        lo = affine if (is_partition or _is_zero(affine)) else Mul(left=affine, right=width)
        new_ranges.append((lo, width))
    return BufferRegion(tensor=region.tensor, ranges=tuple(new_ranges))


def _is_zero(expr: Expr) -> bool:
    """True when ``expr`` is the constant 0 (a loopless axis offsets at 0)."""
    return isinstance(expr, Const) and expr.value == 0


def _tensor_to_axes(tree: KernelTree, block_nid: int) -> dict[str, tuple[str, ...]]:
    """Map each operand tensor in the block to its op's abstract axis tuple.

    Built from the block's ISA leaves (``op_cls.OPERAND_AXES[slot]`` keyed by
    the slot's bound tensor name). Used to project block reads/writes — which
    are keyed by tensor, not slot — back onto operand axes.
    """
    out: dict[str, tuple[str, ...]] = {}
    for nid in _block_local_descendants(tree, block_nid):
        data = tree.data(nid)
        if not isinstance(data, ISANode):
            continue
        for slot, region in data.operand_bindings.items():
            out[region.tensor] = data.op_cls.OPERAND_AXES[slot]
    return out


def _tensor_location(tree: KernelTree, tensor: str) -> str:
    """Return the residency of ``tensor`` from any block's ``alloc_buffers``.

    Kernel parameters carry no ``Buffer`` (they are never allocated), so a
    tensor not declared anywhere is an HBM parameter — ``shared_hbm``.
    """
    for nid in tree.blocks():
        block = tree.data(nid)
        assert isinstance(block, BlockNode)
        for buf in block.alloc_buffers:
            if buf.name == tensor:
                return buf.location
    return "shared_hbm"


def _loopvar_to_dim(tree: KernelTree, block_nid: int, block: BlockNode) -> dict[str, str]:
    """Map each ForNode loop_var in the block to the concrete dim it binds.

    A loop_var binds the iter_var whose iter_value affine contains it. Since
    iter_values are affine over a single dim's loops, the loop_var->dim map is
    each loop_var to the iter_var.axis of the iter_value mentioning it.
    """
    out: dict[str, str] = {}
    for iv, value in zip(block.iter_vars, block.iter_values):
        for name in to_affine(value).keys():
            if name is not None:
                out[name] = iv.axis
    """Fallback for loop_vars not yet in iter_values (freshly inserted by a split):
    parse the stem i_d{dim}_N -> d{dim}."""
    for nid in _block_local_descendants(tree, block_nid):
        data = tree.data(nid)
        if isinstance(data, ForNode) and data.loop_var not in out:
            out[data.loop_var] = _dim_from_loopvar(data.loop_var)
    return out


def _dim_from_loopvar(loop_var: str) -> str:
    """i_d1_0 / i_d1_0_0 -> d1. Strip the i_ prefix and trailing _<int> suffixes."""
    body = loop_var[2:] if loop_var.startswith("i_") else loop_var
    parts = body.split("_")
    return parts[0]


__all__ = ["normalize_block"]
