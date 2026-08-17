"""Normalize loop names, bindings, and regions after structural rewrites."""

from __future__ import annotations

from math import prod

from nkigym.ir.arith.expr import Const, Expr, Mul, Var, from_affine, substitute, to_affine
from nkigym.ir.tree import PARTITION_DIM, AccessPattern, BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.transforms.helper.tree_ops import _block_local_descendants, _replace_in_parent_children


def _iter_value_loopvars(block: BlockNode) -> set[str]:
    """Return loop variables that drive the block's dimensions or regions."""
    values = (
        *block.iter_values,
        *(lower for region in (*block.reads, *block.writes) for lower, _width in region.ranges),
    )
    return {name for value in values for name in to_affine(value) if name is not None}


def normalize_block(tree: KernelTree, block_nid: int) -> None:
    """Drop trip-1 ForNodes and re-densify loop-var names in this block's subtree."""
    _drop_trip1(tree, block_nid)
    _rename_dense(tree, block_nid)
    _recompute_bindings(tree, block_nid)


def normalize_selected_tensor_regions(tree: KernelTree, tensors: frozenset[str]) -> None:
    """Recompute regions for ``tensors`` in one traversal of every block."""
    buffers = {buffer.name: buffer for nid in tree.blocks() for buffer in tree.block(nid).alloc_buffers}
    for block_nid in tree.blocks():
        _recompute_bindings(tree, block_nid, tensors=tensors, buffers=buffers)


def _drop_trip1(tree: KernelTree, block_nid: int) -> None:
    """Remove every trip-1 ForNode under the block, re-linking children to the parent."""
    trivial = [
        n
        for n in _block_local_descendants(tree, block_nid)
        if isinstance((node := tree.data(n)), ForNode) and node.extent == 1
    ]
    for nid in trivial:
        parent = tree.parent(nid)
        assert parent is not None
        children = tree.children(nid)
        _replace_in_parent_children(tree, parent, [nid], children)
        tree.graph.remove_node(nid)


def _enclosing_dim_counts(tree: KernelTree, block_nid: int) -> dict[str, int]:
    """Count enclosing loops per dimension to avoid rendered-name collisions."""
    out: dict[str, int] = {}
    for loop_var, _extent in _all_enclosing_loops(tree, block_nid):
        dim = _dim_from_loopvar(loop_var)
        out[dim] = out.get(dim, 0) + 1
    return out


def _all_enclosing_loops(tree: KernelTree, block_nid: int) -> list[tuple[str, int]]:
    """Return every enclosing loop, crossing intervening block boundaries."""
    return [
        (node.loop_var, node.extent)
        for anc in tree.ancestors(block_nid)
        if isinstance((node := tree.data(anc)), ForNode)
    ]


def _rename_dense(tree: KernelTree, block_nid: int) -> None:
    """Rename local loops densely after all enclosing same-dimension loops."""
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode)
    old_to_dim = _loopvar_to_dim(tree, block_nid, block)
    counters: dict[str, int] = dict(_enclosing_dim_counts(tree, block_nid))
    substitutions: dict[str, Expr] = {}
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
            substitutions[data.loop_var] = Var(name=new_name)
            tree.graph.nodes[nid]["data"] = ForNode(loop_var=new_name, extent=data.extent)
    if substitutions:
        _substitute_block_regions(tree, block_nid, substitutions)


def _substitute_block_regions(tree: KernelTree, block_nid: int, substitutions: dict[str, Expr]) -> None:
    """Rename loop variables in regions before semantic-axis recomputation."""
    block = tree.block(block_nid)

    def rewrite(region: BufferRegion) -> BufferRegion:
        """Apply ``substitutions`` to every range expression."""
        return BufferRegion(
            tensor=region.tensor,
            ranges=tuple(
                (substitute(lower, substitutions), substitute(width, substitutions)) for lower, width in region.ranges
            ),
        )

    def rewrite_pattern(pattern: AccessPattern) -> AccessPattern:
        """Apply ``substitutions`` to one explicit physical view."""
        return AccessPattern(
            pattern=tuple(
                (substitute(stride, substitutions), substitute(extent, substitutions))
                for stride, extent in pattern.pattern
            ),
            offset=substitute(pattern.offset, substitutions),
        )

    tree.graph.nodes[block_nid]["data"] = BlockNode(
        iter_vars=block.iter_vars,
        iter_values=tuple(substitute(value, substitutions) for value in block.iter_values),
        reads=tuple(rewrite(region) for region in block.reads),
        writes=tuple(rewrite(region) for region in block.writes),
        alloc_buffers=block.alloc_buffers,
        annotations=dict(block.annotations),
        axis_map=block.axis_map,
    )
    for nid in _block_local_descendants(tree, block_nid):
        node = tree.data(nid)
        if isinstance(node, ISANode):
            tree.graph.nodes[nid]["data"] = ISANode(
                op_cls=node.op_cls,
                operand_bindings={slot: rewrite(region) for slot, region in node.operand_bindings.items()},
                kwargs=dict(node.kwargs),
                access_patterns={slot: rewrite_pattern(pattern) for slot, pattern in node.access_patterns.items()},
            )


def _recompute_bindings(
    tree: KernelTree, block_nid: int, tensors: frozenset[str] | None = None, buffers: dict[str, Buffer] | None = None
) -> None:
    """Recompute iteration bindings and selected regions from dense loops."""
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode)
    buffer_map = (
        {buffer.name: buffer for nid in tree.blocks() for buffer in tree.block(nid).alloc_buffers}
        if buffers is None
        else buffers
    )
    dim_loops = _dim_loops(tree, block_nid, block)
    tensor_axes = _tensor_to_axes(tree, block_nid)
    new_iter_values = (
        block.iter_values if tensors is not None else tuple(_iter_value(iv.axis, dim_loops) for iv in block.iter_vars)
    )

    def recompute(region: BufferRegion) -> BufferRegion:
        """Recompute ``region`` when it belongs to the selected tensors."""
        result = region
        if tensors is None or region.tensor in tensors:
            result = _recompute_region(region, tensor_axes, block.axis_map, dim_loops, buffer_map)
        return result

    new_block = BlockNode(
        iter_vars=block.iter_vars,
        iter_values=new_iter_values,
        reads=tuple(recompute(region) for region in block.reads),
        writes=tuple(recompute(region) for region in block.writes),
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
            slot: (
                _recompute_region(region, {region.tensor: op_axes[slot]}, block.axis_map, dim_loops, buffer_map)
                if tensors is None or region.tensor in tensors
                else region
            )
            for slot, region in data.operand_bindings.items()
        }
        tree.graph.nodes[nid]["data"] = ISANode(
            op_cls=data.op_cls,
            operand_bindings=new_bindings,
            kwargs=dict(data.kwargs),
            access_patterns=dict(data.access_patterns),
        )


def _dim_loops(tree: KernelTree, block_nid: int, block: BlockNode) -> dict[str, list[tuple[str, int]]]:
    """Map each dimension to its driving loops from outermost to innermost."""
    old_to_dim = _loopvar_to_dim(tree, block_nid, block)
    bound = _iter_value_loopvars(block)
    out: dict[str, list[tuple[str, int]]] = {}
    for loop_var, extent in _all_enclosing_loops(tree, block_nid):
        if loop_var not in bound:
            continue
        out.setdefault(old_to_dim.get(loop_var, _dim_from_loopvar(loop_var)), []).append((loop_var, extent))
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
    """Build the tile-space affine expression for outer-to-inner loops."""
    coeffs: dict[str | None, int] = {None: 0}
    for j, (loop_var, _extent) in enumerate(loops):
        inner_extents = [extent for _v, extent in loops[j + 1 :]]
        coeffs[loop_var] = prod(inner_extents)
    return from_affine(coeffs)


def _fit_loops(loops: list[tuple[str, int]], capacity: int) -> list[tuple[str, int]]:
    """Keep the longest innermost loop suffix that fits the tile capacity."""
    kept: list[tuple[str, int]] = []
    running = 1
    for loop_var, extent in reversed(loops):
        running *= extent
        if running > capacity:
            break
        kept.append((loop_var, extent))
    kept.reverse()
    return kept


def _recompute_region(
    region: BufferRegion,
    tensor_axes: dict[str, tuple[str, ...]],
    axis_map: dict[str, str],
    dim_loops: dict[str, list[tuple[str, int]]],
    buffers: dict[str, Buffer],
) -> BufferRegion:
    """Rebuild region offsets from dense loops while retaining access widths."""
    abstract_axes = tensor_axes.get(region.tensor)
    if abstract_axes is None:
        return region
    present = [a for a in abstract_axes if a in axis_map]
    buf = buffers.get(region.tensor)
    location = "shared_hbm" if buf is None else buf.location
    new_ranges: list[tuple[Expr, Expr]] = []
    for axis_index, (old_lo, width) in enumerate(region.ranges):
        assert isinstance(width, Const), f"region width must be Const; got {width!r}"
        if axis_index >= len(present):
            new_ranges.append((old_lo, width))
            continue
        dim = axis_map.get(present[axis_index])
        loops = dim_loops.get(dim, []) if dim is not None else []
        loops = _fit_loops(loops, _axis_capacity(buf, axis_index, location, width.value))
        affine = _tile_space_affine(loops)
        is_partition = axis_index == 0 and location in ("sbuf", "psum")
        partition_extent = buf.partition_extent() if is_partition and buf is not None else PARTITION_DIM
        if is_partition and width.value % partition_extent != 0:
            raise ValueError(
                f"{region.tensor}: partition-axis width {width.value} must be a multiple of {partition_extent}"
            )
        partition_tiles = width.value // partition_extent
        if _is_zero(affine):
            lo = affine
        elif is_partition and partition_tiles == 1:
            lo = affine
        elif is_partition:
            lo = Mul(left=affine, right=Const(value=partition_tiles))
        else:
            lo = Mul(left=affine, right=width)
        new_ranges.append((lo, width))
    return BufferRegion(tensor=region.tensor, ranges=tuple(new_ranges))


def _axis_capacity(buf: Buffer | None, axis_index: int, location: str, width: int) -> int:
    """Return the number of addressable tiles on one buffer axis."""
    if buf is None or location == "shared_hbm" or axis_index >= len(buf.shape):
        return 1 << 30
    extent = buf.shape[axis_index]
    return extent // width if width > 0 else 0


def _is_zero(expr: Expr) -> bool:
    """True when ``expr`` is the constant 0 (a loopless axis offsets at 0)."""
    return isinstance(expr, Const) and expr.value == 0


def _tensor_to_axes(tree: KernelTree, block_nid: int) -> dict[str, tuple[str, ...]]:
    """Map each operand tensor to its operation's abstract axes."""
    out: dict[str, tuple[str, ...]] = {}
    for nid in _block_local_descendants(tree, block_nid):
        data = tree.data(nid)
        if not isinstance(data, ISANode):
            continue
        for slot, region in data.operand_bindings.items():
            out[region.tensor] = data.op_cls.OPERAND_AXES[slot]
    return out


def _loopvar_to_dim(tree: KernelTree, block_nid: int, block: BlockNode) -> dict[str, str]:
    """Map each local loop variable to the concrete dimension it binds."""
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


__all__ = ["normalize_block", "normalize_selected_tensor_regions"]
