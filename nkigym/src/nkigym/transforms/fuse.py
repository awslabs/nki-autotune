"""``Fuse`` transform — collapse adjacent same-axis ForNodes (or absorb them into a tensorize tile)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import prod

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const, Expr, Var, substitute, to_affine
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, ForNode, ISANode, KernelTree
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption
from nkigym.transforms.helper.access_pattern import subtree_has_access_patterns
from nkigym.transforms.helper.normalize import _dim_from_loopvar, _substitute_block_regions, normalize_block
from nkigym.transforms.helper.tile_region import retile_region
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children, invalidate_stale_software_pipelines
from nkigym.transforms.split import _current_tensorize_width


@dataclass(frozen=True)
class FuseOption(TransformOption):
    """Per-application payload for :class:`Fuse`.

    Attributes:
        target_nids: Exactly two adjacent axis-chain entries to fuse, in
            parent-to-child order.
        target_axis: ``None`` for outer-trip flavour. The concrete iter_var
            axis name (e.g. ``"d1"``) for tensorize flavour; matches
            ``IterVar.axis``.
    """

    target_nids: tuple[int, ...]
    target_axis: str | None = None


class Fuse(Transform[FuseOption]):
    """Collapse two adjacent same-loop-axis entries into one."""

    def analyze(self, ir: KernelIR) -> list[FuseOption]:
        """Return every legal adjacent loop-loop and loop-tensorize pair."""
        options: list[FuseOption] = []
        for nid in ir.tree.preorder():
            data = ir.tree.data(nid)
            if not isinstance(data, ForNode):
                continue
            kids = ir.tree.children(nid)
            if len(kids) != 1:
                continue
            kid_data = ir.tree.data(kids[0])
            if isinstance(kid_data, ForNode):
                if _dim_from_loopvar(data.loop_var) == _dim_from_loopvar(kid_data.loop_var):
                    option = FuseOption(target_nids=(nid, kids[0]), target_axis=None)
                    if self._is_legal(ir, option):
                        options.append(option)
            elif isinstance(kid_data, ISANode):
                _block_nid, block = _find_enclosing_block(ir.tree, kids[0])
                target_axes = [
                    iter_var.axis
                    for iter_var, value in zip(block.iter_vars, block.iter_values)
                    if data.loop_var in to_affine(value)
                ]
                for target_axis in target_axes:
                    option = FuseOption(target_nids=(nid, kids[0]), target_axis=target_axis)
                    if self._is_legal(ir, option):
                        options.append(option)
        return options

    def apply(self, ir: KernelIR, option: FuseOption) -> KernelIR:
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        if option.target_axis is None:
            self._do_outer_trip(new_ir, option)
        else:
            self._do_tensorize(new_ir, option)
        invalidate_stale_software_pipelines(new_ir)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _check_legality(self, ir: KernelIR, option: FuseOption) -> None:
        if len(option.target_nids) != 2:
            raise TransformLegalityError(
                f"Fuse.target_nids must contain exactly two adjacent entries; got {option.target_nids}"
            )
        for nid in option.target_nids:
            if nid not in ir.tree.graph:
                raise TransformLegalityError(f"Fuse.target_nids contains unknown nid {nid}")
        if subtree_has_access_patterns(ir.tree, option.target_nids[0]):
            raise TransformLegalityError("Fuse cannot rewrite a loop or ISA operand with an explicit access pattern")
        nodes = [ir.tree.data(nid) for nid in option.target_nids]
        if option.target_axis is None:
            if not all(isinstance(n, ForNode) for n in nodes):
                raise TransformLegalityError(
                    f"Fuse outer-trip flavour: every target must be ForNode; got {[type(n).__name__ for n in nodes]}"
                )
            for parent_nid, child_nid in zip(option.target_nids, option.target_nids[1:]):
                kids = ir.tree.children(parent_nid)
                if kids != [child_nid]:
                    raise TransformLegalityError(
                        f"Fuse outer-trip flavour: nid {parent_nid} must have a single child {child_nid}; got {kids}"
                    )
            _check_no_partial_loop_dependence(ir.tree, option.target_nids)
        else:
            """Tensorize flavour: prefix is ForNodes; last is the ISA leaf."""
            leaf = nodes[-1]
            if not isinstance(leaf, ISANode):
                raise TransformLegalityError(
                    f"Fuse tensorize flavour: last target must be ISANode; got {type(leaf).__name__}"
                )
            for n in nodes[:-1]:
                if not isinstance(n, ForNode):
                    raise TransformLegalityError(
                        f"Fuse tensorize flavour: prefix must be all ForNodes; got {type(n).__name__}"
                    )
            for parent_nid, child_nid in zip(option.target_nids, option.target_nids[1:]):
                kids = ir.tree.children(parent_nid)
                if kids != [child_nid]:
                    raise TransformLegalityError(
                        f"Fuse tensorize flavour: nid {parent_nid} must have a single child {child_nid}; got {kids}"
                    )
            _block_nid, block = _find_enclosing_block(ir.tree, option.target_nids[-1])
            target_values = [
                value
                for iter_var, value in zip(block.iter_vars, block.iter_values)
                if iter_var.axis == option.target_axis
            ]
            loop_var = ir.tree.loop(option.target_nids[0]).loop_var
            if len(target_values) != 1 or loop_var not in to_affine(target_values[0]):
                raise TransformLegalityError(
                    f"Fuse.target_axis={option.target_axis!r} is not bound by loop {option.target_nids[0]}"
                )
            self._check_tensorize_loop_uses(leaf, block, loop_var, option.target_axis)
            current_width = _current_tensorize_width(leaf, block, option.target_axis)
            if current_width is None:
                raise TransformLegalityError(
                    f"Fuse.target_axis={option.target_axis!r}: no tensorize width on this leaf"
                )
            inverse_axis_map = {concrete: abstract for abstract, concrete in block.axis_map.items()}
            abstract_axis = inverse_axis_map.get(option.target_axis)
            max_tile = _maximum_tensorize_width(ir, leaf, abstract_axis)
            absorbed_extent = prod(ir.tree.loop(nid).extent for nid in option.target_nids[:-1])
            fused_width = current_width * absorbed_extent
            if max_tile is not None and fused_width > max_tile:
                raise TransformLegalityError(
                    f"Fuse.target_axis={option.target_axis!r}: fused tile {fused_width} > MAX_TILE_SIZE {max_tile}"
                )

    def _check_tensorize_loop_uses(self, leaf: ISANode, block: BlockNode, loop_var: str, target_axis: str) -> None:
        """Reject loop uses that widening the selected operand axis cannot absorb."""
        abstract_axis = next(
            (abstract for abstract, concrete in block.axis_map.items() if concrete == target_axis), None
        )
        invalid_uses: list[str] = []
        for iter_var, value in zip(block.iter_vars, block.iter_values):
            if iter_var.axis != target_axis and loop_var in to_affine(value):
                invalid_uses.append(f"iter_var {iter_var.axis}")
        for slot, region in leaf.operand_bindings.items():
            axes = leaf.op_cls.OPERAND_AXES[slot]
            target_index = axes.index(abstract_axis) if abstract_axis in axes else None
            for index, (lower, width) in enumerate(region.ranges):
                if (loop_var in to_affine(lower) or loop_var in to_affine(width)) and index != target_index:
                    invalid_uses.append(f"operand {slot}[{index}]")
        if invalid_uses:
            uses = ", ".join(invalid_uses)
            raise TransformLegalityError(f"Fuse loop {loop_var!r} has uses outside target_axis={target_axis!r}: {uses}")

    def _is_legal(self, ir: KernelIR, option: FuseOption) -> bool:
        """Return whether ``option`` passes the same checks used by :meth:`apply`."""
        try:
            self._check_legality(ir, option)
        except TransformLegalityError:
            return False
        return True

    def _do_outer_trip(self, ir: KernelIR, option: FuseOption) -> None:
        """Outer-trip Fuse: merge two parent-child same-dim ForNodes into one loop.

        Only the loop topology changes: the chain is replaced by a single
        ForNode whose extent is the product of the chain extents (the access
        tile width is unchanged). :func:`normalize_block` then assigns the
        dense name and rebuilds the iter_values + region offsets from the new
        loop structure.
        """
        nids = option.target_nids
        first = ir.tree.data(nids[0])
        assert isinstance(first, ForNode)
        parent_nid = ir.tree.parent(nids[0])
        assert parent_nid is not None
        deepest_kids = ir.tree.children(nids[-1])
        new_extent = prod(ir.tree.loop(nid).extent for nid in nids)
        block_nid, _block = _find_enclosing_block(ir.tree, nids[0])
        old_loop_vars = tuple(ir.tree.loop(nid).loop_var for nid in nids)

        new_nid = ir.tree.add_node(ForNode(loop_var=f"{first.loop_var}__fused", extent=new_extent), parent=None)
        for child_nid in deepest_kids:
            ir.tree.graph.add_edge(new_nid, child_nid)
        _replace_in_parent_children(ir.tree, parent_nid, [nids[0]], [new_nid])
        for nid in nids:
            ir.tree.graph.remove_node(nid)

        nested_blocks = {nid for nid in ir.tree.preorder(new_nid) if isinstance(ir.tree.data(nid), BlockNode)}
        substitutions: dict[str, Expr] = {loop_var: Const(value=0) for loop_var in old_loop_vars}
        substitutions[old_loop_vars[-1]] = Var(name=ir.tree.loop(new_nid).loop_var)
        _normalize_block_hierarchy(ir.tree, block_nid, substitutions, nested_blocks)

    def _do_tensorize(self, ir: KernelIR, option: FuseOption) -> None:
        """Tensorize Fuse: absorb one same-axis ForNode into an ISA tile.

        ``option.target_nids[-1]`` is the ISA leaf and the first entry is the
        ForNode to absorb. The loop is removed and the affected-axis access
        width grows by the product of the absorbed extents;
        :func:`normalize_block` then drops any now-trip-1 loops, re-densifies
        names, and recomputes the region offsets from the surviving loops.
        """
        leaf_nid = option.target_nids[-1]
        leaf = ir.tree.data(leaf_nid)
        assert isinstance(leaf, ISANode)
        for_chain = option.target_nids[:-1]
        chain_root = for_chain[0]
        chain_root_parent = ir.tree.parent(chain_root)
        assert chain_root_parent is not None
        block_nid, block = _find_enclosing_block(ir.tree, leaf_nid)

        absorbed_extent = prod(ir.tree.loop(nid).extent for nid in for_chain)
        ir.tree.graph.remove_edge(for_chain[-1], leaf_nid)
        _replace_in_parent_children(ir.tree, chain_root_parent, [chain_root], [leaf_nid])
        for nid in for_chain:
            ir.tree.graph.remove_node(nid)

        inverse_axis_map = {concrete: abstract for abstract, concrete in block.axis_map.items()}
        assert option.target_axis is not None
        abstract_axis = inverse_axis_map.get(option.target_axis)

        def _widen(lo: Expr, width: int) -> tuple[Expr, int]:
            """Keep the offset (normalize recomputes it); grow the tile width."""
            return lo, width * absorbed_extent

        new_bindings = {
            slot: retile_region(region, leaf.op_cls.OPERAND_AXES[slot], abstract_axis, _widen)
            for slot, region in leaf.operand_bindings.items()
        }
        ir.tree.graph.nodes[leaf_nid]["data"] = ISANode(
            op_cls=leaf.op_cls, operand_bindings=new_bindings, kwargs=dict(leaf.kwargs)
        )

        """Block reads/writes are keyed by tensor name, not slot; map tensor->axes via the leaf
        so each region uses its own operand's axes (matmul stationary lacks N -> no-op)."""
        tensor_to_axes = {leaf.operand_bindings[s].tensor: leaf.op_cls.OPERAND_AXES[s] for s in leaf.operand_bindings}
        new_block = BlockNode(
            iter_vars=block.iter_vars,
            iter_values=block.iter_values,
            reads=tuple(retile_region(r, tensor_to_axes.get(r.tensor, ()), abstract_axis, _widen) for r in block.reads),
            writes=tuple(
                retile_region(w, tensor_to_axes.get(w.tensor, ()), abstract_axis, _widen) for w in block.writes
            ),
            alloc_buffers=block.alloc_buffers,
            annotations=dict(block.annotations),
            axis_map=block.axis_map,
        )
        ir.tree.graph.nodes[block_nid]["data"] = new_block

        normalize_block(ir.tree, block_nid)


def _maximum_tensorize_width(ir: KernelIR, leaf: ISANode, abstract_axis: str | None) -> int | None:
    """Return the hardware tile limit for one tensorized operation axis."""
    maximum = leaf.op_cls.MAX_TILE_SIZE.get(abstract_axis) if abstract_axis is not None else None
    if leaf.op_cls is NKIDMATranspose and abstract_axis is not None:
        source = leaf.operand_bindings["src"].tensor
        if ir.buffer(source).location == "shared_hbm":
            maximum = NKIDMATranspose.HBM_SOURCE_MAX_TILE_SIZE[abstract_axis]
    return maximum


def _find_enclosing_block(tree: KernelTree, nid: int) -> tuple[int, BlockNode]:
    """Walk ancestors of ``nid`` until we hit a BlockNode."""
    for ancestor in reversed(tree.ancestors(nid)):
        data = tree.data(ancestor)
        if isinstance(data, BlockNode):
            return ancestor, data
    raise TransformLegalityError(f"no enclosing BlockNode for nid {nid}")


def _normalize_block_hierarchy(
    tree: KernelTree, block_nid: int, substitutions: dict[str, Expr], nested_scope: set[int]
) -> None:
    """Normalize one block and propagate its loop rewrites into nested blocks."""
    nested_paths = {
        nested_block: _loop_path_names(tree, block_nid, nested_block)
        for nested_block in _immediate_nested_blocks(tree, block_nid)
        if nested_block in nested_scope
    }
    if substitutions:
        _substitute_block_regions(tree, block_nid, substitutions)
    normalize_block(tree, block_nid)

    for nested_block, path_names in nested_paths.items():
        local_substitutions: dict[str, Expr] = {}
        for nid, old_name in path_names.items():
            replacement: Expr
            if nid in tree.graph:
                replacement = Var(name=tree.loop(nid).loop_var)
            else:
                replacement = Const(value=0)
            if replacement != Var(name=old_name):
                local_substitutions[old_name] = replacement
        child_substitutions = {
            old_name: substitute(replacement, local_substitutions) for old_name, replacement in substitutions.items()
        }
        child_substitutions.update(local_substitutions)
        _normalize_block_hierarchy(tree, nested_block, child_substitutions, nested_scope)


def _immediate_nested_blocks(tree: KernelTree, block_nid: int) -> list[int]:
    """Return nested blocks whose nearest enclosing block is ``block_nid``."""
    nested_blocks: list[int] = []
    stack = [block_nid]
    while stack:
        for child_nid in tree.children(stack.pop()):
            if isinstance(tree.data(child_nid), BlockNode):
                nested_blocks.append(child_nid)
            else:
                stack.append(child_nid)
    return nested_blocks


def _loop_path_names(tree: KernelTree, block_nid: int, nested_block: int) -> dict[int, str]:
    """Return loop names on the path from ``block_nid`` to ``nested_block``."""
    names: dict[int, str] = {}
    current = tree.parent(nested_block)
    while current is not None and current != block_nid:
        node = tree.data(current)
        if isinstance(node, ForNode):
            names[current] = node.loop_var
        current = tree.parent(current)
    if current != block_nid:
        raise AssertionError(f"block {nested_block} is not nested under block {block_nid}")
    return names


def _check_no_partial_loop_dependence(tree: KernelTree, target_nids: tuple[int, ...]) -> None:
    """Reject accesses that cannot remain affine after the target loops are fused.

    A descendant expression may be invariant across every target loop or use
    their mixed-radix coefficients as one contiguous index. Any other
    coefficient pattern requires floor division or modulo to recover an
    original loop position. ``normalize_block`` cannot represent that mapping
    and would otherwise silently change the access offset.
    """
    target_loops = [(tree.loop(nid).loop_var, tree.loop(nid).extent) for nid in target_nids]
    target_strides: dict[str, int] = {}
    stride = 1
    for loop_var, extent in reversed(target_loops):
        target_strides[loop_var] = stride
        stride *= extent
    innermost_var = target_loops[-1][0]
    for nid in tree.preorder(target_nids[0]):
        for expression in _binding_and_access_offsets(tree.data(nid)):
            coefficients = to_affine(expression)
            scale = coefficients.get(innermost_var, 0)
            actual = tuple(coefficients.get(loop_var, 0) for loop_var, _extent in target_loops)
            expected = tuple(scale * target_strides[loop_var] for loop_var, _extent in target_loops)
            if actual != expected:
                raise TransformLegalityError(
                    "Fuse outer-trip flavour: descendant "
                    f"nid {nid} uses non-contiguous coefficients {actual} for fused loops "
                    f"{tuple(loop_var for loop_var, _extent in target_loops)}; expected coefficients "
                    f"proportional to {tuple(target_strides[loop_var] for loop_var, _extent in target_loops)}"
                )


def _binding_and_access_offsets(node: BlockNode | ForNode | ISANode) -> list[Expr]:
    """Return block bindings and region lower bounds carried by ``node``."""
    expressions: list[Expr] = []
    if isinstance(node, BlockNode):
        expressions.extend(node.iter_values)
        regions = (*node.reads, *node.writes)
    elif isinstance(node, ISANode):
        regions = tuple(node.operand_bindings.values())
    else:
        regions = ()
    expressions.extend(lower for region in regions for lower, _width in region.ranges)
    return expressions


__all__ = ["Fuse", "FuseOption"]
