"""``Reorder`` transform — swap an adjacent parent-child ForNode pair via payload swap."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Var, to_affine
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, KernelTree, role_of
from nkigym.ops.base import AxisRole
from nkigym.transforms._access_pattern import subtree_has_access_patterns
from nkigym.transforms._normalize import normalize_block
from nkigym.transforms._tree_ops import invalidate_stale_software_pipelines
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class ReorderOption(TransformOption):
    """Swap the payloads of two adjacent parent-child ForNodes."""

    outer_nid: int
    inner_nid: int


class Reorder(Transform[ReorderOption]):
    """Swap an adjacent parent-child ForNode pair via payload swap."""

    def analyze(self, ir: KernelIR) -> list[ReorderOption]:
        options: list[ReorderOption] = []
        for nid in ir.tree.preorder():
            data = ir.tree.data(nid)
            if not isinstance(data, ForNode):
                continue
            kids = ir.tree.children(nid)
            if len(kids) != 1:
                continue
            kid_data = ir.tree.data(kids[0])
            if not isinstance(kid_data, ForNode):
                continue
            opt = ReorderOption(outer_nid=nid, inner_nid=kids[0])
            if self._is_legal(ir, opt):
                options.append(opt)
        return options

    def apply(self, ir: KernelIR, option: ReorderOption) -> KernelIR:
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        outer_data = new_ir.tree.data(option.outer_nid)
        inner_data = new_ir.tree.data(option.inner_nid)
        new_ir.tree.graph.nodes[option.outer_nid]["data"] = inner_data
        new_ir.tree.graph.nodes[option.inner_nid]["data"] = outer_data
        self._renormalize_same_dim_swap(new_ir, option)
        invalidate_stale_software_pipelines(new_ir)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _renormalize_same_dim_swap(self, ir: KernelIR, option: ReorderOption) -> None:
        """Restore the dense-name / stride invariant when the swap interchanges two
        loops of the SAME dim.

        A pure payload swap leaves the loop names in physical order that no longer
        matches their dense ordinal (the physically-outer loop may now be named
        ``i_d0_1`` while ``i_d0_0`` sits inside it). The enclosing block keeps its
        pre-swap tile-linearization binding, so a LATER transform that
        renormalizes a co-located block (e.g. CodeMotion sinking a load) recomputes
        that block's stride from physical order and disagrees with this block on
        which tile is which -> wrong offset / OOB. Re-normalizing the swapped
        loops' enclosing block (and its nested sub-blocks) here re-derives names +
        bindings from physical order immediately, so every block shares one
        convention. A cross-dim swap (each dim has a single loop) leaves
        name-order == physical-order, so this is a no-op — the byte-exact ladder
        (whose only Reorders are cross-dim) is unaffected.
        """
        if _dim_of(ir.tree, option.outer_nid) != _dim_of(ir.tree, option.inner_nid):
            return
        block_nid = _enclosing_block_nid(ir.tree, option.outer_nid)
        for nid in (block_nid, *_nested_block_nids(ir.tree, block_nid)):
            normalize_block(ir.tree, nid)

    def _is_legal(self, ir: KernelIR, option: ReorderOption) -> bool:
        try:
            self._check_legality(ir, option)
        except TransformLegalityError:
            return False
        return True

    def _check_legality(self, ir: KernelIR, option: ReorderOption) -> None:
        for nid in (option.outer_nid, option.inner_nid):
            if nid not in ir.tree.graph:
                raise TransformLegalityError(f"Reorder: nid {nid} not in tree")
        outer = ir.tree.data(option.outer_nid)
        inner = ir.tree.data(option.inner_nid)
        if not isinstance(outer, ForNode) or not isinstance(inner, ForNode):
            raise TransformLegalityError(
                f"Reorder: both targets must be ForNode; got {type(outer).__name__}, {type(inner).__name__}"
            )
        kids = ir.tree.children(option.outer_nid)
        if kids != [option.inner_nid]:
            raise TransformLegalityError(f"Reorder: inner must be sole child of outer; got children {kids}")
        if subtree_has_access_patterns(ir.tree, option.outer_nid):
            raise TransformLegalityError("Reorder cannot rewrite loops containing an explicit access pattern")
        outer_loop_var = outer.loop_var
        inner_loop_var = inner.loop_var
        for descendant in ir.tree.blocks(option.inner_nid):
            block = ir.tree.data(descendant)
            assert isinstance(block, BlockNode)
            for loop_var in (outer_loop_var, inner_loop_var):
                axis = _axis_for_loop_var(block, loop_var)
                if axis is None:
                    continue
                if role_of(block, axis) == AxisRole.SEQUENTIAL:
                    raise TransformLegalityError(
                        f"Reorder rejected: descendant block has SEQUENTIAL role on loop_var {loop_var!r}"
                    )
        _check_internal_dependency_accesses(ir, option, outer_loop_var, inner_loop_var)


def _axis_for_loop_var(block: BlockNode, loop_var: str) -> str | None:
    """Return the iter_var axis bound by ``loop_var``, if any."""
    for iv, value in zip(block.iter_vars, block.iter_values):
        if isinstance(value, Var) and value.name == loop_var:
            return iv.axis
    return None


def _check_internal_dependency_accesses(
    ir: KernelIR, option: ReorderOption, outer_loop_var: str, inner_loop_var: str
) -> None:
    """Reject interchange that changes which iteration satisfies an internal dependency.

    For a dependency whose endpoints are both inside the inner loop, each
    endpoint must agree on whether the dependent tensor is indexed by each
    swapped loop. A mismatch means the value is carried between iterations of
    that loop. Interchanging the loops can then place another outer-loop
    iteration between the producer and consumer.
    """
    subtree = ir.tree.descendants(option.inner_nid)
    for producer, consumer, attrs in ir.dependency.graph.edges(data=True):
        if producer not in subtree or consumer not in subtree:
            continue
        tensor = attrs.get("tensor")
        kind = attrs.get("kind")
        if tensor is None or kind not in {"RAW", "WAR", "WAW"}:
            continue
        producer_side = "write" if kind in {"RAW", "WAW"} else "read"
        consumer_side = "read" if kind == "RAW" else "write"
        producer_regions = ir.dependency._regions_for(producer, tensor, producer_side)
        consumer_regions = ir.dependency._regions_for(consumer, tensor, consumer_side)
        for loop_var in (outer_loop_var, inner_loop_var):
            producer_invariant = _regions_invariant(producer_regions, loop_var)
            consumer_invariant = _regions_invariant(consumer_regions, loop_var)
            if producer_invariant != consumer_invariant:
                raise TransformLegalityError(
                    f"Reorder rejected: dependency {producer}->{consumer} on tensor {tensor!r} "
                    f"has different dependence on loop_var {loop_var!r} at its endpoints"
                )


def _regions_invariant(regions: tuple[BufferRegion, ...], loop_var: str) -> bool:
    """Return whether every region offset is invariant in ``loop_var``."""
    return bool(regions) and not any(
        loop_var in to_affine(lower) for region in regions for lower, _width in region.ranges
    )


def _dim_of(tree: KernelTree, loop_nid: int) -> str:
    """Concrete dim a ForNode drives, parsed from its dense name ``i_d{dim}_{N}`` -> ``d{dim}``."""
    loop_var = tree.loop(loop_nid).loop_var
    body = loop_var[2:] if loop_var.startswith("i_") else loop_var
    return body.split("_")[0]


def _enclosing_block_nid(tree: KernelTree, nid: int) -> int:
    """Nearest BlockNode ancestor of ``nid``."""
    result: int | None = None
    for anc in reversed(tree.ancestors(nid)):
        if isinstance(tree.data(anc), BlockNode):
            result = anc
            break
    if result is None:
        raise ValueError(f"no enclosing BlockNode for {nid}")
    return result


def _nested_block_nids(tree: KernelTree, block_nid: int) -> list[int]:
    """BlockNode descendants of ``block_nid`` (co-located sub-blocks), excluding itself."""
    return [d for d in tree.descendants(block_nid) if d != block_nid and isinstance(tree.data(d), BlockNode)]


__all__ = ["Reorder", "ReorderOption"]
