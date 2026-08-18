"""``Reorder`` transform — swap an adjacent parent-child ForNode pair via payload swap."""

from __future__ import annotations

from dataclasses import dataclass

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Expr, Var, to_affine
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, KernelTree, role_of
from nkigym.ops.base import AxisRole
from nkigym.search.program_sharding import configured_program_shards
from nkigym.transforms.base import (
    Transform,
    TransformLegalityError,
    TransformOption,
    copy_for_rewrite,
    intersects_software_pipeline,
    software_pipeline_overlap_nodes,
)
from nkigym.transforms.helper.access_pattern import subtree_has_access_patterns
from nkigym.transforms.helper.normalize import _substitute_block_regions


@dataclass(frozen=True)
class ReorderOption(TransformOption):
    """Swap the payloads of two adjacent parent-child ForNodes."""

    outer_nid: int
    inner_nid: int


class Reorder(Transform[ReorderOption]):
    """Swap an adjacent parent-child ForNode pair via payload swap."""

    def analyze(self, ir: KernelIR) -> list[ReorderOption]:
        options: list[ReorderOption] = []
        overlap_nodes = software_pipeline_overlap_nodes(ir)
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
            if self._is_legal(ir, opt, overlap_nodes):
                options.append(opt)
        return options

    def apply(self, ir: KernelIR, option: ReorderOption) -> KernelIR:
        self._check_legality(ir, option)
        same_axis = _axis_of_loop(ir.tree, option.outer_nid) == _axis_of_loop(ir.tree, option.inner_nid)
        new_ir = copy_for_rewrite(ir)
        outer_data = new_ir.tree.data(option.outer_nid)
        inner_data = new_ir.tree.data(option.inner_nid)
        new_ir.tree.graph.nodes[option.outer_nid]["data"] = inner_data
        new_ir.tree.graph.nodes[option.inner_nid]["data"] = outer_data
        self._renormalize_same_axis_swap(new_ir, option, same_axis)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _renormalize_same_axis_swap(self, ir: KernelIR, option: ReorderOption, same_axis: bool) -> None:
        """Preserve existing regions when interchanging two loops of one axis.

        A pure payload swap leaves the loop names in physical order that no longer
        matches their dense ordinal (the physically-outer loop may now be named
        ``i_d0_1`` while ``i_d0_0`` sits inside it). The enclosing block keeps its
        pre-swap regions, so swapping the two loop variables in those regions
        preserves the same logical tiles under the new traversal order. Full
        normalization is intentionally excluded because localized buffers may
        omit unrelated outer-loop coordinates.
        """
        if not same_axis:
            return
        outer = ir.tree.loop(option.outer_nid)
        inner = ir.tree.loop(option.inner_nid)
        substitutions: dict[str, Expr] = {
            outer.loop_var: Var(name=inner.loop_var),
            inner.loop_var: Var(name=outer.loop_var),
        }
        ir.tree.graph.nodes[option.outer_nid]["data"] = ForNode(loop_var=inner.loop_var, extent=outer.extent)
        ir.tree.graph.nodes[option.inner_nid]["data"] = ForNode(loop_var=outer.loop_var, extent=inner.extent)
        block_nid = _enclosing_block_nid(ir.tree, option.outer_nid)
        affected_blocks = (block_nid, *ir.tree.blocks(option.outer_nid))
        for nid in affected_blocks:
            _substitute_block_regions(ir.tree, nid, substitutions)

    def _is_legal(self, ir: KernelIR, option: ReorderOption, overlap_nodes: frozenset[int] | None = None) -> bool:
        try:
            self._check_legality(ir, option, overlap_nodes)
        except TransformLegalityError:
            return False
        return True

    def _check_legality(self, ir: KernelIR, option: ReorderOption, overlap_nodes: frozenset[int] | None = None) -> None:
        for nid in (option.outer_nid, option.inner_nid):
            if nid not in ir.tree.graph:
                raise TransformLegalityError(f"Reorder: nid {nid} not in tree")
        if intersects_software_pipeline(ir, (option.outer_nid, option.inner_nid), overlap_nodes):
            raise TransformLegalityError("Reorder cannot alter an active software-pipeline scope")
        shards = configured_program_shards(ir)
        if option.outer_nid in shards or option.inner_nid in shards:
            raise TransformLegalityError("Reorder cannot alter a program-sharded loop")
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
        affected_blocks = {_enclosing_block_nid(ir.tree, option.outer_nid), *ir.tree.blocks(option.inner_nid)}
        for block_nid in affected_blocks:
            block = ir.tree.block(block_nid)
            loop_vars = (outer_loop_var, inner_loop_var)
            axes_by_loop = tuple(_axes_for_loop_var(block, loop_var) for loop_var in loop_vars)
            for loop_var, axes in zip(loop_vars, axes_by_loop):
                if any(role_of(block, axis) == AxisRole.SEQUENTIAL for axis in axes):
                    raise TransformLegalityError(
                        f"Reorder rejected: affected block has SEQUENTIAL role on loop_var {loop_var!r}"
                    )
            if all(any(role_of(block, axis) == AxisRole.ACCUMULATION for axis in axes) for axes in axes_by_loop):
                raise TransformLegalityError("Reorder cannot change floating-point accumulation order")
        _check_internal_dependency_accesses(ir, option, outer_loop_var, inner_loop_var)


def _axes_for_loop_var(block: BlockNode, loop_var: str) -> tuple[str, ...]:
    """Return every block axis whose affine binding uses ``loop_var``."""
    return tuple(iv.axis for iv, value in zip(block.iter_vars, block.iter_values) if loop_var in to_affine(value))


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
        loop_vars = frozenset((outer_loop_var, inner_loop_var))
        invariant = not _regions_depend_on(producer_regions, loop_vars) and not _regions_depend_on(
            consumer_regions, loop_vars
        )
        if not invariant and _region_signatures(producer_regions) != _region_signatures(consumer_regions):
            raise TransformLegalityError(
                f"Reorder rejected: dependency {producer}->{consumer} on tensor {tensor!r} "
                "is carried by a swapped loop"
            )


def _regions_depend_on(regions: tuple[BufferRegion, ...], loop_vars: frozenset[str]) -> bool:
    """Return whether any region bound depends on a swapped loop."""
    return any(
        bool(loop_vars & {name for expression in (lower, width) for name in to_affine(expression) if name is not None})
        for region in regions
        for lower, width in region.ranges
    )


def _region_signatures(regions: tuple[BufferRegion, ...]) -> tuple[str, ...]:
    """Return an order-independent exact affine signature for regions."""
    signatures = []
    for region in regions:
        ranges = tuple(
            (
                tuple(sorted(to_affine(lower).items(), key=lambda item: str(item[0]))),
                tuple(sorted(to_affine(width).items(), key=lambda item: str(item[0]))),
            )
            for lower, width in region.ranges
        )
        signatures.append(repr((region.tensor, ranges)))
    return tuple(sorted(signatures))


def _axis_of_loop(tree: KernelTree, loop_nid: int) -> str | None:
    """Return the unique concrete block axis driven by one loop."""
    loop_var = tree.loop(loop_nid).loop_var
    block_nids = {_enclosing_block_nid(tree, loop_nid), *tree.blocks(loop_nid)}
    axes = {
        iter_var.axis
        for block_nid in block_nids
        for iter_var, value in zip(tree.block(block_nid).iter_vars, tree.block(block_nid).iter_values)
        if loop_var in to_affine(value)
    }
    return next(iter(axes)) if len(axes) == 1 else None


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


__all__ = ["Reorder", "ReorderOption"]
