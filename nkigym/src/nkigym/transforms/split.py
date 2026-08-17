"""``Split`` transform — partition one loop or tensorized tile into two factors."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from nkigym.ir import KernelIR
from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import Add, Const, Expr, Mul, Var
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, ForNode, ISANode, KernelTree
from nkigym.ops.base import ReductionContract
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption
from nkigym.transforms.helper.access_pattern import subtree_has_access_patterns
from nkigym.transforms.helper.normalize import _substitute_block_regions, normalize_block
from nkigym.transforms.helper.tile_region import retile_region
from nkigym.transforms.helper.tree_ops import (
    _block_local_descendants,
    _replace_in_parent_children,
    invalidate_stale_software_pipelines,
)


@dataclass(frozen=True)
class SplitOption(TransformOption):
    """Per-application payload for :class:`Split`.

    Attributes:
        target_nid: Node id in ``ir.tree`` to split. Either a
            :class:`ForNode` (outer-trip flavour) or an :class:`ISANode`
            (tensorize flavour).
        factors: Two replacement factors, outermost-first.
        target_axis: ``None`` for outer-trip flavour. The concrete iter_var
            axis name (e.g. ``"d1"``) for tensorize flavour; matches
            ``IterVar.axis``. Translated to the abstract op-axis via the
            enclosing block's ``axis_map`` for the ``OPERAND_AXES`` lookup.
    """

    target_nid: int
    factors: tuple[int, ...]
    target_axis: str | None = None


class Split(Transform[SplitOption]):
    """Replace one loop or tensorized tile with exactly two factors."""

    def analyze(self, ir: KernelIR) -> list[SplitOption]:
        options: list[SplitOption] = []
        for nid in ir.tree.preorder():
            data = ir.tree.data(nid)
            if isinstance(data, (ForNode, ISANode)) and subtree_has_access_patterns(ir.tree, nid):
                continue
            if isinstance(data, ForNode):
                if _encloses_multiple_blocks(ir.tree, nid):
                    """Outer-trip Split of a shared (post-CodeMotion) loop is illegal — see
                    _reject_if_shared_loop; do not offer it as a candidate action."""
                    continue
                for factors in _factorizations(data.extent):
                    options.append(SplitOption(target_nid=nid, factors=factors, target_axis=None))
            elif isinstance(data, ISANode):
                """Tensorize flavour: walk the enclosing block's iter_vars."""
                block_nid, block = _find_enclosing_block(ir.tree, nid)
                for iv in block.iter_vars:
                    concrete = iv.axis
                    if _is_slot_reduction_axis(ir, nid, concrete):
                        continue
                    """Tile width currently bound on the leaf (max_tile or full extent)."""
                    current = _current_tensorize_width(data, block, concrete)
                    if current is None or current < 2:
                        continue
                    floor = _min_tile_floor(data, block, concrete)
                    for factors in _factorizations(current):
                        if floor is not None and factors[-1] < floor:
                            continue
                        options.append(SplitOption(target_nid=nid, factors=factors, target_axis=concrete))
        return options

    def apply(self, ir: KernelIR, option: SplitOption) -> KernelIR:
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        if option.target_axis is None:
            self._do_outer_trip(new_ir, option)
        else:
            self._do_tensorize(new_ir, option)
        invalidate_stale_software_pipelines(new_ir)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _check_legality(self, ir: KernelIR, option: SplitOption) -> None:
        if len(option.factors) != 2:
            raise TransformLegalityError(f"Split.factors must contain exactly two factors; got {option.factors}")
        if any(f < 2 for f in option.factors):
            raise TransformLegalityError(f"Split.factors entries must be >= 2; got {option.factors}")
        target = _resolve(ir.tree, option.target_nid)
        if subtree_has_access_patterns(ir.tree, option.target_nid):
            raise TransformLegalityError("Split cannot rewrite a loop or ISA operand with an explicit access pattern")
        if option.target_axis is None:
            if not isinstance(target, ForNode):
                raise TransformLegalityError(
                    f"Split outer-trip flavour requires target to be ForNode; got {type(target).__name__}"
                )
            if not _covers_exactly(option.factors, target.extent):
                raise TransformLegalityError(
                    f"Split.factors {option.factors} do not exactly tile ForNode.extent {target.extent}"
                )
            _reject_if_shared_loop(ir.tree, option.target_nid)
        else:
            if not isinstance(target, ISANode):
                raise TransformLegalityError(
                    f"Split tensorize flavour requires target to be ISANode; got {type(target).__name__}"
                )
            if _is_slot_reduction_axis(ir, option.target_nid, option.target_axis):
                raise TransformLegalityError("Split cannot partition a slot reduction; use RFactor")
            block_nid, block = _find_enclosing_block(ir.tree, option.target_nid)
            if not any(iv.axis == option.target_axis for iv in block.iter_vars):
                raise TransformLegalityError(
                    f"Split.target_axis={option.target_axis!r} not declared by enclosing block"
                )
            current = _current_tensorize_width(target, block, option.target_axis)
            if current is None:
                raise TransformLegalityError(
                    f"Split.target_axis={option.target_axis!r}: no tensorize width on this leaf"
                )
            if not _covers_exactly(option.factors, current):
                raise TransformLegalityError(
                    f"Split.factors {option.factors} do not exactly tile tensorize width {current}"
                )
            floor = _min_tile_floor(target, block, option.target_axis)
            if floor is not None and option.factors[-1] < floor:
                raise TransformLegalityError(
                    f"Split.target_axis={option.target_axis!r}: innermost tile {option.factors[-1]} "
                    f"< MIN_TILE_SIZE {floor}"
                )

    def _do_outer_trip(self, ir: KernelIR, option: SplitOption) -> None:
        """Outer-trip Split: replace the target ForNode with a chain of factor ForNodes.

        Only the loop topology changes (the access tile width is unchanged);
        :func:`normalize_block` then assigns dense names and rebuilds the
        iter_values + region offsets from the new loop structure.
        """
        target_nid = option.target_nid
        target = ir.tree.data(target_nid)
        assert isinstance(target, ForNode)
        parent_nid = ir.tree.parent(target_nid)
        assert parent_nid is not None
        original_children = ir.tree.children(target_nid)
        block_nid, _block = _find_enclosing_block(ir.tree, target_nid)

        new_top_nid, new_bottom_nid = _build_for_chain(ir.tree, target.loop_var, option.factors)
        for child_nid in original_children:
            ir.tree.graph.add_edge(new_bottom_nid, child_nid)
        _replace_in_parent_children(ir.tree, parent_nid, [target_nid], [new_top_nid])
        ir.tree.graph.remove_node(target_nid)

        outer = ir.tree.loop(new_top_nid)
        inner = ir.tree.loop(new_bottom_nid)
        composed = Add(
            left=Mul(left=Var(name=outer.loop_var), right=Const(value=inner.extent)), right=Var(name=inner.loop_var)
        )
        _substitute_block_regions(ir.tree, block_nid, {target.loop_var: composed})
        _normalize_split_block(ir.tree, block_nid)

    def _do_tensorize(self, ir: KernelIR, option: SplitOption) -> int:
        """Tensorize Split: insert ``factors[:-1]`` loops above the leaf, set the access width.

        The new loops carry temporary names and the affected-axis access
        width is set to ``factors[-1]``; :func:`normalize_block` then assigns
        dense names and recomputes the region offsets from the loop strides.

        Scope — exact-division-only. ``Split`` (both flavours) splits a factor
        into sub-factors that exactly tile the factor: ``_factorizations`` only
        enumerates exact divisors (``remaining % f == 0``) and
        :meth:`_check_legality` rejects any non-exact cover via
        :func:`_covers_exactly`. Ragged /
        non-divisible splits — where the innermost factor does not divide the
        extent and TVM appends a ``BlockPredicate`` (``floormod`` guard) to mask
        the out-of-range tail (its ``BlockPredicateAppender``) — are out of
        scope here: the IR only ever generates tile-multiple splits (constrained
        by the hardware tile multiples / per-op ``MIN_TILE_SIZE``), so no ragged
        split is reachable and no predicate-elision path exists. The bespoke
        affine work this path would otherwise carry (region-offset recompute)
        lives in :func:`normalize_block` (our equivalent of TVM's
        ``IterMapSimplifyBlockBinding``); this method only does structural loop
        insertion plus a constant width-set via :func:`retile_region`.
        """
        leaf_nid = option.target_nid
        leaf = ir.tree.data(leaf_nid)
        assert isinstance(leaf, ISANode)
        parent_nid = ir.tree.parent(leaf_nid)
        assert parent_nid is not None
        block_nid, block = _find_enclosing_block(ir.tree, leaf_nid)

        base_loop_var = f"i_{option.target_axis}"
        """Build the new loop chain DETACHED, then splice its top into the leaf's
        original child slot. Adding loops directly under ``parent_nid`` would
        append them (``nx.DiGraph.add_edge`` appends to the successor list),
        moving the leaf's subtree to the END of its siblings — when the leaf is
        e.g. a memset that must precede a co-located matmul block, that reorders
        it after the matmul (zeroing the accumulator post-compute). Splicing at
        the original index preserves sibling dataflow order."""
        top_nid, bottom_nid = _build_for_chain(ir.tree, base_loop_var, option.factors[:-1])
        ir.tree.graph.add_edge(bottom_nid, leaf_nid)
        _replace_in_parent_children(ir.tree, parent_nid, [leaf_nid], [top_nid])

        inverse_axis_map = {concrete: abstract for abstract, concrete in block.axis_map.items()}
        assert option.target_axis is not None
        abstract_axis = inverse_axis_map.get(option.target_axis)
        new_width = option.factors[-1]

        def _set_width(lo: Expr, _width: int) -> tuple[Expr, int]:
            """Keep the offset (normalize recomputes it); set the new tile width."""
            return lo, new_width

        new_bindings = {
            slot: retile_region(region, leaf.op_cls.OPERAND_AXES[slot], abstract_axis, _set_width)
            for slot, region in leaf.operand_bindings.items()
        }
        ir.tree.graph.nodes[leaf_nid]["data"] = ISANode(
            op_cls=leaf.op_cls, operand_bindings=new_bindings, kwargs=dict(leaf.kwargs)
        )

        """Block reads/writes are keyed by tensor name, not slot; map tensor->axes via the leaf
        so each region uses its own operand's axes. A region whose tensor is not an operand
        gets () axes -> no-op."""
        tensor_to_axes = {leaf.operand_bindings[s].tensor: leaf.op_cls.OPERAND_AXES[s] for s in leaf.operand_bindings}
        new_block = BlockNode(
            iter_vars=block.iter_vars,
            iter_values=block.iter_values,
            reads=tuple(
                retile_region(r, tensor_to_axes.get(r.tensor, ()), abstract_axis, _set_width) for r in block.reads
            ),
            writes=tuple(
                retile_region(w, tensor_to_axes.get(w.tensor, ()), abstract_axis, _set_width) for w in block.writes
            ),
            alloc_buffers=block.alloc_buffers,
            annotations=dict(block.annotations),
            axis_map=block.axis_map,
        )
        ir.tree.graph.nodes[block_nid]["data"] = new_block

        _normalize_split_block(ir.tree, block_nid)
        return top_nid


def _resolve(tree: KernelTree, nid: int):
    if nid not in tree.graph:
        raise TransformLegalityError(f"Split.target_nid={nid} is not a node in the IR tree")
    return tree.data(nid)


def _normalize_split_block(tree: KernelTree, block_nid: int) -> None:
    """Normalize a split block and rebind renamed loops in nested blocks."""
    loop_names = {
        nid: node.loop_var
        for nid in _block_local_descendants(tree, block_nid)
        if isinstance((node := tree.data(nid)), ForNode)
    }
    nested_blocks = [nid for nid in tree.descendants(block_nid) if isinstance(tree.data(nid), BlockNode)]
    normalize_block(tree, block_nid)
    for nested_block in nested_blocks:
        ancestors = set(tree.ancestors(nested_block))
        substitutions: dict[str, Expr] = {}
        for loop_nid, old_name in loop_names.items():
            if loop_nid in ancestors and loop_nid in tree.graph:
                new_name = tree.loop(loop_nid).loop_var
                if new_name != old_name:
                    substitutions[old_name] = Var(name=new_name)
        if substitutions:
            _substitute_block_regions(tree, nested_block, substitutions)


def _enclosing_block_of(tree: KernelTree, nid: int) -> int | None:
    """Nearest BlockNode ancestor of ``nid`` (None if none — e.g. above root)."""
    return next((a for a in reversed(tree.ancestors(nid)) if isinstance(tree.data(a), BlockNode)), None)


def _blocks_under_loop(tree: KernelTree, loop_nid: int) -> set[int]:
    """The distinct enclosing BlockNodes of every ISA leaf beneath ``loop_nid``."""
    out: set[int] = set()
    for d in tree.descendants(loop_nid):
        if isinstance(tree.data(d), ISANode):
            owner = _enclosing_block_of(tree, d)
            if owner is not None:
                out.add(owner)
    return out


def _encloses_multiple_blocks(tree: KernelTree, loop_nid: int) -> bool:
    """True when ``loop_nid`` encloses ISA leaves owned by more than one block.

    Such a loop was made shared by a prior ``CodeMotion`` co-location; an
    outer-trip Split of it is unsafe (see :func:`_reject_if_shared_loop`).
    """
    return len(_blocks_under_loop(tree, loop_nid) - {_enclosing_block_of(tree, loop_nid)}) > 0


def _reject_if_shared_loop(tree: KernelTree, loop_nid: int) -> None:
    """Reject an outer-trip Split of a loop shared across more than one block.

    ``_do_outer_trip`` rewrites only the target loop's *enclosing* BlockNode
    (``normalize_block`` recomputes that block's bindings from the new loop
    chain). A loop that a prior ``CodeMotion`` made shared — i.e. one enclosing
    ISA leaves of a nested sub-block as well as the enclosing block's own leaf —
    would have only the enclosing block rewritten, leaving the nested block's
    ``iter_value`` referencing the old single loop var while its sibling now
    indexes the composed split affine. The two then address one buffer
    inconsistently (sim out-of-bounds / wrong accumulation).

    Splitting a dim is orthogonal to co-locating producers: do the Split on the
    private per-op loop *before* the ``CodeMotion`` that shares it. This guard
    keeps the broken ordering a loud rejection rather than a wrong kernel.
    """
    if _encloses_multiple_blocks(tree, loop_nid):
        enclosing_block = _enclosing_block_of(tree, loop_nid)
        extra = sorted(_blocks_under_loop(tree, loop_nid) - {enclosing_block})
        raise TransformLegalityError(
            f"Split target loop {loop_nid} is shared across multiple blocks "
            f"(encloses leaves of nested block(s) {extra} besides its enclosing "
            f"block {enclosing_block}); split the per-op loop before CodeMotion co-locates them"
        )


def _find_enclosing_block(tree: KernelTree, nid: int) -> tuple[int, BlockNode]:
    """Walk ancestors of ``nid`` until we hit a BlockNode."""
    for ancestor in reversed(tree.ancestors(nid)):
        data = tree.data(ancestor)
        if isinstance(data, BlockNode):
            return ancestor, data
    raise TransformLegalityError(f"no enclosing BlockNode for nid {nid}")


def _is_slot_reduction_axis(ir: KernelIR, leaf_nid: int, target_axis: str) -> bool:
    """Return whether ``target_axis`` is a slot recipe's reduction axis."""
    result = False
    leaf = ir.tree.data(leaf_nid)
    if isinstance(leaf, ISANode) and leaf.op_cls.RFACTOR_RECIPE == "slot":
        _block_nid, block = _find_enclosing_block(ir.tree, leaf_nid)
        contract = leaf.op_cls.algebraic_contract(leaf.kwargs)
        if isinstance(contract, ReductionContract):
            result = block.axis_map.get(contract.reduction_axis) == target_axis
    return result


def _build_for_chain(tree: KernelTree, stem_loop_var: str, factors: tuple[int, ...]) -> tuple[int, int]:
    """Add a detached chain of ``len(factors)`` ForNodes; return ``(top_nid, bottom_nid)``.

    The loop vars carry temporary names derived from ``stem_loop_var``;
    :func:`normalize_block` renames them dense once the chain is spliced in.
    """
    top_nid: int | None = None
    prev_nid: int | None = None
    for i, extent in enumerate(factors):
        new_nid = tree.add_node(ForNode(loop_var=f"{stem_loop_var}__tmp{i}", extent=extent), parent=None)
        if top_nid is None:
            top_nid = new_nid
        if prev_nid is not None:
            tree.graph.add_edge(prev_nid, new_nid)
        prev_nid = new_nid
    assert top_nid is not None and prev_nid is not None
    return top_nid, prev_nid


def _current_tensorize_width(leaf: ISANode, block: BlockNode, concrete_axis: str) -> int | None:
    """Tile width currently on the leaf for the operand axis matching ``concrete_axis``.

    ``concrete_axis`` is a block iter_var dim (e.g. ``d1``); translate it to
    the abstract op-axis name (e.g. ``F``) via ``block.axis_map`` before
    looking it up in ``OPERAND_AXES`` (which is keyed by abstract names).
    """
    inverse = {concrete: abstract for abstract, concrete in block.axis_map.items()}
    abstract = inverse.get(concrete_axis)
    width: int | None = None
    if abstract is not None:
        op_cls = leaf.op_cls
        for slot, axes in op_cls.OPERAND_AXES.items():
            if abstract not in axes or slot not in leaf.operand_bindings:
                continue
            region = leaf.operand_bindings[slot]
            axis_index = axes.index(abstract)
            if axis_index < len(region.ranges):
                _lo, hi = region.ranges[axis_index]
                if isinstance(hi, Const):
                    width = hi.value
                    break
    return width


def _min_tile_floor(leaf: ISANode, block: BlockNode, concrete_axis: str) -> int | None:
    """Minimum legal innermost tile for ``concrete_axis``, or ``None`` if unconstrained.

    Translates the block iter_var dim (e.g. ``d1``) to the abstract op-axis
    (e.g. ``M``) via ``block.axis_map`` and reads the op's
    ``MIN_TILE_SIZE``. A tensorize-split whose innermost factor falls below
    this floor would shrink the access tile past the operation's scheduling
    minimum, so such a split is illegal.
    """
    inverse = {concrete: abstract for abstract, concrete in block.axis_map.items()}
    abstract = inverse.get(concrete_axis)
    floor: int | None = None
    if abstract is not None:
        floor = leaf.op_cls.MIN_TILE_SIZE.get(abstract)
    return floor


def _covers_exactly(factors: tuple[int, ...], extent: int) -> bool:
    """Whether ``factors`` exactly tile ``extent`` (no under- or over-cover).

    Mirrors TVM Split's mechanism (``loop_transformation.cc`` ~line 421): build
    ``substitute_value = Σ_i var_i · Π(factor_j, j>i)`` with each ``var_i`` bound
    to ``[0, factor_i)`` on an :class:`Analyzer`, then read its constant-integer
    upper bound (TVM's ``ConstIntBoundAnalyzer``). The loop emits this sum in
    Horner-factored form (``acc = acc * factor_i + var_i``), the same affine.
    The substitution ranges over
    ``[0, Π factors)``, so its max is ``Π factors - 1``; exact tiling is
    ``hi + 1 == extent``. TVM accepts ``Π >= extent`` and predicates the ragged
    tail — we are exact-division-only (no predicate path in the renderer), so we
    require equality, rejecting both under-cover and over-cover.
    """
    analyzer = Analyzer()
    substitute: Expr = Const(value=0)
    for i, factor in enumerate(factors):
        var = Var(name=f"_split_v{i}")
        analyzer.bind(var.name, 0, factor)
        substitute = Add(left=Mul(left=substitute, right=Const(value=factor)), right=var)
    _lo, hi = analyzer.const_int_bound(substitute)
    return hi is not None and hi + 1 == extent


def _factorizations(n: int) -> list[tuple[int, int]]:
    """Return every ordered binary factorization of ``n``.

    Each tuple contains factors ``>= 2`` whose product is exactly ``n``.
    Order is significant because ``(2, 4)`` and ``(4, 2)`` name distinct loop
    nests. Deeper factorizations are composed from separate binary Split actions.
    """
    return [(outer, n // outer) for outer in range(2, n) if n % outer == 0 and n // outer >= 2]


__all__ = ["Split", "SplitOption"]
