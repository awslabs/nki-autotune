"""``RFactor`` transform — one-stage → two-stage accumulation (spec 2026-06-12 §2).

Factors a reduction loop into ``ko``/``ki`` and restructures the one-stage
accumulation (init once, reduce, drain once) into the FUSED two-stage form:
``ki`` produces a per-``ko`` partial in PSUM, and ``ko`` sums the partials into an
SBUF accumulator via a ``tensor_tensor`` fold. Loops are NOT reordered.

This is the fused single-accumulator form, NOT TVM's multi-slot terminal: the
PSUM accumulator is per-output-tile (it is re-zeroed every ``ko``, never grown by
``factor`` and never carries a ``ko`` slot), so no ``ko``-stride term ever rides
its M (partition-tile) axis — a later ``Split(M)`` cannot corrupt it. See
``docs/superpowers/specs/2026-06-12-same-prefix-computeat-and-two-stage-rfactor-design.md``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.codegen.compact import compact_shapes
from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const, Expr, Var, substitute, to_affine
from nkigym.ir.buffer_placement import place_buffers
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import PARTITION_DIM, BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.base import AxisRole, NKIOp
from nkigym.ops.memset import NKIMemset
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_tensor import NKITensorTensor
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption

_RMW_STAGING_BUFFER = "sbuf_rfactor"
"""The ``"rmw"`` recipe's transient SBUF staging buffer (spec §2.3). The per-``ko``
PSUM partial is copied here before the ``tensor_tensor`` fold, because
``tensor_tensor`` cannot read a PSUM operand."""


@dataclass(frozen=True)
class RFactorOption(TransformOption):
    """Factor the reduction loop ``target_loop_nid``.

    Attributes:
        target_loop_nid: the ForNode (a reduction/ACCUMULATION loop) to factor.
        factor_axis: retained for API parity with TVM ``rfactor(loop, factor_axis)``;
            the fused form keeps a per-output-tile accumulator (no factor slot), so
            this is accepted but does not change the emission.
    """

    target_loop_nid: int
    factor_axis: int = 0


class RFactor(Transform):
    """One-stage → two-stage accumulation: per-``ko`` PSUM partial + SBUF fold."""

    def analyze(self, ir: KernelIR) -> list[RFactorOption]:
        """Enumerate every ForNode that binds an ACCUMULATION axis of an
        rfactorable op (RFACTOR_RECIPE not None)."""
        options: list[RFactorOption] = []
        for nid in ir.tree.preorder():
            if not isinstance(ir.tree.data(nid), ForNode):
                continue
            if self._rfactorable(ir, nid):
                options.append(RFactorOption(target_loop_nid=nid, factor_axis=0))
        return options

    def apply(self, ir: KernelIR, option: RFactorOption) -> KernelIR:
        """Re-check legality, deep-copy, emit the two-stage accumulation, return."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        self._emit_rmw(new_ir, option)
        return new_ir

    def _rfactorable(self, ir: KernelIR, loop_nid: int) -> bool:
        """True iff ``loop_nid`` binds an ACCUMULATION axis whose owning op
        declares RFACTOR_RECIPE='rmw' and a REDUCE_COMBINATOR."""
        leaf = self._owning_matmul_leaf(ir, loop_nid)
        result = False
        if leaf is not None:
            op_cls = ir.tree.data(leaf).op_cls
            block = self._enclosing_block(ir, leaf)
            axis = self._loop_axis(ir, loop_nid, block)
            if (
                op_cls.RFACTOR_RECIPE == "rmw"
                and op_cls.REDUCE_COMBINATOR is not None
                and axis is not None
                and self._role_of(block, axis) == AxisRole.ACCUMULATION
            ):
                result = True
        return result

    def _check_legality(self, ir: KernelIR, option: RFactorOption) -> None:
        """Raise TransformLegalityError if the option is not a valid rmw rfactor."""
        nid = option.target_loop_nid
        if nid not in ir.tree.graph or not isinstance(ir.tree.data(nid), ForNode):
            raise TransformLegalityError(f"RFactor target {nid} is not a ForNode in the tree")
        if not self._rfactorable(ir, nid):
            raise TransformLegalityError(
                f"RFactor target loop {nid} does not bind an ACCUMULATION axis of an "
                f"op with RFACTOR_RECIPE='rmw' + a REDUCE_COMBINATOR"
            )

    def _emit_rmw(self, ir: KernelIR, option: RFactorOption) -> None:
        """Restructure one-stage → fused two-stage accumulation on ``ir`` (spec §2.1).

        The factored loop ``ko`` (``option.target_loop_nid``) is the matmul's
        outermost reduction loop; ``ki`` is its INNERMOST reduction loop, whose
        subtree ends in the matmul run-op. The emission keeps the PSUM accumulator
        per-output-tile and folds the ``factor`` partials into the SBUF output
        ``out_sbuf``, anchoring the gadgets to ``ki`` (spec 2026-07-07 §B):

        - ``init_two_stage_0``: the canonical flat ``memset`` that zeroed ``psum``
          is RETARGETED to zero ``out_sbuf`` (the second-stage accumulator), staying
          a root sibling before ``ko``.
        - ``init_two_stage_1``: a NEW ``memset`` zeroing ``psum`` (bare, NO slot),
          spliced as ``ki``'s PRECEDING sibling.
        - run-op: the matmul block's K iter_var role flips ACCUMULATION → PARALLEL
          (each ``ko`` is an independent partial; ``ki`` HW-accumulates in PSUM); its
          ``dst`` region is UNCHANGED (no slot).
        - ``drain_two_stage_0``: a ``tensor_copy`` (``psum`` → SBUF ``sbuf_rfactor``)
          then a ``tensor_tensor`` fold (``out_sbuf = combiner(out_sbuf,
          sbuf_rfactor)``), spliced as ``ki``'s FOLLOWING siblings. The fold block
          carries ``ko`` as ACCUMULATION (the closing second-stage reduction on
          ``out_sbuf``).
        - ``drain_two_stage_1``: empty — the result is already in ``out_sbuf``.

        The gadgets are sized to the footprint R — the accumulator region the
        ``ki``-subtree writes over one full ``ki`` execution: partition loops between
        ``ki`` and the matmul are MATERIALIZED (early-packed: the 16-trip ``M`` loop),
        free loops are ABSORBED into the op width. ``place_buffers`` (LCA) +
        ``compact_shapes`` (shape + ``list_len`` shrink) + a rebuilt ``Dependency``
        follow, per contract.
        """
        tree = ir.tree
        ko_loop = tree.data(option.target_loop_nid)
        assert isinstance(ko_loop, ForNode)
        ko_var = ko_loop.loop_var

        matmul_leaf = self._owning_matmul_leaf(ir, option.target_loop_nid)
        assert matmul_leaf is not None
        matmul_block_nid = self._enclosing_block_nid(tree, matmul_leaf)
        matmul_node = tree.data(matmul_leaf)
        assert isinstance(matmul_node, ISANode)
        op_cls = matmul_node.op_cls

        psum_name = matmul_node.operand_bindings["dst"].tensor
        out_name = self._drain_out_tensor(tree, psum_name)
        identity = float(op_cls.REDUCE_COMBINATOR.identity)
        combiner = op_cls.REDUCE_COMBINATOR.combiner

        ki_nid = self._ki_loop_nid(ir, option.target_loop_nid)
        footprint = self._footprint(ir, ki_nid, matmul_leaf)
        part_lo = matmul_node.operand_bindings["dst"].ranges[0][0]
        free_extent = self._absorbed_free_width(ir, ki_nid, matmul_leaf)
        free_lo = self._gadget_free_lo(ir, ki_nid, matmul_leaf)

        self._add_rf_buffer(ir, psum_name, out_name)
        self._flip_matmul_k_role(tree, matmul_block_nid)
        self._retarget_init(tree, psum_name, out_name)
        self._remove_flat_block(tree, self._reader_leaf(tree, psum_name, "tensor_copy"))
        self._nest_memset(ir, matmul_leaf, ki_nid, psum_name, ko_var, footprint, part_lo, free_lo, free_extent, identity)
        self._nest_copy(ir, matmul_leaf, ki_nid, psum_name, ko_var, footprint, part_lo, free_lo, free_extent)
        self._nest_combine(ir, matmul_leaf, ki_nid, out_name, ko_var, footprint, part_lo, free_lo, free_extent, combiner)

        place_buffers(tree)
        compact_shapes(tree)
        ir.dependency = Dependency(tree)

    def _enclosing_block_nid(self, tree: KernelTree, nid: int) -> int:
        """Nearest enclosing BlockNode nid of ``nid`` (deepest ancestor block)."""
        for anc in reversed(tree.ancestors(nid)):
            if isinstance(tree.data(anc), BlockNode):
                return anc
        raise TransformLegalityError(f"no enclosing BlockNode for {nid}")

    def _ki_loop_nid(self, ir: KernelIR, ko_loop_nid: int) -> int:
        """The INNERMOST ACCUMULATION (K-axis) loop enclosing the matmul, at or below ko.

        The matmul's reduction is driven by its K-axis loops; ki is the deepest of them.
        Found as the last (innermost) matmul-ancestor ForNode whose loop_var binds the
        matmul block's K axis. For early-packed ki sits directly under ko; for k28 ki is
        the innermost loop and the matmul is its sole body.
        """
        tree = ir.tree
        matmul_leaf = self._owning_matmul_leaf(ir, ko_loop_nid)
        assert matmul_leaf is not None
        block = self._enclosing_block(ir, matmul_leaf)
        op_cls = self._op_cls_of_block(tree, self._enclosing_block_nid(tree, matmul_leaf))
        reduction_abstract = next(a for a, role in op_cls.AXIS_ROLES.items() if role == AxisRole.ACCUMULATION)
        k_axis = block.axis_map[reduction_abstract]
        k_binding_vars = self._axis_binding_loopvars(block, k_axis)
        k_loops = [
            a
            for a in tree.ancestors(matmul_leaf)
            if isinstance(tree.data(a), ForNode) and tree.data(a).loop_var in k_binding_vars
        ]
        return k_loops[-1]

    def _axis_binding_loopvars(self, block: BlockNode, axis: str) -> set[str]:
        """Loop vars appearing in the iter_value of ``axis`` (the loops that bind it)."""
        value = next(v for iv, v in zip(block.iter_vars, block.iter_values) if iv.axis == axis)
        return {n for n in to_affine(value) if n is not None}

    def _footprint(self, ir: KernelIR, ki_loop_nid: int, matmul_leaf: int) -> list[tuple[str, int]]:
        """Ordered (loop_var, extent) of the ForNodes STRICTLY between ki and the matmul
        leaf whose loop_var binds the matmul's PARTITION (dst axis-0) dim.

        These are the partition-tile loops the ki-subtree sweeps over one ki execution;
        the gadgets materialize them (early-packed: the 16-trip M loop). Free-axis loops
        between ki and the matmul are absorbed into the op width, so they are NOT returned.
        For k28 there are no loops between ki and the matmul, so this is empty (R = one
        tile, loopless gadgets).
        """
        tree = ir.tree
        block = self._enclosing_block(ir, matmul_leaf)
        m_abstract = tree.data(matmul_leaf).op_cls.OPERAND_AXES["dst"][0]
        m_axis = block.axis_map[m_abstract]
        m_binding_vars = self._axis_binding_loopvars(block, m_axis)
        between = [
            a for a in tree.ancestors(matmul_leaf) if isinstance(tree.data(a), ForNode) and ki_loop_nid in tree.ancestors(a)
        ]
        return [(tree.data(a).loop_var, tree.data(a).extent) for a in between if tree.data(a).loop_var in m_binding_vars]

    def _absorbed_free_width(self, ir: KernelIR, ki_loop_nid: int, matmul_leaf: int) -> int:
        """Full free extent the ki-subtree sweeps: the matmul dst free-tile width times the
        product of the FREE-binding loop trips strictly between ki and the matmul.

        Free loops between ki and the matmul are absorbed into one wide gadget op (memset /
        tensor_copy / tensor_tensor free cap >= 2048). Early-packed: 512 * 4 (i_d2_0) =
        2048. k28: 512 * 1 (no free loop between) = 512.
        """
        tree = ir.tree
        block = self._enclosing_block(ir, matmul_leaf)
        dst_region = tree.data(matmul_leaf).operand_bindings["dst"]
        free_abstract = tree.data(matmul_leaf).op_cls.OPERAND_AXES["dst"][1]
        free_axis = block.axis_map[free_abstract]
        free_binding_vars = self._axis_binding_loopvars(block, free_axis)
        tile_width = dst_region.ranges[1][1]
        assert isinstance(tile_width, Const)
        width = tile_width.value
        for a in tree.ancestors(matmul_leaf):
            data = tree.data(a)
            if isinstance(data, ForNode) and ki_loop_nid in tree.ancestors(a) and data.loop_var in free_binding_vars:
                width *= data.extent
        return width

    def _gadget_free_lo(self, ir: KernelIR, ki_loop_nid: int, matmul_leaf: int) -> Expr:
        """Free-axis (dst axis-1) lo the gadgets must share with the matmul dst region.

        RFactor leaves the matmul dst region UNCHANGED, so its free lo still carries the
        enclosing output-tile loop offset (e.g. ``i_d2_0*512`` for k28). The gadgets must
        address the SAME frame on the free axis, else ``compact_shapes`` sees inconsistent
        offsets across the psum/out touchers and never rebases them (the shape stays wide).

        The gadgets ABSORB the free-binding loops strictly between ki and the matmul into
        one wide op (see :meth:`_absorbed_free_width`), so those loop vars index WITHIN the
        gadget's own width and must be zeroed in the shared lo. This substitutes each such
        absorbed free-loop var -> ``Const(0)`` in the matmul dst free lo:

        - k28: no free loop between ki and the matmul -> substitute nothing -> ``i_d2_0*512``
          (the enclosing output-tile offset). All psum touchers then share ``i_d2_0*512``,
          so ``i_d2_0`` is a consistent anchor and the free axis rebases to ``0:512``.
        - early-packed: ``i_d2_0`` IS an absorbed free loop -> substitute ``i_d2_0`` -> 0 ->
          ``0*512`` (normalises to ``Const(0)``), identical to the previous hardcoded lo, so
          the free axis stays the full absorbed 2048 and the render is unchanged.
        """
        tree = ir.tree
        block = self._enclosing_block(ir, matmul_leaf)
        dst_region = tree.data(matmul_leaf).operand_bindings["dst"]
        free_abstract = tree.data(matmul_leaf).op_cls.OPERAND_AXES["dst"][1]
        free_axis = block.axis_map[free_abstract]
        free_binding_vars = self._axis_binding_loopvars(block, free_axis)
        absorbed = {
            tree.data(a).loop_var
            for a in tree.ancestors(matmul_leaf)
            if isinstance(tree.data(a), ForNode)
            and ki_loop_nid in tree.ancestors(a)
            and tree.data(a).loop_var in free_binding_vars
        }
        return substitute(dst_region.ranges[1][0], {var: Const(value=0) for var in absorbed})

    def _op_cls_of_block(self, tree: KernelTree, block_nid: int) -> type[NKIOp]:
        """Return the op class of the rfactorable (reduction) leaf under ``block_nid``.

        A block may own several ISA leaves once co-location nested a memset / drain
        beside the matmul. ``tree.descendants`` is an unordered set, so select the
        single leaf whose op declares an ``RFACTOR_RECIPE`` (the matmul).
        """
        leaves = [d for d in tree.descendants(block_nid) if isinstance(tree.data(d), ISANode)]
        rfactorable = [n for n in leaves if tree.data(n).op_cls.RFACTOR_RECIPE is not None]
        if len(rfactorable) != 1:
            raise TransformLegalityError(
                f"block {block_nid} must own exactly one rfactorable leaf; got {len(rfactorable)}"
            )
        return tree.data(rfactorable[0]).op_cls

    def _drain_out_tensor(self, tree: KernelTree, psum_name: str) -> str:
        """Tensor the drain ``tensor_copy`` writes (reads ``psum_name``, writes SBUF out)."""
        for nid in tree.preorder():
            data = tree.data(nid)
            if isinstance(data, ISANode) and data.op_cls.NAME == "tensor_copy":
                if data.operand_bindings["src"].tensor == psum_name:
                    return data.operand_bindings["dst"].tensor
        raise TransformLegalityError(f"no drain tensor_copy reading {psum_name!r}")

    def _add_rf_buffer(self, ir: KernelIR, psum_name: str, out_name: str) -> None:
        """Add the SBUF staging buffer ``sbuf_rfactor`` (spec §2.3 transient).

        Mirrors the output tile's logical shape (per-output-tile, NOT grown by
        ``factor``) in SBUF; ``place_buffers`` relocates it to its LCA afterwards.
        Declared on whichever block currently allocs ``psum_name`` (a stable
        anchor; the final placement is LCA-driven regardless).
        """
        tree = ir.tree
        out_buf = ir.buffer(out_name)
        rf_buf = Buffer(name=_RMW_STAGING_BUFFER, shape=out_buf.shape, dtype=out_buf.dtype, location="sbuf")
        for nid in tree.blocks():
            block = tree.data(nid)
            assert isinstance(block, BlockNode)
            if any(buf.name == psum_name for buf in block.alloc_buffers):
                tree.graph.nodes[nid]["data"] = replace(block, alloc_buffers=(*block.alloc_buffers, rf_buf))
                return
        raise TransformLegalityError(f"no block allocates {psum_name!r}")

    def _flip_matmul_k_role(self, tree: KernelTree, block_nid: int) -> None:
        """Flip the matmul block's K iter_var ACCUMULATION → PARALLEL (regions unchanged).

        Each ``ko`` is now an independent partial, so the K axis (driving both
        ``ko`` and ``ki``) is data-parallel in the run-op block; the inner ``ki``
        accumulate is the HW ``+=`` encapsulated in the single matmul leaf. The
        closing cross-``ko`` reduction re-emerges as ACCUMULATION on the stage-2
        ``tensor_tensor`` block. Legal only because the reducer is assoc + comm.
        """
        block = tree.data(block_nid)
        assert isinstance(block, BlockNode)
        op_cls = self._op_cls_of_block(tree, block_nid)
        reduction_abstract = next(a for a, role in op_cls.AXIS_ROLES.items() if role == AxisRole.ACCUMULATION)
        k_axis = block.axis_map[reduction_abstract]
        new_iter_vars = tuple(
            replace(iv, role=AxisRole.PARALLEL) if iv.axis == k_axis else iv for iv in block.iter_vars
        )
        tree.graph.nodes[block_nid]["data"] = replace(block, iter_vars=new_iter_vars)

    def _retarget_init(self, tree: KernelTree, psum_name: str, out_name: str) -> None:
        """Retarget the canonical flat ``memset(psum)`` to ``memset(out_sbuf)`` in place.

        ``init_one_stage`` zeroed the PSUM accumulator; the two-stage form's
        ``init_two_stage_0`` zeros the SBUF accumulator instead (PSUM is re-zeroed
        per-``ko`` by ``init_two_stage_1``). The block keeps its loop nest, iter_vars,
        and position (a root sibling before ``ko``); only the written tensor name is
        retargeted ``psum_name`` → ``out_name`` (the region ranges are preserved).
        """
        leaf_nid = self._writer_leaf(tree, psum_name, "memset")
        block_nid = self._enclosing_block_nid(tree, leaf_nid)
        block = tree.data(block_nid)
        assert isinstance(block, BlockNode)
        new_writes = tuple(self._retarget_region(w, psum_name, out_name) for w in block.writes)
        new_reads = tuple(self._retarget_region(r, psum_name, out_name) for r in block.reads)
        tree.graph.nodes[block_nid]["data"] = replace(block, reads=new_reads, writes=new_writes)
        leaf = tree.data(leaf_nid)
        assert isinstance(leaf, ISANode)
        new_bindings = {
            slot: self._retarget_region(region, psum_name, out_name) for slot, region in leaf.operand_bindings.items()
        }
        tree.graph.nodes[leaf_nid]["data"] = ISANode(
            op_cls=leaf.op_cls, operand_bindings=new_bindings, kwargs=dict(leaf.kwargs)
        )

    def _retarget_region(self, region: BufferRegion, old_tensor: str, new_tensor: str) -> BufferRegion:
        """Rename ``region``'s tensor ``old_tensor`` → ``new_tensor`` (ranges unchanged)."""
        if region.tensor != old_tensor:
            return region
        return BufferRegion(tensor=new_tensor, ranges=region.ranges)

    def _nest_memset(
        self,
        ir: KernelIR,
        matmul_leaf: int,
        ki_nid: int,
        psum_name: str,
        ko_var: str,
        footprint: list[tuple[str, int]],
        part_lo: Expr,
        free_lo: Expr,
        free_extent: int,
        identity: float,
    ) -> None:
        """Splice ``init_two_stage_1``: a ``memset(psum)`` as ``ki``'s preceding sibling.

        Zeros the per-``ki`` PSUM partial before the ``ki`` matmul nest. The block
        binds ``ko`` (axis ``d0``, role PARALLEL — matching the flipped matmul ``ko``,
        so this is an ordinary init, not a re-zero of ``ko``'s own reduction) and the
        output partition dim; its dst is bare ``psum[part_lo]`` (NO ``ko`` slot). The
        ``footprint`` partition loops (if any) are materialized by ``_splice_beside_ki``.
        """
        region = self._partition_region(psum_name, part_lo, free_lo, free_extent)
        block = self._gadget_block(
            ir, matmul_leaf, ko_var, footprint, free_extent, AxisRole.PARALLEL, reads=(), writes=(region,)
        )
        leaf = ISANode(op_cls=NKIMemset, operand_bindings={"dst": region}, kwargs={"value": identity})
        self._splice_beside_ki(ir.tree, ki_nid, block, footprint, leaf, at_front=True)

    def _nest_copy(
        self,
        ir: KernelIR,
        matmul_leaf: int,
        ki_nid: int,
        psum_name: str,
        ko_var: str,
        footprint: list[tuple[str, int]],
        part_lo: Expr,
        free_lo: Expr,
        free_extent: int,
    ) -> None:
        """Splice the first half of ``drain_two_stage_0``: ``tensor_copy(psum → sbuf_rfactor)``.

        Staged as ``ki``'s following sibling. Moves the per-``ki`` PSUM partial into
        SBUF so the following ``tensor_tensor`` fold can read it (``tensor_tensor``
        cannot read a PSUM operand). The block binds ``ko`` (PARALLEL) + the output
        partition dim; both regions are bare ``[part_lo]``.
        """
        src = self._partition_region(psum_name, part_lo, free_lo, free_extent)
        dst = self._partition_region(_RMW_STAGING_BUFFER, part_lo, free_lo, free_extent)
        block = self._gadget_block(
            ir, matmul_leaf, ko_var, footprint, free_extent, AxisRole.PARALLEL, reads=(src,), writes=(dst,)
        )
        leaf = ISANode(op_cls=NKITensorCopy, operand_bindings={"src": src, "dst": dst}, kwargs={})
        self._splice_beside_ki(ir.tree, ki_nid, block, footprint, leaf, at_front=False)

    def _nest_combine(
        self,
        ir: KernelIR,
        matmul_leaf: int,
        ki_nid: int,
        out_name: str,
        ko_var: str,
        footprint: list[tuple[str, int]],
        part_lo: Expr,
        free_lo: Expr,
        free_extent: int,
        combiner: str,
    ) -> None:
        """Splice the second half of ``drain_two_stage_0``: the cross-``ko`` SBUF fold.

        ``tensor_tensor(data1=out_sbuf, data2=sbuf_rfactor, dst=out_sbuf, op=combiner)``
        as ``ki``'s following sibling (after the copy). ``data1``/``dst`` are the RMW
        accumulator (constant address across ``ko``), so the block binds ``ko`` as
        ACCUMULATION — its cross-``ko`` carry on ``out_sbuf`` is the closing
        second-stage reduction. ``combiner`` is the op's ``REDUCE_COMBINATOR.combiner``
        (``"add"`` for matmul).
        """
        out_region = self._partition_region(out_name, part_lo, free_lo, free_extent)
        rf_region = self._partition_region(_RMW_STAGING_BUFFER, part_lo, free_lo, free_extent)
        block = self._gadget_block(
            ir,
            matmul_leaf,
            ko_var,
            footprint,
            free_extent,
            AxisRole.ACCUMULATION,
            reads=(out_region, rf_region),
            writes=(out_region,),
        )
        leaf = ISANode(
            op_cls=NKITensorTensor,
            operand_bindings={"data1": out_region, "data2": rf_region, "dst": out_region},
            kwargs={"op": combiner},
        )
        self._splice_beside_ki(ir.tree, ki_nid, block, footprint, leaf, at_front=False)

    def _remove_flat_block(self, tree: KernelTree, leaf_nid: int) -> None:
        """Delete the canonical flat block owning ``leaf_nid`` (block + loop + leaf).

        The flat memset/drain blocks are perfect single-leaf nests under the root: a
        BlockNode, one partition ForNode, one ISA leaf. Detach the block from its
        parent (preserving sibling order) and remove all three nodes.
        """
        block_nid = self._enclosing_block_nid(tree, leaf_nid)
        parent = tree.parent(block_nid)
        assert parent is not None
        remaining = [c for c in tree.children(parent) if c != block_nid]
        _replace_in_parent_children(tree, parent, [block_nid], [])
        assert tree.children(parent) == remaining
        for nid in tree.descendants(block_nid) | {block_nid}:
            tree.graph.remove_node(nid)

    def _partition_region(self, tensor: str, part_lo: Expr, free_lo: Expr, free_extent: int) -> BufferRegion:
        """Canonical ``tensor[part_lo : +128, free_lo : +free_extent]`` region.

        ``part_lo`` is the matmul dst partition (axis-0) offset — a bare loop Var when a
        footprint loop is materialized (early-packed ``i_d1_0``) or a compound affine
        inherited from the enclosing output-tile loops (k28 ``i_d1_0*4 + i_d1_1``).
        ``free_lo`` is the matmul dst free (axis-1) offset with the absorbed free-loop vars
        stripped (see :meth:`_gadget_free_lo`) — ``i_d2_0*512`` for k28 (so ``i_d2_0``
        anchors psum consistently and the free axis rebases to ``0:512``), ``Const(0)`` for
        early-packed. The free axis spans the full absorbed ``free_extent``.
        """
        return BufferRegion(
            tensor=tensor,
            ranges=((part_lo, Const(value=PARTITION_DIM)), (free_lo, Const(value=free_extent))),
        )

    def _gadget_block(
        self,
        ir: KernelIR,
        matmul_leaf: int,
        ko_var: str,
        footprint: list[tuple[str, int]],
        free_extent: int,
        d0_role: AxisRole,
        reads: tuple[BufferRegion, ...],
        writes: tuple[BufferRegion, ...],
    ) -> BlockNode:
        """Build a per-``ki`` gadget :class:`BlockNode` bracketing the ``ki`` loop.

        iter_vars mirror the matmul block: ``d0`` (K, bound to ``ko_var``, role ``d0_role``),
        ``d1`` (the output partition dim, bound to the matmul's own partition iter_value —
        ``i_d1_0`` early-packed, ``i_d1_0*4 + i_d1_1`` k28), and ``d2`` (free, loopless).
        The ``footprint`` partition loops are materialized by the caller as ForNodes; when
        empty (k28) the ``d1`` value is inherited from the enclosing output-tile loops.

        The ``d2`` iter_value is left as ``Const(0)`` for all gadgets. This is intentionally
        inert: the free axis is loopless (the gadgets absorb free loops into their op width),
        so no loop resolves to ``d2``; the free position lives on the ``BufferRegion`` ranges
        (``free_lo``), which is what codegen and dependency analysis consume.
        """
        tree = ir.tree
        block = self._enclosing_block(ir, matmul_leaf)
        op_cls = tree.data(matmul_leaf).op_cls
        m_axis = block.axis_map[op_cls.OPERAND_AXES["dst"][0]]
        m_value = next(v for iv, v in zip(block.iter_vars, block.iter_values) if iv.axis == m_axis)
        m_dom = next(iv.dom for iv in block.iter_vars if iv.axis == m_axis)
        return BlockNode(
            iter_vars=(
                IterVar(axis="d0", dom=(0, m_dom[1]), role=d0_role),
                IterVar(axis="d1", dom=m_dom, role=AxisRole.PARALLEL),
                IterVar(axis="d2", dom=(0, free_extent), role=AxisRole.PARALLEL),
            ),
            iter_values=(Var(name=ko_var), m_value, Const(value=0)),
            reads=reads,
            writes=writes,
            alloc_buffers=(),
            axis_map={"K": "d0", "P": "d1", "F": "d2"},
        )

    def _splice_beside_ki(
        self,
        tree: KernelTree,
        ki_loop_nid: int,
        block: BlockNode,
        footprint: list[tuple[str, int]],
        leaf: ISANode,
        at_front: bool,
    ) -> None:
        """Splice ``block`` (with the ``footprint`` partition loops + ``leaf``) as a sibling
        of ``ki_loop_nid`` under ki's parent.

        ``at_front`` puts it immediately before ``ki`` (the init memset); otherwise
        immediately after (the copy, then the fold), so the order under ki's parent is
        ``memset -> ki -> tensor_copy -> tensor_tensor``. Each footprint entry becomes a
        materialized ForNode (outer->inner); an empty footprint (k28) attaches ``leaf``
        directly to ``block`` (a single loopless op).
        """
        parent = tree.parent(ki_loop_nid)
        assert parent is not None, f"ki loop {ki_loop_nid} has no parent"
        block_nid = tree.add_node(block, parent=parent)
        cursor = block_nid
        for loop_var, extent in footprint:
            cursor = tree.add_node(ForNode(loop_var=loop_var, extent=extent), parent=cursor)
        tree.add_node(leaf, parent=cursor)
        siblings = [c for c in tree.children(parent) if c != block_nid]
        new_order = [block_nid, *siblings] if at_front else [*siblings, block_nid]
        for child in tree.children(parent):
            tree.graph.remove_edge(parent, child)
        for child in new_order:
            tree.graph.add_edge(parent, child)

    def _writer_leaf(self, tree: KernelTree, tensor: str, op_name: str) -> int:
        """The single ISA leaf with NAME ``op_name`` that writes ``tensor`` (dst slot)."""
        for nid in tree.preorder():
            data = tree.data(nid)
            if isinstance(data, ISANode) and data.op_cls.NAME == op_name:
                if data.operand_bindings.get("dst") is not None and data.operand_bindings["dst"].tensor == tensor:
                    return nid
        raise TransformLegalityError(f"no {op_name} writing {tensor!r}")

    def _reader_leaf(self, tree: KernelTree, tensor: str, op_name: str) -> int:
        """The single ISA leaf with NAME ``op_name`` that reads ``tensor`` (src slot)."""
        for nid in tree.preorder():
            data = tree.data(nid)
            if isinstance(data, ISANode) and data.op_cls.NAME == op_name:
                if data.operand_bindings.get("src") is not None and data.operand_bindings["src"].tensor == tensor:
                    return nid
        raise TransformLegalityError(f"no {op_name} reading {tensor!r}")

    def _owning_matmul_leaf(self, ir: KernelIR, loop_nid: int) -> int | None:
        """The single ISA leaf under ``loop_nid`` whose op is rfactorable, or None."""
        leaves = [
            d
            for d in ir.tree.descendants(loop_nid)
            if isinstance(ir.tree.data(d), ISANode) and ir.tree.data(d).op_cls.RFACTOR_RECIPE is not None
        ]
        return leaves[0] if len(leaves) == 1 else None

    def _enclosing_block(self, ir: KernelIR, nid: int) -> BlockNode:
        """Nearest enclosing BlockNode payload of ``nid``."""
        for anc in reversed(ir.tree.ancestors(nid)):
            if isinstance(ir.tree.data(anc), BlockNode):
                return ir.tree.data(anc)
        raise TransformLegalityError(f"no enclosing BlockNode for {nid}")

    def _loop_axis(self, ir: KernelIR, loop_nid: int, block: BlockNode) -> str | None:
        """The concrete axis the loop's loop_var binds, via the block's iter_values."""
        loop_var = ir.tree.data(loop_nid).loop_var
        for iv, value in zip(block.iter_vars, block.iter_values):
            if loop_var in to_affine(value):
                return iv.axis
        return None

    def _role_of(self, block: BlockNode, axis: str) -> AxisRole:
        """Role the block assigns to ``axis`` (default PARALLEL if absent)."""
        for iv in block.iter_vars:
            if iv.axis == axis:
                return iv.role
        return AxisRole.PARALLEL


__all__ = ["RFactor", "RFactorOption"]
