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

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const, Var, to_affine
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

        The factored loop ``ko`` (``option.target_loop_nid``, extent ``factor``) is
        the matmul's outermost reduction loop; the output tile is ``m_tiles`` rows
        of 128. The emission keeps the PSUM accumulator per-output-tile and folds
        the ``factor`` partials into the SBUF output ``out_sbuf``:

        - ``init_two_stage_0``: the canonical flat ``memset`` that zeroed ``psum``
          is RETARGETED to zero ``out_sbuf`` (the second-stage accumulator), staying
          a root sibling before ``ko``.
        - ``init_two_stage_1``: a NEW per-``ko`` ``memset`` zeroing ``psum`` (bare,
          NO slot), spliced as the FIRST child of ``ko`` (before the ``ki`` nest).
        - run-op: the matmul block's K iter_var role flips ACCUMULATION → PARALLEL
          (each ``ko`` is an independent partial; ``ki`` HW-accumulates in PSUM); its
          ``dst`` region is UNCHANGED (no slot).
        - ``drain_two_stage_0``: a per-``ko`` ``tensor_copy`` (``psum`` → SBUF
          ``sbuf_rfactor``) then a per-``ko`` ``tensor_tensor`` fold
          (``out_sbuf = combiner(out_sbuf, sbuf_rfactor)``), spliced as the LAST two
          children of ``ko``. The fold block carries ``ko`` as ACCUMULATION (the
          closing second-stage reduction on ``out_sbuf``).
        - ``drain_two_stage_1``: empty — the result is already in ``out_sbuf``.

        ``place_buffers`` (LCA) + a rebuilt ``Dependency`` follow, per contract.
        """
        tree = ir.tree
        ko_loop = tree.data(option.target_loop_nid)
        assert isinstance(ko_loop, ForNode)
        ko_var = ko_loop.loop_var

        matmul_leaf = self._owning_matmul_leaf(ir, option.target_loop_nid)
        assert matmul_leaf is not None
        matmul_block_nid = self._enclosing_block_nid(tree, matmul_leaf)
        matmul_leaf_node = tree.data(matmul_leaf)
        assert isinstance(matmul_leaf_node, ISANode)
        op_cls = matmul_leaf_node.op_cls

        psum_name = matmul_leaf_node.operand_bindings["dst"].tensor
        m_tiles = self._partition_tiles(ir, matmul_block_nid, psum_name)
        out_name = self._drain_out_tensor(tree, psum_name)
        identity = float(op_cls.REDUCE_COMBINATOR.identity)
        combiner = op_cls.REDUCE_COMBINATOR.combiner
        free_extent = ir.buffer(out_name).shape[1]

        self._add_rf_buffer(ir, psum_name, out_name)
        self._flip_matmul_k_role(tree, matmul_block_nid)
        self._retarget_init(tree, psum_name, out_name)
        self._remove_flat_block(tree, self._reader_leaf(tree, psum_name, "tensor_copy"))
        self._nest_memset(tree, option.target_loop_nid, psum_name, ko_var, m_tiles, free_extent, identity)
        self._nest_copy(tree, option.target_loop_nid, psum_name, ko_var, m_tiles, free_extent)
        self._nest_combine(tree, option.target_loop_nid, out_name, ko_var, m_tiles, free_extent, combiner)

        place_buffers(tree)
        ir.dependency = Dependency(tree)

    def _enclosing_block_nid(self, tree: KernelTree, nid: int) -> int:
        """Nearest enclosing BlockNode nid of ``nid`` (deepest ancestor block)."""
        for anc in reversed(tree.ancestors(nid)):
            if isinstance(tree.data(anc), BlockNode):
                return anc
        raise TransformLegalityError(f"no enclosing BlockNode for {nid}")

    def _partition_tiles(self, ir: KernelIR, block_nid: int, psum_name: str) -> int:
        """Number of 128-row partition tiles in the matmul output (M_extent // 128).

        Read from the matmul block's iter_var whose abstract axis is the first
        ``dst`` (output) axis — the partition (M) dim of the accumulator.
        """
        block = ir.tree.data(block_nid)
        assert isinstance(block, BlockNode)
        m_abstract = self._op_cls_of_block(ir.tree, block_nid).OPERAND_AXES["dst"][0]
        m_dim = block.axis_map[m_abstract]
        m_extent = next(iv.dom[1] - iv.dom[0] for iv in block.iter_vars if iv.axis == m_dim)
        return m_extent // PARTITION_DIM

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
        and position (a root sibling before ``ko``); only the written tensor changes.
        ``out_name`` and ``psum_name`` share the same ``(m_tiles, free)`` tile shape.
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
        tree: KernelTree,
        ko_loop_nid: int,
        psum_name: str,
        ko_var: str,
        m_tiles: int,
        free_extent: int,
        identity: float,
    ) -> None:
        """Splice ``init_two_stage_1``: a per-``ko`` ``memset(psum)`` as ko's FIRST child.

        Zeros the per-``ko`` PSUM partial before the ``ki`` matmul nest. The block
        binds ``ko`` (axis ``d0``, role PARALLEL — matching the flipped matmul ``ko``,
        so this is an ordinary init, not a re-zero of ``ko``'s own reduction) and the
        partition loop ``m``; its dst is bare ``psum[m]`` (NO ``ko`` slot).
        """
        m_var = "i_d1_0"
        region = self._partition_region(psum_name, m_var, free_extent)
        block = self._partition_block(
            ko_var, m_var, m_tiles, free_extent, AxisRole.PARALLEL, reads=(), writes=(region,)
        )
        leaf = ISANode(op_cls=NKIMemset, operand_bindings={"dst": region}, kwargs={"value": identity})
        self._splice_block_under_ko(tree, ko_loop_nid, block, m_var, m_tiles, leaf, at_front=True)

    def _nest_copy(
        self, tree: KernelTree, ko_loop_nid: int, psum_name: str, ko_var: str, m_tiles: int, free_extent: int
    ) -> None:
        """Splice the first half of ``drain_two_stage_0``: ``tensor_copy(psum → sbuf_rfactor)``.

        Staged after the ``ki`` nest (appended as a child of ``ko``). Moves the
        per-``ko`` PSUM partial into SBUF so the following ``tensor_tensor`` fold can
        read it (``tensor_tensor`` cannot read a PSUM operand). The block binds
        ``ko`` (PARALLEL) + partition loop ``m``; both regions are bare ``[m]``.
        """
        m_var = "i_d1_0"
        src = self._partition_region(psum_name, m_var, free_extent)
        dst = self._partition_region(_RMW_STAGING_BUFFER, m_var, free_extent)
        block = self._partition_block(
            ko_var, m_var, m_tiles, free_extent, AxisRole.PARALLEL, reads=(src,), writes=(dst,)
        )
        leaf = ISANode(op_cls=NKITensorCopy, operand_bindings={"src": src, "dst": dst}, kwargs={})
        self._splice_block_under_ko(tree, ko_loop_nid, block, m_var, m_tiles, leaf, at_front=False)

    def _nest_combine(
        self,
        tree: KernelTree,
        ko_loop_nid: int,
        out_name: str,
        ko_var: str,
        m_tiles: int,
        free_extent: int,
        combiner: str,
    ) -> None:
        """Splice the second half of ``drain_two_stage_0``: the cross-``ko`` SBUF fold.

        ``tensor_tensor(data1=out_sbuf, data2=sbuf_rfactor, dst=out_sbuf, op=combiner)``
        as ko's LAST child. ``data1``/``dst`` are the RMW accumulator (constant
        address across ``ko``), so the block binds ``ko`` as ACCUMULATION — its
        cross-``ko`` carry on ``out_sbuf`` is the closing second-stage reduction.
        ``combiner`` is the op's ``REDUCE_COMBINATOR.combiner`` (``"add"`` for matmul).
        """
        m_var = "i_d1_0"
        out_region = self._partition_region(out_name, m_var, free_extent)
        rf_region = self._partition_region(_RMW_STAGING_BUFFER, m_var, free_extent)
        block = self._partition_block(
            ko_var,
            m_var,
            m_tiles,
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
        self._splice_block_under_ko(tree, ko_loop_nid, block, m_var, m_tiles, leaf, at_front=False)

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

    def _partition_region(self, tensor: str, m_var: str, free_extent: int) -> BufferRegion:
        """Build the canonical ``tensor[m : +128, 0 : +free_extent]`` partition region.

        Axis 0 is the bare partition-tile index ``Var(m_var)`` with width 128; the
        free axis is loopless (full ``free_extent``).
        """
        return BufferRegion(
            tensor=tensor,
            ranges=((Var(name=m_var), Const(value=PARTITION_DIM)), (Const(value=0), Const(value=free_extent))),
        )

    def _partition_block(
        self,
        ko_var: str,
        m_var: str,
        m_tiles: int,
        free_extent: int,
        d0_role: AxisRole,
        reads: tuple[BufferRegion, ...],
        writes: tuple[BufferRegion, ...],
    ) -> BlockNode:
        """Build a per-``ko`` memset/copy/combine :class:`BlockNode` nested under ``ko``.

        Declares three iter_vars — ``ko`` (axis ``d0``, role ``d0_role``, bound to the
        shared matmul ``ko`` ForNode's ``ko_var``), the partition tile ``m`` (axis
        ``d1``, bound to ``m_var``), and the loopless free axis (``d2``). Only the
        ``m`` loop is materialized as a ForNode by the caller; ``ko`` is the shared
        parent loop, so the block reads ``ko_var`` from it. ``d0_role`` is PARALLEL
        for init/copy (ordinary per-``ko`` ops) and ACCUMULATION for the closing
        ``tensor_tensor`` fold (whose cross-``ko`` carry is the second-stage reduction).
        """
        return BlockNode(
            iter_vars=(
                IterVar(axis="d0", dom=(0, m_tiles * PARTITION_DIM), role=d0_role),
                IterVar(axis="d1", dom=(0, m_tiles * PARTITION_DIM), role=AxisRole.PARALLEL),
                IterVar(axis="d2", dom=(0, free_extent), role=AxisRole.PARALLEL),
            ),
            iter_values=(Var(name=ko_var), Var(name=m_var), Const(value=0)),
            reads=reads,
            writes=writes,
            alloc_buffers=(),
            axis_map={"K": "d0", "P": "d1", "F": "d2"},
        )

    def _splice_block_under_ko(
        self,
        tree: KernelTree,
        ko_loop_nid: int,
        block: BlockNode,
        m_var: str,
        m_tiles: int,
        leaf: ISANode,
        at_front: bool,
    ) -> None:
        """Add ``block`` (with its ``m`` ForNode + ``leaf``) as a child of ``ko``.

        The new block becomes a child of the matmul's ``ko`` ForNode, carrying its
        own partition loop ``m`` (extent ``m_tiles``) and the single ISA ``leaf``.
        ``at_front`` places it before all existing children (the per-``ko`` memset);
        otherwise after them (the copy, then the combine). The resulting dataflow
        order under ``ko`` is ``memset → ki-matmul-nest → tensor_copy → tensor_tensor``.
        """
        block_nid = tree.add_node(block, parent=ko_loop_nid)
        m_nid = tree.add_node(ForNode(loop_var=m_var, extent=m_tiles), parent=block_nid)
        tree.add_node(leaf, parent=m_nid)
        existing = [c for c in tree.children(ko_loop_nid) if c != block_nid]
        new_order = [block_nid, *existing] if at_front else [*existing, block_nid]
        for child in tree.children(ko_loop_nid):
            tree.graph.remove_edge(ko_loop_nid, child)
        for child in new_order:
            tree.graph.add_edge(ko_loop_nid, child)

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
