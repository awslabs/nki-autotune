"""``RFactor`` transform — faithful port of TVM ``tir.Schedule.rfactor``.

Splits a reduction loop into an rf-block (fills a ``[factor, *tile(out)]``
rf-buffer; factored loop flipped reduction→parallel) and a separate wb-block
(reduces the slots into the original output). Loops are NOT reordered. See
``docs/superpowers/specs/2026-06-07-rfactor-transform-design.md``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Add, Const, Expr, Mul, Var, to_affine
from nkigym.ir.buffer_placement import place_buffers
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import PARTITION_DIM, BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.base import AxisRole, NKIOp
from nkigym.ops.memset import NKIMemset
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_tensor import NKITensorTensor
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class RFactorOption(TransformOption):
    """Factor the reduction loop ``target_loop_nid``.

    Attributes:
        target_loop_nid: the ForNode (a reduction/ACCUMULATION loop) to factor.
        factor_axis: position of the prepended ``[factor]`` dim in the rf-buffer
            (TVM ``rfactor(loop, factor_axis)``; default 0).
    """

    target_loop_nid: int
    factor_axis: int = 0


class RFactor(Transform):
    """Faithful ``tir.Schedule.rfactor``: rf-buffer + write-back block."""

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
        """Re-check legality, deep-copy, emit rf-block + wb-block, return."""
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
        """Emit the rf-buffer + rf-block + wb-block on the deep-copied ``ir``.

        Nested per-``ko`` form of ``tir.Schedule.rfactor`` for the ``"rmw"``
        recipe (spec §3.1/§3.4). The factored loop ``ko``
        (``option.target_loop_nid``) has extent ``factor``; the matmul output tile
        is partitioned into ``m_tiles`` rows of 128. The existing ``psum``
        accumulator and a NEW SBUF rf-buffer ``B_rf`` both grow their leading
        (partition-tile) dim by ``factor`` so each ``ko`` slot owns ``m_tiles``
        distinct tiles (slot stride ``m_tiles``). The rf-block's init, compute,
        and drain all sit INSIDE the ``ko`` loop, each slot-indexed by ``ko``:

        - rf-init: a per-``ko`` ``memset`` nested as the FIRST child of the ``ko``
          loop (before the ``ki`` nest), zeroing ``psum[ko * m_tiles + m]``.
        - rf-compute: the matmul block flips the K iter_var role to PARALLEL (the
          ``ko`` slot is data-parallel) and its ``dst`` is slot-indexed
          ``psum[ko * m_tiles + m]``; ``ki`` still HW-accumulates per slot.
        - rf-drain: a per-``ko`` ``tensor_copy`` nested as the LAST child of the
          ``ko`` loop (after the ``ki`` nest), copying ``psum[ko * m_tiles + m]``
          to ``B_rf[ko * m_tiles + m]``.
        - wb-init: a NEW ``memset`` zeros the original output tile ``out_sbuf``.
        - wb-combine: a NEW ``tensor_tensor(add)`` block reduces the ``factor``
          ``B_rf`` slots into ``out_sbuf`` over a ``ko`` loop declared
          ACCUMULATION (the carried RAW on ``out_sbuf`` is the closing reduction).

        ``place_buffers`` (LCA) + a rebuilt ``Dependency`` follow, per the
        Transform contract. ``compact_shapes`` is intentionally NOT run: the
        multi-slot rf-buffer must survive (folding it is a downstream step).
        """
        tree = ir.tree
        ko_loop = tree.data(option.target_loop_nid)
        assert isinstance(ko_loop, ForNode)
        factor = ko_loop.extent
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
        rf_name = f"{psum_name}_rf"
        identity = float(op_cls.REDUCE_COMBINATOR.identity)
        combiner = op_cls.REDUCE_COMBINATOR.combiner

        free_extent = ir.buffer(out_name).shape[1]
        self._grow_psum_and_add_rf(ir, psum_name, rf_name, factor)
        self._flip_and_slot_matmul(tree, matmul_block_nid, matmul_leaf, psum_name, ko_var, m_tiles)
        self._nest_memset(tree, option.target_loop_nid, psum_name, ko_var, m_tiles, free_extent, identity)
        self._nest_drain(tree, option.target_loop_nid, psum_name, rf_name, ko_var, m_tiles, free_extent)
        self._insert_writeback(ir, matmul_block_nid, rf_name, out_name, factor, m_tiles, identity, combiner)

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
        beside the matmul (the fused per-tile form). ``tree.descendants`` is an
        unordered set, so picking the first leaf could return the memset (whose
        ``dst`` axes are ``("P", "F")``) and break the matmul-axis lookups in
        ``_partition_tiles`` / ``_flip_and_slot_matmul``. Select the leaf whose op
        declares an ``RFACTOR_RECIPE`` (the matmul) instead.
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

    def _grow_psum_and_add_rf(self, ir: KernelIR, psum_name: str, rf_name: str, factor: int) -> None:
        """Grow ``psum_name``'s leading dim by ``factor`` and add the SBUF rf-buffer.

        Both buffers stack ``factor`` slots along the leading (partition-tile)
        dim, so the logical leading extent becomes ``factor * M`` (physical tile
        dim ``factor * (M // 128)``). The new ``B_rf`` mirrors the grown psum's
        shape but lives in SBUF. The matmul output dtype is the original output's
        dtype; ``place_buffers`` relocates ``B_rf`` to its LCA afterwards.
        """
        tree = ir.tree
        out_buf = ir.buffer(self._drain_out_tensor(tree, psum_name))
        for nid in tree.blocks():
            block = tree.data(nid)
            assert isinstance(block, BlockNode)
            new_allocs = list(block.alloc_buffers)
            changed = False
            for i, buf in enumerate(new_allocs):
                if buf.name == psum_name:
                    leading, free = buf.shape
                    new_allocs[i] = replace(buf, shape=(leading * factor, free))
                    changed = True
            if changed:
                rf_leading, rf_free = out_buf.shape
                rf_buf = Buffer(
                    name=rf_name, shape=(rf_leading * factor, rf_free), dtype=out_buf.dtype, location="sbuf"
                )
                new_allocs.append(rf_buf)
                tree.graph.nodes[nid]["data"] = replace(block, alloc_buffers=tuple(new_allocs))

    def _flip_and_slot_matmul(
        self, tree: KernelTree, block_nid: int, leaf_nid: int, psum_name: str, ko_var: str, m_tiles: int
    ) -> None:
        """Flip the matmul block's K iter_var to PARALLEL and slot-index its psum dst.

        The K iter_var (driving both ``ko`` and ``ki``) becomes PARALLEL — the
        ``ko`` slot is data-parallel over distinct ``psum`` slots; the inner
        ``ki`` accumulate is HW ``+=`` encapsulated in the single matmul leaf, so
        the reduction is not tracked via a carry edge here (it re-emerges in the
        wb-block). The slot prefix ``ko_var * m_tiles`` is added to axis-0 ``lo``
        of every ``psum_name`` region (leaf binding + block reads/writes).
        """
        block = tree.data(block_nid)
        assert isinstance(block, BlockNode)
        op_cls = self._op_cls_of_block(tree, block_nid)
        reduction_abstract = next(a for a, role in op_cls.AXIS_ROLES.items() if role == AxisRole.ACCUMULATION)
        k_axis = block.axis_map[reduction_abstract]
        new_iter_vars = tuple(
            replace(iv, role=AxisRole.PARALLEL) if iv.axis == k_axis else iv for iv in block.iter_vars
        )
        new_reads = tuple(self._slot_region(r, psum_name, ko_var, m_tiles) for r in block.reads)
        new_writes = tuple(self._slot_region(w, psum_name, ko_var, m_tiles) for w in block.writes)
        tree.graph.nodes[block_nid]["data"] = replace(
            block, iter_vars=new_iter_vars, reads=new_reads, writes=new_writes
        )
        leaf = tree.data(leaf_nid)
        assert isinstance(leaf, ISANode)
        new_bindings = {
            slot: self._slot_region(region, psum_name, ko_var, m_tiles)
            for slot, region in leaf.operand_bindings.items()
        }
        tree.graph.nodes[leaf_nid]["data"] = ISANode(
            op_cls=leaf.op_cls, operand_bindings=new_bindings, kwargs=dict(leaf.kwargs)
        )

    def _slot_region(self, region: BufferRegion, target_tensor: str, ko_var: str, m_tiles: int) -> BufferRegion:
        """Prefix the partition-tile slot ``ko_var * m_tiles`` to ``region``'s axis-0 lo.

        Applies only to regions on ``target_tensor`` (the psum accumulator); all
        other regions pass through unchanged. Axis 0 of an SBUF/PSUM region is the
        partition-tile index, so the slot offset stacks before the existing
        per-tile ``lo`` (e.g. ``i_d1_0`` -> ``ko * m_tiles + i_d1_0``).
        """
        if region.tensor != target_tensor:
            return region
        lo0, w0 = region.ranges[0]
        slotted: Expr = Add(left=Mul(left=Var(name=ko_var), right=Const(value=m_tiles)), right=lo0)
        new_ranges = ((slotted, w0), *region.ranges[1:])
        return BufferRegion(tensor=region.tensor, ranges=new_ranges)

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
        """Replace the flat psum-zeroing memset with a per-``ko`` slot-indexed one.

        The canonical flat memset (a sibling block under the root that zeros
        ``psum`` over a full ``m_tiles`` partition sweep) is removed and a NEW
        block is spliced as the FIRST child of the matmul's ``ko`` ForNode. The
        new block declares a ``ko`` iter-binding (axis ``d0``, role PARALLEL to
        match the flipped matmul ``ko``) plus the partition loop ``m``; its dst
        is slot-indexed ``psum[ko * m_tiles + m]`` via :meth:`_slot_region`, so
        each ``ko`` iteration zeros only its own slot before the ``ki`` matmul.
        """
        old_leaf = self._writer_leaf(tree, psum_name, "memset")
        self._remove_flat_block(tree, old_leaf)
        m_var = "i_d1_0"
        region = self._slot_region(self._partition_region(psum_name, m_var, free_extent), psum_name, ko_var, m_tiles)
        block = self._partition_block(ko_var, m_var, m_tiles, free_extent, reads=(), writes=(region,))
        leaf = ISANode(op_cls=NKIMemset, operand_bindings={"dst": region}, kwargs={"value": identity})
        self._splice_block_under_ko(tree, ko_loop_nid, block, m_var, m_tiles, leaf, at_front=True)

    def _nest_drain(
        self,
        tree: KernelTree,
        ko_loop_nid: int,
        psum_name: str,
        rf_name: str,
        ko_var: str,
        m_tiles: int,
        free_extent: int,
    ) -> None:
        """Replace the flat drain ``tensor_copy`` with a per-``ko`` slot-indexed one.

        The canonical flat drain (a sibling block copying ``psum -> out_sbuf``
        over a full ``m_tiles`` sweep) is removed and a NEW block is spliced as
        the LAST child of the matmul's ``ko`` ForNode (after the ``ki`` nest).
        The new block declares a ``ko`` iter-binding (axis ``d0``, role PARALLEL)
        plus the partition loop ``m``; it reads ``psum[ko * m_tiles + m]`` and
        writes the rf-buffer ``B_rf[ko * m_tiles + m]`` (both slot-indexed via
        :meth:`_slot_region`), draining each ``ko`` slot to its own ``B_rf`` slot.
        """
        old_leaf = self._reader_leaf(tree, psum_name, "tensor_copy")
        self._remove_flat_block(tree, old_leaf)
        m_var = "i_d1_0"
        src = self._slot_region(self._partition_region(psum_name, m_var, free_extent), psum_name, ko_var, m_tiles)
        dst = self._slot_region(self._partition_region(rf_name, m_var, free_extent), rf_name, ko_var, m_tiles)
        block = self._partition_block(ko_var, m_var, m_tiles, free_extent, reads=(src,), writes=(dst,))
        leaf = ISANode(op_cls=NKITensorCopy, operand_bindings={"src": src, "dst": dst}, kwargs={})
        self._splice_block_under_ko(tree, ko_loop_nid, block, m_var, m_tiles, leaf, at_front=False)

    def _remove_flat_block(self, tree: KernelTree, leaf_nid: int) -> None:
        """Delete the canonical flat block owning ``leaf_nid`` (block + loop + leaf).

        The flat memset/drain blocks are perfect single-leaf nests under the
        root: a BlockNode, one partition ForNode, one ISA leaf. Detach the block
        from its parent (preserving sibling order) and remove all three nodes.
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

        Axis 0 is the bare partition-tile index ``Var(m_var)`` with width 128;
        the free axis is loopless (full ``free_extent``). The slot prefix is
        added separately by :meth:`_slot_region`.
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
        reads: tuple[BufferRegion, ...],
        writes: tuple[BufferRegion, ...],
    ) -> BlockNode:
        """Build a per-``ko`` memset/drain :class:`BlockNode` nested under ``ko``.

        Declares three iter_vars — ``ko`` (axis ``d0``, role PARALLEL, bound to
        the shared matmul ``ko`` ForNode's ``ko_var``), the partition tile ``m``
        (axis ``d1``, bound to ``m_var``), and the loopless free axis (``d2``).
        Only the ``m`` loop is materialized as a ForNode by the caller; ``ko`` is
        the shared parent loop, so the block reads ``ko_var`` from it. Role
        PARALLEL on ``d0`` matches the flipped matmul ``ko`` and keeps the
        dependency carry analysis from treating this slot as reduction-carried.
        """
        return BlockNode(
            iter_vars=(
                IterVar(axis="d0", dom=(0, m_tiles * PARTITION_DIM), role=AxisRole.PARALLEL),
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

        The new block becomes a child of the matmul's ``ko`` ForNode, carrying
        its own partition loop ``m`` (extent ``m_tiles``) and the single ISA
        ``leaf``. ``at_front`` places it before the ``ki`` sub-nest (the memset);
        otherwise after every existing child (the drain). Sibling order under
        ``ko`` is the dataflow order ``memset -> ki-matmul-nest -> drain``.
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

    def _insert_writeback(
        self,
        ir: KernelIR,
        matmul_block_nid: int,
        rf_name: str,
        out_name: str,
        factor: int,
        m_tiles: int,
        identity: float,
        combiner: str,
    ) -> None:
        """Build + splice the wb-init (memset out) and wb-combine (reduce slots) blocks.

        Inserted as siblings immediately after the matmul (rf-compute) block —
        which owns the nested per-``ko`` rf-init / matmul / rf-drain — under the
        root, preserving dataflow order:
        ``rf-block -> wb-init -> wb-combine -> store``. Anchoring on the matmul
        block (not the now-nested drain block, whose parent is the ``ko`` loop)
        keeps the wb-blocks top-level siblings, not nested inside ``ko``. The
        wb-combine declares the slot loop ACCUMULATION so the carried RAW on
        ``out_name`` is the closing reduction; ``B_rf`` is read at slot
        ``ko * m_tiles + m``.
        """
        tree = ir.tree
        out_buf = ir.buffer(out_name)
        free_extent = out_buf.shape[1]
        parent = tree.parent(matmul_block_nid)
        assert parent is not None
        init_block = self._build_wb_init(tree, out_name, m_tiles, free_extent, identity)
        combine_block = self._build_wb_combine(tree, out_name, rf_name, factor, m_tiles, free_extent, combiner)
        siblings = tree.children(parent)
        insert_at = siblings.index(matmul_block_nid) + 1
        new_order = siblings[:insert_at] + [init_block, combine_block] + siblings[insert_at:]
        for child in siblings:
            tree.graph.remove_edge(parent, child)
        for child in new_order:
            tree.graph.add_edge(parent, child)

    def _build_wb_init(self, tree: KernelTree, out_name: str, m_tiles: int, free_extent: int, identity: float) -> int:
        """Build a sibling block that memsets ``out_name`` to the reducer ``identity``.

        One partition-tile loop ``i_d1_0`` (extent ``m_tiles``), loopless free
        axis (full ``free_extent``). Mirrors the canonical memset sub-block shape.
        The fill value is the op's ``REDUCE_COMBINATOR.identity`` (``0.0`` for the
        matmul sum), since this init dominates the wb-block's closing reduction.
        """
        loop_var = "i_d1_0"
        region = BufferRegion(
            tensor=out_name,
            ranges=((Var(name=loop_var), Const(value=PARTITION_DIM)), (Const(value=0), Const(value=free_extent))),
        )
        block = BlockNode(
            iter_vars=(
                IterVar(axis="d1", dom=(0, m_tiles * PARTITION_DIM), role=AxisRole.PARALLEL),
                IterVar(axis="d2", dom=(0, free_extent), role=AxisRole.PARALLEL),
            ),
            iter_values=(Var(name=loop_var), Const(value=0)),
            reads=(),
            writes=(region,),
            alloc_buffers=(),
            axis_map={"P": "d1", "F": "d2"},
        )
        block_nid = tree.add_node(block, parent=None)
        loop_nid = tree.add_node(ForNode(loop_var=loop_var, extent=m_tiles), parent=block_nid)
        tree.add_node(
            ISANode(op_cls=NKIMemset, operand_bindings={"dst": region}, kwargs={"value": identity}), parent=loop_nid
        )
        return block_nid

    def _build_wb_combine(
        self, tree: KernelTree, out_name: str, rf_name: str, factor: int, m_tiles: int, free_extent: int, combiner: str
    ) -> int:
        """Build a sibling block reducing ``factor`` ``B_rf`` slots into ``out_name``.

        Loop nest ``ko (i_d0_0, extent factor) -> m (i_d1_0, extent m_tiles)``,
        loopless free axis. ``tensor_tensor(data1=out, data2=B_rf[slot], dst=out,
        op=combiner)`` HW-reads-and-writes ``out`` (RMW data1) so the ``ko`` loop —
        declared ACCUMULATION — carries the closing reduction. ``combiner`` is the
        op's ``REDUCE_COMBINATOR.combiner`` (``"add"`` for matmul). ``B_rf`` is read
        at partition slot ``ko * m_tiles + m``.
        """
        ko_var = "i_d0_0"
        m_var = "i_d1_0"
        out_region = BufferRegion(
            tensor=out_name,
            ranges=((Var(name=m_var), Const(value=PARTITION_DIM)), (Const(value=0), Const(value=free_extent))),
        )
        rf_slot: Expr = Add(left=Mul(left=Var(name=ko_var), right=Const(value=m_tiles)), right=Var(name=m_var))
        rf_region = BufferRegion(
            tensor=rf_name, ranges=((rf_slot, Const(value=PARTITION_DIM)), (Const(value=0), Const(value=free_extent)))
        )
        block = BlockNode(
            iter_vars=(
                IterVar(axis="d0", dom=(0, factor), role=AxisRole.ACCUMULATION),
                IterVar(axis="d1", dom=(0, m_tiles * PARTITION_DIM), role=AxisRole.PARALLEL),
                IterVar(axis="d2", dom=(0, free_extent), role=AxisRole.PARALLEL),
            ),
            iter_values=(Var(name=ko_var), Var(name=m_var), Const(value=0)),
            reads=(out_region, rf_region),
            writes=(out_region,),
            alloc_buffers=(),
            axis_map={"K": "d0", "P": "d1", "F": "d2"},
        )
        block_nid = tree.add_node(block, parent=None)
        ko_nid = tree.add_node(ForNode(loop_var=ko_var, extent=factor), parent=block_nid)
        m_nid = tree.add_node(ForNode(loop_var=m_var, extent=m_tiles), parent=ko_nid)
        tree.add_node(
            ISANode(
                op_cls=NKITensorTensor,
                operand_bindings={"data1": out_region, "data2": rf_region, "dst": out_region},
                kwargs={"op": combiner},
            ),
            parent=m_nid,
        )
        return block_nid

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
