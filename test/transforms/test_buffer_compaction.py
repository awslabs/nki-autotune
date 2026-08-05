"""Tests for the BufferCompaction transform and its per-buffer helpers."""

from __future__ import annotations

import copy
from test.transforms._fixtures import build_ladder_state, f_lhs_matmul

from nkigym.ir import KernelIR, build_initial_ir
from nkigym.ir.arith.expr import to_affine
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms import BufferCompaction, BufferCompactionOption, CodeMotion, CodeMotionOption, Split, SplitOption
from nkigym.transforms.base import TransformLegalityError
from nkigym.transforms.code_motion import _move


def _build_m_outer_store_sink_intermediate() -> KernelIR:
    """Build M-OUTER structural store-sink intermediate.

    Builds a topology where both touchers (tensor_copy, store) share an enclosing
    i_d1_0 loop, giving sbuf_prod BOTH M and N compaction anchors. Shape compacts
    to (128, 512). The manual-ladder N-outer case (only N anchored, compacts to
    (2048, 512)) is covered by the byte-exact manual-ladder kernel_14 test.
    """
    """Start from state 12, apply Split store (12->13), then CodeMotion _move (13->14) without the tail."""
    ir12 = build_ladder_state(12)

    """ Helper to find blocks and leaves by op name """

    def blk(ir, op_name):
        return next(
            nid
            for nid in ir.tree.blocks()
            if nid != ir.tree.root
            and any(
                isinstance(ir.tree.data(d), ISANode) and ir.tree.data(d).op_cls.__name__ == op_name
                for d in ir.tree.descendants(nid)
            )
        )

    def leaf(ir, block_nid):
        return next(d for d in ir.tree.descendants(block_nid) if isinstance(ir.tree.data(d), ISANode))

    def tc_loop(ir, var):
        """Find loop var in tensor_copy block."""
        tc = blk(ir, "NKITensorCopy")
        return next(
            d
            for d in ir.tree.descendants(tc)
            if hasattr(ir.tree.data(d), "loop_var") and ir.tree.data(d).loop_var == var
        )

    """ Rung 12->13: Split store """
    ir13 = Split().apply(
        ir12, SplitOption(target_nid=leaf(ir12, blk(ir12, "NKIStore")), factors=(4, 512), target_axis="d2")
    )

    """ Rung 13->14: CodeMotion _move WITHOUT place/compact tail """
    ir13_intermediate = copy.deepcopy(ir13)
    store_blk = blk(ir13_intermediate, "NKIStore")
    d2_loop = tc_loop(ir13_intermediate, "i_d2_0")
    _move(ir13_intermediate, block_nid=store_blk, target_loop_nid=d2_loop, index=-1)
    ir13_intermediate.dependency = Dependency(ir13_intermediate.tree)

    return ir13_intermediate


def _build_transpose_drain_intermediate() -> KernelIR:
    """Move the transpose drain under an inner loop with an outer live axis."""
    specs = {"lhs": ((512, 256), "bfloat16"), "rhs": ((256, 128), "bfloat16")}
    ir = build_initial_ir(f_lhs_matmul, specs)
    drain_leaf = next(
        nid
        for nid in ir.tree.preorder()
        if isinstance(ir.tree.data(nid), ISANode)
        and ir.tree.isa(nid).op_cls.__name__ == "NKITensorCopy"
        and ir.tree.isa(nid).operand_bindings["src"].tensor == "psum_lhs_T"
    )
    transpose_leaf = next(
        nid
        for nid in ir.tree.preorder()
        if isinstance(ir.tree.data(nid), ISANode) and ir.tree.isa(nid).op_cls.__name__ == "NKITranspose"
    )
    drain_block = next(nid for nid in reversed(ir.tree.ancestors(drain_leaf)) if nid in set(ir.tree.blocks()))
    inner_loop = next(
        nid
        for nid in ir.tree.ancestors(transpose_leaf)
        if isinstance(ir.tree.data(nid), ForNode) and ir.tree.loop(nid).loop_var == "i_d1_0"
    )
    return CodeMotion().apply(ir, CodeMotionOption(drain_block, inner_loop, 1))


def _regions_of(ir, tensor):
    """Every (leaf nid, region) pair naming ``tensor`` in ``ir.tree``."""
    out = []
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ISANode):
            for region in data.operand_bindings.values():
                if region.tensor == tensor:
                    out.append((nid, region))
    return out


def _decl_block(ir, tensor):
    """The block nid whose alloc_buffers declares ``tensor``."""
    return next(nid for nid in ir.tree.blocks() if any(b.name == tensor for b in ir.tree.data(nid).alloc_buffers))


def _unselected_buffer_state(ir: KernelIR, selected: str) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Return declarations and operand regions for every non-selected buffer."""
    declarations = tuple(
        (nid, tuple(buffer for buffer in ir.tree.block(nid).alloc_buffers if buffer.name != selected))
        for nid in ir.tree.blocks()
    )
    regions = tuple(
        (
            nid,
            tuple(
                (slot, region)
                for slot, region in ir.tree.isa(nid).operand_bindings.items()
                if region.tensor != selected
            ),
        )
        for nid in ir.tree.preorder()
        if isinstance(ir.tree.data(nid), ISANode)
    )
    return declarations, regions


def test_apply_compacts_sbuf_prod_end_to_end():
    """BufferCompaction atomically places, shrinks, and normalizes ``sbuf_prod``."""
    ir = _build_m_outer_store_sink_intermediate()
    new_ir = BufferCompaction().apply(ir, BufferCompactionOption(tensor="sbuf_prod"))
    assert _decl_block(new_ir, "sbuf_prod") != new_ir.tree.root
    assert new_ir.buffer("sbuf_prod").shape == (128, 512)
    for _nid, region in _regions_of(new_ir, "sbuf_prod"):
        assert to_affine(region.ranges[1][0]).get("i_d2_0", 0) == 0
    """apply must not mutate the input ir (deep-copy contract)."""
    assert ir.buffer("sbuf_prod").shape == (2048, 2048)


def test_apply_does_not_place_or_rewrite_unselected_buffers() -> None:
    """Compacting one tensor leaves other pending compactions untouched."""
    ir = _build_m_outer_store_sink_intermediate()
    before = _unselected_buffer_state(ir, selected="sbuf_lhs_T")
    compacted = BufferCompaction().apply(ir, BufferCompactionOption(tensor="sbuf_lhs_T"))
    assert _unselected_buffer_state(compacted, selected="sbuf_lhs_T") == before
    assert _decl_block(compacted, "sbuf_prod") == ir.tree.root


def test_compaction_does_not_anchor_inside_a_live_outer_loop() -> None:
    """An inner selector cannot alias values retained across an outer loop."""
    ir = _build_transpose_drain_intermediate()
    compacted = BufferCompaction().apply(ir, BufferCompactionOption(tensor="psum_lhs_T"))
    assert compacted.buffer("psum_lhs_T").shape == (256, 512)
    for _nid, region in _regions_of(compacted, "psum_lhs_T"):
        assert to_affine(region.ranges[0][0]).get("i_d1_0", 0) == 1


def test_apply_rejects_shared_hbm():
    """Compacting a shared_hbm buffer is a loud legality error (no tile axis)."""
    ir = _build_m_outer_store_sink_intermediate()
    try:
        BufferCompaction().apply(ir, BufferCompactionOption(tensor="hbm_out"))
        raised = False
    except TransformLegalityError:
        raised = True
    assert raised, "expected TransformLegalityError for shared_hbm"


def test_apply_rejects_noop():
    """Compacting an already-compact buffer (no scope/shape/frame change) is a loud
    legality error — a no-op-returning-success is disallowed."""
    ir = _build_m_outer_store_sink_intermediate()
    once = BufferCompaction().apply(ir, BufferCompactionOption(tensor="sbuf_prod"))
    try:
        BufferCompaction().apply(once, BufferCompactionOption(tensor="sbuf_prod"))
        raised = False
    except TransformLegalityError:
        raised = True
    assert raised, "expected TransformLegalityError for no-op re-compaction"


def test_analyze_offers_uncompacted_buffers():
    """analyze offers sbuf_prod (uncompacted at state 13) and not hbm_out (shared_hbm)."""
    ir = _build_m_outer_store_sink_intermediate()
    tensors = {opt.tensor for opt in BufferCompaction().analyze(ir)}
    assert "sbuf_prod" in tensors
    assert "hbm_out" not in tensors
