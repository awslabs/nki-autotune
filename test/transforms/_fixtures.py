"""Shared canonical-IR fixture for transform tests.

Builds the canonical :class:`KernelIR` for the same matmul described
by ``kernel_transforms.py``: ``lhs_T(K=2048, M=2048).T @ rhs(K, N=2048)``.
"""

from __future__ import annotations

from nkigym.ir import KernelIR, build_initial_ir
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}


@nkigym_kernel
def f_matmul(lhs_T, rhs):
    """``lhs_T.T @ rhs`` — load, matmul, drain, store (SSA)."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


def build_canonical_ir() -> KernelIR:
    """Build the canonical :class:`KernelIR` for the matmul fixture."""
    return build_initial_ir(f_matmul, INPUT_SPECS)


def _ladder_helpers():
    """Return (blk, leaf, loop, inner) target-locators bound to a fresh closure."""
    from nkigym.ir.tree import ForNode, ISANode

    def blk(ir, op_name, which=0):
        found = [
            nid
            for nid in ir.tree.blocks()
            if nid != ir.tree.root
            and sum(1 for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)) == 1
            and ir.tree.data(
                next(d for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode))
            ).op_cls.__name__
            == op_name
        ]
        return found[which]

    def leaf(ir, block_nid):
        return next(d for d in ir.tree.preorder(block_nid) if isinstance(ir.tree.data(d), ISANode))

    def loop(ir, block_nid, loop_var):
        return next(
            d
            for d in ir.tree.preorder(block_nid)
            if isinstance(ir.tree.data(d), ForNode) and ir.tree.data(d).loop_var == loop_var
        )

    def inner(ir, block_nid):
        return ir.tree.ancestors(leaf(ir, block_nid))[-1]

    def mm_loop(ir, loop_var):
        """Return the ForNode with ``loop_var`` enclosing the (unique) matmul leaf.

        Once a producer is sunk into the matmul's block its ``isa_count`` rises
        above one, so ``blk(ir, "NKIMatmul")`` no longer finds it; this locator
        keys off the matmul leaf directly.
        """
        mm_leaf = next(
            nid
            for nid in ir.tree.preorder()
            if isinstance(ir.tree.data(nid), ISANode) and ir.tree.data(nid).op_cls.__name__ == "NKIMatmul"
        )
        return next(
            a
            for a in ir.tree.ancestors(mm_leaf)
            if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == loop_var
        )

    def tc_loop(ir, loop_var):
        """Return the ForNode with ``loop_var`` enclosing the (unique) tensor_copy leaf."""
        tc_leaf = next(
            nid
            for nid in ir.tree.preorder()
            if isinstance(ir.tree.data(nid), ISANode) and ir.tree.data(nid).op_cls.__name__ == "NKITensorCopy"
        )
        return next(
            a
            for a in ir.tree.ancestors(tc_leaf)
            if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == loop_var
        )

    return blk, leaf, loop, inner, mm_loop, tc_loop


def build_ladder_state(n: int) -> KernelIR:
    """Replay the kernel_transforms.py transform sequence from canonical to kernel_n.

    Rungs appended one at a time. Each lambda takes ir, applies one transform,
    returns the new ir. Raises NotImplementedError for unwired rungs (loud).
    """
    from nkigym.ir.tree import ISANode
    from nkigym.transforms import (  # noqa: F401
        ComputeAt,
        ComputeAtOption,
        Reorder,
        ReorderOption,
        ReverseComputeAt,
        ReverseComputeAtOption,
        Split,
        SplitOption,
    )

    blk, leaf, loop, inner, mm_loop, tc_loop = _ladder_helpers()

    def load_blk(ir, tensor):
        """Return the single-leaf load block whose ISA ``src`` reads ``tensor``.

        Locating loads by their source tensor (not positional ``which``) keeps
        each rung robust to the tree-order shifts a sink introduces.
        """
        return next(
            nid
            for nid in ir.tree.blocks()
            if nid != ir.tree.root
            and sum(1 for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)) == 1
            and ir.tree.data(leaf(ir, nid)).op_cls.__name__ == "NKILoad"
            and ir.tree.data(leaf(ir, nid)).operand_bindings["src"].tensor == tensor
        )

    rungs: list = []
    """--- rungs appended below ---"""

    def rung_0_1(ir):
        """0->1: Split load_lhsT leaf d1 -> (16, 128)."""
        load = load_blk(ir, "lhs_T")
        return Split().apply(ir, SplitOption(target_nid=leaf(ir, load), factors=(16, 128), target_axis="d1"))

    rungs.append(rung_0_1)

    def rung_1_2(ir):
        """1->2: ComputeAt load_lhsT block under matmul d1 loop (cover d0,d1)."""
        load = load_blk(ir, "lhs_T")
        d1 = mm_loop(ir, "i_d1_0")
        return ComputeAt().apply(ir, ComputeAtOption(block_nid=load, target_loop_nid=d1, index=-2))

    rungs.append(rung_1_2)

    def rung_2_3(ir):
        """2->3: Split load_rhs leaf d2 -> (4, 512)."""
        rhs = load_blk(ir, "rhs")
        return Split().apply(ir, SplitOption(target_nid=leaf(ir, rhs), factors=(4, 512), target_axis="d2"))

    rungs.append(rung_2_3)

    def rung_3_4(ir):
        """3->4: ComputeAt load_rhs block under matmul d2 loop (cover d0,d2)."""
        rhs = load_blk(ir, "rhs")
        d2 = mm_loop(ir, "i_d2_0")
        return ComputeAt().apply(ir, ComputeAtOption(block_nid=rhs, target_loop_nid=d2, index=0))

    rungs.append(rung_3_4)

    def rung_4_5(ir):
        """4->5: Split memset leaf d2 -> (4, 512)."""
        ms = blk(ir, "NKIMemset")
        return Split().apply(ir, SplitOption(target_nid=leaf(ir, ms), factors=(4, 512), target_axis="d2"))

    rungs.append(rung_4_5)

    def rung_5_6(ir):
        """5->6: Reorder matmul outer pair d0<->d1 (adjacent swap)."""
        d0 = mm_loop(ir, "i_d0_0")
        d1 = mm_loop(ir, "i_d1_0")
        return Reorder().apply(ir, ReorderOption(outer_nid=d0, inner_nid=d1))

    rungs.append(rung_5_6)

    def rung_6_7(ir):
        """6->7: ComputeAt memset block under matmul d1 loop (cover d1; d2 residual)."""
        ms = blk(ir, "NKIMemset")
        d1 = mm_loop(ir, "i_d1_0")
        return ComputeAt().apply(ir, ComputeAtOption(block_nid=ms, target_loop_nid=d1, index=0))

    rungs.append(rung_6_7)

    def rung_7_8(ir):
        """7->8: ComputeAt x2 — sink load_lhsT then load_rhs under matmul inner d2 loop."""
        lhs = load_blk(ir, "lhs_T")
        ir = ComputeAt().apply(ir, ComputeAtOption(block_nid=lhs, target_loop_nid=mm_loop(ir, "i_d2_0"), index=0))
        rhs = load_blk(ir, "rhs")
        ir = ComputeAt().apply(ir, ComputeAtOption(block_nid=rhs, target_loop_nid=mm_loop(ir, "i_d2_0"), index=1))
        return ir

    rungs.append(rung_7_8)

    def rung_8_9(ir):
        """8->9: Reorder matmul d0<->d2 (adjacent swap)."""
        d0 = mm_loop(ir, "i_d0_0")
        d2 = mm_loop(ir, "i_d2_0")
        return Reorder().apply(ir, ReorderOption(outer_nid=d0, inner_nid=d2))

    rungs.append(rung_8_9)

    def rung_9_10(ir):
        """9->10: ComputeAt memset block under matmul d2 loop (cover d2)."""
        ms = blk(ir, "NKIMemset")
        d2 = mm_loop(ir, "i_d2_0")
        return ComputeAt().apply(ir, ComputeAtOption(block_nid=ms, target_loop_nid=d2, index=0))

    rungs.append(rung_9_10)

    def rung_10_11(ir):
        """10->11: Split tensor_copy leaf d2 -> (4, 512)."""
        tc = blk(ir, "NKITensorCopy")
        return Split().apply(ir, SplitOption(target_nid=leaf(ir, tc), factors=(4, 512), target_axis="d2"))

    rungs.append(rung_10_11)

    def rung_11_12(ir):
        """11->12: ReverseComputeAt tensor_copy block under matmul d2 loop (PSUM hoist)."""
        tc = blk(ir, "NKITensorCopy")
        d2 = mm_loop(ir, "i_d2_0")
        return ReverseComputeAt().apply(ir, ReverseComputeAtOption(block_nid=tc, target_loop_nid=d2, index=-1))

    rungs.append(rung_11_12)

    def rung_12_13(ir):
        """12->13: Split store leaf d2 -> (4, 512)."""
        st = blk(ir, "NKIStore")
        return Split().apply(ir, SplitOption(target_nid=leaf(ir, st), factors=(4, 512), target_axis="d2"))

    rungs.append(rung_12_13)

    def rung_13_14(ir):
        """13->14: ReverseComputeAt store block under tensor_copy d2 loop."""
        st = blk(ir, "NKIStore")
        d2 = tc_loop(ir, "i_d2_0")
        return ReverseComputeAt().apply(ir, ReverseComputeAtOption(block_nid=st, target_loop_nid=d2, index=-1))

    rungs.append(rung_13_14)
    if n > len(rungs):
        raise NotImplementedError(f"build_ladder_state({n}): only {len(rungs)} rungs wired")
    ir = build_canonical_ir()
    for rung in rungs[:n]:
        ir = rung(ir)
    return ir
