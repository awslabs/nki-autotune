"""Shared fixture: canonical matmul IR with K split into (ko, ki) — the RFactor input."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.ir import KernelIR
from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms import Split, SplitOption


def matmul_leaf_nid(ir: KernelIR) -> int:
    """Return the nc_matmul ISANode nid."""
    return next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.isa(n).op_cls.__name__ == "NKIMatmul"
    )


def _matmul_k_loop_nid(ir: KernelIR) -> int:
    """Return the ForNode binding the matmul's K loop (the 'i_d0_0' that ENCLOSES
    the matmul leaf — NOT a load's same-named loop).
    """
    matmul = matmul_leaf_nid(ir)
    return next(
        a
        for a in ir.tree.ancestors(matmul)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.loop(a).loop_var == "i_d0_0"
    )


def split_k_ir() -> KernelIR:
    """Canonical matmul IR after Split(matmul K -> ko=2, ki=8). One PSUM accumulator.

    The matmul K loop ('i_d0_0', 16 trips) is split outer-trip into ko=2 over
    ki=8. Both resulting loops bind the matmul's K axis (ACCUMULATION).
    """
    ir = build_canonical_ir()
    k_loop = _matmul_k_loop_nid(ir)
    return Split().apply(ir, SplitOption(target_nid=k_loop, factors=(2, 8), target_axis=None))


def ko_loop_nid(ir: KernelIR) -> int:
    """Return the OUTER K loop (ko) ForNode nid in a post-Split IR.

    After Split, two K loops enclose the matmul; ko is the OUTER (first among the
    matmul's K-axis ForNodes in ancestor order, root-first). Used as
    RFactorOption.target_loop_nid.
    """
    matmul = matmul_leaf_nid(ir)
    k_loops = [
        a
        for a in ir.tree.ancestors(matmul)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.loop(a).loop_var.startswith("i_d0_")
    ]
    return k_loops[0]


def _mm_m_loop(ir: KernelIR) -> int:
    """nid of the matmul-enclosing M loop (i_d1_0), not a load's same-named loop."""
    mm = matmul_leaf_nid(ir)
    return next(
        a
        for a in ir.tree.ancestors(mm)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.loop(a).loop_var == "i_d1_0"
    )


def mid_ladder_ir() -> KernelIR:
    """Mid state: Split(K -> ko=2, ki=8) then Split(M -> 4, 4); buffers stay packed.

    K split (ko/ki) + M tiled (i_d1_0 x i_d1_1), every buffer still a packed ndarray,
    no load sunk — isolates RFactor's role-location + geometry from the list-buffer
    dimension. Pinned by ``test_rfactor.py::test_apply_sim_matches_matmul_mid_tiled_m``.
    """
    ir = split_k_ir()
    return Split().apply(ir, SplitOption(target_nid=_mm_m_loop(ir), factors=(4, 4), target_axis=None))


def k32_ir() -> KernelIR:
    """The pre-RFactor endpoint (manual k32): fully-tiled, all-list-buffer state, matmul
    nest N > ko > Mo > Mi > ki. Built by the shipped driven ladder in kernel_transforms,
    so this fixture tracks that ladder exactly (the RFactor input the rewrite targets).

    Selected by NAME (``kernel_32``), not position: the ladder ends with the
    RFactor + two BufferCompaction rungs (``kernel_35``), so ``[-1]`` returns
    the fully compacted state. The named lookup is stable against further
    ladder appends.
    """
    from examples.kernel_transforms import _build_ladder

    return next(ir for name, ir in _build_ladder() if name == "kernel_32")


def k32_ko_loop_nid(ir: KernelIR) -> int:
    """The OUTER matmul K loop (ko) nid in a k32-shaped IR: the first matmul-enclosing
    i_d0_* ForNode (root-first ancestor order)."""
    matmul = matmul_leaf_nid(ir)
    return next(
        a
        for a in ir.tree.ancestors(matmul)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.loop(a).loop_var.startswith("i_d0_")
    )
