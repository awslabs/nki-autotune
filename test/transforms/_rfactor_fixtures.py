"""Shared fixture: canonical matmul IR with K split into (ko, ki) — the RFactor input."""

from __future__ import annotations

from nkigym.ir import KernelIR
from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms import Split, SplitOption
from test.transforms._fixtures import build_canonical_ir


def matmul_leaf_nid(ir: KernelIR) -> int:
    """Return the nc_matmul ISANode nid."""
    return next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKIMatmul"
    )


def _matmul_k_loop_nid(ir: KernelIR) -> int:
    """Return the ForNode binding the matmul's K loop (the 'i_d0_0' that ENCLOSES
    the matmul leaf — NOT a load's same-named loop).
    """
    matmul = matmul_leaf_nid(ir)
    return next(
        a
        for a in ir.tree.ancestors(matmul)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d0_0"
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
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var.startswith("i_d0_")
    ]
    return k_loops[0]
