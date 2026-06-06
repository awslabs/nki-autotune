"""Shared fixtures: tuned-matmul IR + M-loop/child discovery for pipeline tests."""

from __future__ import annotations

from examples.tune_matmul_lhsT_rhs import INPUT_SPECS, TRACE, f_nkigym

from nkigym.environment import KernelMDP
from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, ForNode, ISANode
from nkigym.transforms import ComputeAt, Fuse, Reorder, ReverseComputeAt, SoftwarePipeline, Split


def tuned_ir() -> KernelIR:
    """Replay the example TRACE up to (but excluding) the SoftwarePipeline atom.

    Returns the PRE-pipeline tuned state (M>N>K, sunk memset+drain, single
    PSUM bank) — the input the pipeline tests operate on. The example's TRACE
    now ends with a SoftwarePipeline atom; replaying it here would yield the
    already-pipelined state (and ``analyze`` would correctly find nothing to
    do). Skipping SoftwarePipeline atoms keeps this fixture the un-pipelined
    state regardless of future TRACE additions.
    """
    env = KernelMDP(f_nkigym, INPUT_SPECS, transforms=[Split(), Fuse(), Reorder(), ComputeAt(), ReverseComputeAt()])
    state = env.reset()
    for transform, option in TRACE:
        if isinstance(transform, SoftwarePipeline):
            break
        state = env.step(state, (transform, option))
    return state


def m_loop_and_children(ir: KernelIR) -> tuple[int, list[int]]:
    """Return the i_d1_0 ForNode enclosing the matmul leaf and ITS DIRECT CHILDREN in order.

    The children are ALL direct children of the loop (BlockNode or ForNode),
    the stageable units — NOT BlockNode-filtered. The tuned M-loop body is
    ``[memset-block, matmul-loopnest(ForNode), drain-block]`` = 3 units; the
    matmul is a nested loop subtree, not a sibling block. This matches TVM's
    one-stage-per-SeqStmt-child model that SoftwarePipeline enumerates over.
    """
    mm_leaf = next(
        n
        for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKIMatmul"
    )
    m_loop = next(
        a
        for a in ir.tree.ancestors(mm_leaf)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d1_0"
    )
    children = list(ir.tree.children(m_loop))
    return m_loop, children


def parent_block_of(ir: KernelIR, loop_nid: int) -> int:
    """Return the nearest enclosing BlockNode of a loop (for writing the annotation in tests)."""
    result = ir.tree.root
    for anc in reversed(ir.tree.ancestors(loop_nid)):
        if isinstance(ir.tree.data(anc), BlockNode):
            result = anc
            break
    return result
