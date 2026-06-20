"""Shared fixtures: tuned-matmul IR + M-loop/child discovery for pipeline tests.

Self-contained: the canonical kernel ``f_nkigym``, the tuned ``TRACE``, and
``INPUT_SPECS`` are defined here rather than imported from an example driver, so
the test suite carries no dependency on ``examples/`` code.
"""

from __future__ import annotations

from nkigym.environment import KernelMDP
from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, ForNode, ISANode
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.transforms import (
    ComputeAt,
    ComputeAtOption,
    Fuse,
    Reorder,
    ReorderOption,
    ReverseComputeAt,
    SoftwarePipeline,
    SoftwarePipelineOption,
    Split,
)

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}


@nkigym_kernel
def f_nkigym(lhs_T, rhs):
    """``lhs_T.T @ rhs`` SSA body — the canonical (un-tiled) matmul."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


"""The tuned transform sequence (one ``(transform, option)`` atom per rung) that
the pipeline + body tests replay. Axes ``d0=K``, ``d1=M``, ``d2=N``; literal nids
are stable from the deterministic ``build_initial_ir``. Two Reorders rotate the
matmul nest ``K>M>N`` -> ``M>N>K``, two ComputeAts sink the PSUM memset + drain
under the M loop, and the final SoftwarePipeline double-buffers the accumulator."""
TRACE: list[tuple[object, object]] = [
    (Reorder(), ReorderOption(outer_nid=11, inner_nid=12)),
    (Reorder(), ReorderOption(outer_nid=12, inner_nid=13)),
    (ComputeAt(), ComputeAtOption(block_nid=7, target_loop_nid=11, index=0)),
    (ComputeAt(), ComputeAtOption(block_nid=15, target_loop_nid=11, index=2)),
    (SoftwarePipeline(), SoftwarePipelineOption(loop_nid=11, stages=(0, 0, 1), order=(0, 1, 2))),
]


def tuned_ir() -> KernelIR:
    """Replay ``TRACE`` up to (but excluding) the SoftwarePipeline atom.

    Returns the PRE-pipeline tuned state (M>N>K, sunk memset+drain, single
    PSUM bank) — the input the pipeline tests operate on. ``TRACE`` ends with a
    SoftwarePipeline atom; replaying it here would yield the
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
