"""Test-only transform recipe for the canonical ``lhs_T.T @ rhs`` workload.

The example surface contains only runnable demonstrations. This fixture keeps
the transform-driven reconstruction used by dependency and byte-exact tests.
"""

from __future__ import annotations

from nkigym.ir import KernelIR, build_initial_ir
from nkigym.ir.tree import ForNode, ISANode
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.transforms import (
    BufferCompaction,
    BufferCompactionOption,
    BufferLayout,
    BufferLayoutOption,
    CodeMotion,
    CodeMotionOption,
    Reorder,
    ReorderOption,
    RFactor,
    RFactorOption,
    Split,
    SplitOption,
)

INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {
    "lhs_T": ((2048, 2048), "bfloat16"),
    "rhs": ((2048, 2048), "bfloat16"),
}


@nkigym_kernel
def f_matmul(lhs_T, rhs):
    """Return the canonical SSA graph for ``lhs_T.T @ rhs``."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


def _isa_descendants(ir: KernelIR, nid: int) -> list[ISANode]:
    """Return the ISA nodes below ``nid``."""
    leaves: list[ISANode] = []
    for descendant in ir.tree.descendants(nid):
        node = ir.tree.data(descendant)
        if isinstance(node, ISANode):
            leaves.append(node)
    return leaves


def block_for_op(ir: KernelIR, op_name: str) -> int:
    """Return the single-leaf block containing the named operation."""
    for nid in ir.tree.blocks():
        leaves = _isa_descendants(ir, nid)
        if len(leaves) == 1 and leaves[0].op_cls.__name__ == op_name:
            return nid
    raise AssertionError(f"no leaf block for {op_name}")


def leaf_for_op(ir: KernelIR, op_name: str) -> int:
    """Return the sole ISA leaf for an operation name."""
    leaves: list[int] = []
    for nid in ir.tree.preorder():
        node = ir.tree.data(nid)
        if isinstance(node, ISANode) and node.op_cls.__name__ == op_name:
            leaves.append(nid)
    if len(leaves) != 1:
        raise AssertionError(f"expected one {op_name} leaf, found {leaves}")
    return leaves[0]


def load_block_reading(ir: KernelIR, tensor: str) -> int:
    """Return the single-leaf load block reading the named tensor."""
    for nid in ir.tree.blocks():
        leaves = _isa_descendants(ir, nid)
        if (
            len(leaves) == 1
            and leaves[0].op_cls.__name__ == "NKILoad"
            and leaves[0].operand_bindings["src"].tensor == tensor
        ):
            return nid
    raise AssertionError(f"no single-leaf load block reading {tensor}")


def matmul_loop(ir: KernelIR, loop_var: str) -> int:
    """Return the named loop enclosing the matmul leaf."""
    leaf = leaf_for_op(ir, "NKIMatmul")
    for ancestor in ir.tree.ancestors(leaf):
        node = ir.tree.data(ancestor)
        if isinstance(node, ForNode) and node.loop_var == loop_var:
            return ancestor
    raise AssertionError(f"no matmul loop named {loop_var}")


def _load_leaf(ir: KernelIR, tensor: str) -> int:
    """Return the load ISA leaf reading ``tensor``."""
    return next(
        nid for nid in ir.tree.descendants(load_block_reading(ir, tensor)) if isinstance(ir.tree.data(nid), ISANode)
    )


def _load_for(ir: KernelIR, tensor: str, loop_var: str) -> int:
    """Return a named loop inside the load block reading ``tensor``."""
    return next(
        nid
        for nid in ir.tree.descendants(load_block_reading(ir, tensor))
        if isinstance(ir.tree.data(nid), ForNode) and ir.tree.loop(nid).loop_var == loop_var
    )


def _psum_memset_leaf(ir: KernelIR) -> int:
    """Return the memset leaf writing the PSUM accumulator."""
    return next(
        nid
        for nid in ir.tree.preorder()
        if isinstance(ir.tree.data(nid), ISANode)
        and ir.tree.isa(nid).op_cls.NAME == "memset"
        and ir.tree.isa(nid).operand_bindings["dst"].tensor.startswith("psum")
    )


def _psum_memset_blk(ir: KernelIR) -> int:
    """Return the single-leaf block owning the PSUM memset."""
    leaf = _psum_memset_leaf(ir)
    return next(
        nid
        for nid in ir.tree.blocks()
        if nid != ir.tree.root
        and leaf in ir.tree.descendants(nid)
        and not any(isinstance(ir.tree.data(child), ISANode) for child in ir.tree.descendants(nid) if child != leaf)
    )


def _blk_loop(ir: KernelIR, blk_nid: int, loop_var: str) -> int:
    """Return a named loop within ``blk_nid``."""
    return next(
        nid
        for nid in ir.tree.descendants(blk_nid)
        if isinstance(ir.tree.data(nid), ForNode) and ir.tree.loop(nid).loop_var == loop_var
    )


def _reorder_blk_to_nm(ir: KernelIR, blk_nid: int) -> KernelIR:
    """Reorder one block's M/N loops to N/M order."""
    return Reorder().apply(
        ir, ReorderOption(outer_nid=_blk_loop(ir, blk_nid, "i_d1_0"), inner_nid=_blk_loop(ir, blk_nid, "i_d2_0"))
    )


def _build_ladder() -> list[tuple[str, KernelIR]]:
    """Build the 36 states matching manual ladder kernels 0 through 35."""
    steps = [
        lambda ir: Reorder().apply(
            ir, ReorderOption(outer_nid=matmul_loop(ir, "i_d1_0"), inner_nid=matmul_loop(ir, "i_d2_0"))
        ),
        lambda ir: Reorder().apply(
            ir, ReorderOption(outer_nid=matmul_loop(ir, "i_d0_0"), inner_nid=matmul_loop(ir, "i_d2_0"))
        ),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=matmul_loop(ir, "i_d0_0"), factors=(2, 8), target_axis=None)
        ),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=matmul_loop(ir, "i_d1_0"), factors=(4, 4), target_axis=None)
        ),
        lambda ir: Reorder().apply(
            ir, ReorderOption(outer_nid=matmul_loop(ir, "i_d0_1"), inner_nid=matmul_loop(ir, "i_d1_0"))
        ),
        lambda ir: Reorder().apply(
            ir, ReorderOption(outer_nid=matmul_loop(ir, "i_d0_1"), inner_nid=matmul_loop(ir, "i_d1_1"))
        ),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=16)),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=leaf_for_op(ir, "NKITensorCopy"), factors=(4, 512), target_axis="d2")
        ),
        lambda ir: _reorder_blk_to_nm(ir, block_for_op(ir, "NKITensorCopy")),
        lambda ir: CodeMotion().apply(
            ir,
            CodeMotionOption(
                block_nid=block_for_op(ir, "NKITensorCopy"), target_loop_nid=matmul_loop(ir, "i_d2_0"), index=-1
            ),
        ),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=leaf_for_op(ir, "NKIStore"), factors=(4, 512), target_axis="d2")
        ),
        lambda ir: _reorder_blk_to_nm(ir, block_for_op(ir, "NKIStore")),
        lambda ir: CodeMotion().apply(
            ir,
            CodeMotionOption(
                block_nid=block_for_op(ir, "NKIStore"), target_loop_nid=matmul_loop(ir, "i_d2_0"), index=-1
            ),
        ),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption(tensor="sbuf_prod")),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption(tensor="sbuf_prod", list_len=16)),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_psum_memset_leaf(ir), factors=(4, 512), target_axis="d2")),
        lambda ir: _reorder_blk_to_nm(ir, _psum_memset_blk(ir)),
        lambda ir: CodeMotion().apply(
            ir, CodeMotionOption(block_nid=_psum_memset_blk(ir), target_loop_nid=matmul_loop(ir, "i_d2_0"), index=0)
        ),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption(tensor="psum_prod")),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_load_for(ir, "rhs", "i_d0_0"), factors=(2, 8), target_axis=None)
        ),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_load_leaf(ir, "rhs"), factors=(4, 512), target_axis="d2")),
        lambda ir: Reorder().apply(
            ir, ReorderOption(outer_nid=_load_for(ir, "rhs", "i_d0_1"), inner_nid=_load_for(ir, "rhs", "i_d2_0"))
        ),
        lambda ir: Reorder().apply(
            ir, ReorderOption(outer_nid=_load_for(ir, "rhs", "i_d0_0"), inner_nid=_load_for(ir, "rhs", "i_d2_0"))
        ),
        lambda ir: CodeMotion().apply(
            ir,
            CodeMotionOption(
                block_nid=load_block_reading(ir, "rhs"), target_loop_nid=matmul_loop(ir, "i_d0_0"), index=0
            ),
        ),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption(tensor="sbuf_rhs")),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption(tensor="sbuf_rhs", list_len=8)),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_load_leaf(ir, "lhs_T"), factors=(4, 512), target_axis="d1")
        ),
        lambda ir: Split().apply(
            ir, SplitOption(target_nid=_load_for(ir, "lhs_T", "i_d0_0"), factors=(2, 8), target_axis=None)
        ),
        lambda ir: Reorder().apply(
            ir, ReorderOption(outer_nid=_load_for(ir, "lhs_T", "i_d0_1"), inner_nid=_load_for(ir, "lhs_T", "i_d1_0"))
        ),
        lambda ir: CodeMotion().apply(
            ir,
            CodeMotionOption(
                block_nid=load_block_reading(ir, "lhs_T"), target_loop_nid=matmul_loop(ir, "i_d1_0"), index=0
            ),
        ),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption(tensor="sbuf_lhs_T")),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption(tensor="sbuf_lhs_T", list_len=8)),
        lambda ir: RFactor().apply(ir, RFactorOption(target_loop_nid=matmul_loop(ir, "i_d0_0"), factor_axis=0)),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption(tensor="psum_prod")),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption(tensor="sbuf_rfactor")),
    ]
    ir = build_initial_ir(f_matmul, INPUT_SPECS)
    ladder = [("kernel_0", ir)]
    for step in steps:
        ir = step(ir)
        ladder.append((f"kernel_{len(ladder)}", ir))
    return ladder
