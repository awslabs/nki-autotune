"""Test-only transform ladder for the canonical ``lhs @ rhs`` workload."""

from __future__ import annotations

from collections.abc import Callable
from test.transforms._fixtures import LHS_INPUT_SPECS, f_lhs_matmul

from nkigym.ir import KernelIR, build_initial_ir
from nkigym.ir.tree import ForNode, ISANode
from nkigym.transforms import (
    BufferCompaction,
    BufferCompactionOption,
    BufferLayout,
    BufferLayoutOption,
    CodeMotion,
    CodeMotionOption,
    LoadTranspose,
    Reorder,
    ReorderOption,
    RFactor,
    RFactorOption,
    Split,
    SplitOption,
)

Step = Callable[[KernelIR], KernelIR]


def _op_leaves(ir: KernelIR, op_name: str) -> list[int]:
    """Return ISA leaves whose class name is ``op_name``."""
    return [
        nid
        for nid in ir.tree.preorder()
        if isinstance(ir.tree.data(nid), ISANode) and ir.tree.isa(nid).op_cls.__name__ == op_name
    ]


def _leaf_with_tensor(ir: KernelIR, op_name: str, slot: str, tensor: str) -> int:
    """Return the unique operation leaf binding ``slot`` to ``tensor``."""
    matches = [
        nid
        for nid in _op_leaves(ir, op_name)
        if slot in ir.tree.isa(nid).operand_bindings and ir.tree.isa(nid).operand_bindings[slot].tensor == tensor
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {op_name} with {slot}={tensor}, found {matches}")
    return matches[0]


def _block_for_leaf(ir: KernelIR, leaf_nid: int) -> int:
    """Return the closest block containing ``leaf_nid``."""
    blocks = set(ir.tree.blocks())
    return next(nid for nid in reversed(ir.tree.ancestors(leaf_nid)) if nid in blocks)


def _block_loop(ir: KernelIR, block_nid: int, loop_var: str) -> int:
    """Return one loop named ``loop_var`` within ``block_nid``."""
    matches = [
        nid
        for nid in ir.tree.descendants(block_nid)
        if isinstance(ir.tree.data(nid), ForNode) and ir.tree.loop(nid).loop_var == loop_var
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {loop_var} in block {block_nid}, found {matches}")
    return matches[0]


def _matmul_leaf(ir: KernelIR) -> int:
    """Return the sole matmul ISA leaf."""
    leaves = _op_leaves(ir, "NKIMatmul")
    if len(leaves) != 1:
        raise ValueError(f"expected one NKIMatmul, found {leaves}")
    return leaves[0]


def _matmul_loop(ir: KernelIR, loop_var: str) -> int:
    """Return a named loop enclosing the matmul leaf."""
    matches = [
        nid
        for nid in ir.tree.ancestors(_matmul_leaf(ir))
        if isinstance(ir.tree.data(nid), ForNode) and ir.tree.loop(nid).loop_var == loop_var
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one matmul loop {loop_var}, found {matches}")
    return matches[0]


def _load_leaf(ir: KernelIR, source: str) -> int:
    """Return the load reading ``source``."""
    return _leaf_with_tensor(ir, "NKILoad", "src", source)


def _load_block(ir: KernelIR, source: str) -> int:
    """Return the block containing the load of ``source``."""
    return _block_for_leaf(ir, _load_leaf(ir, source))


def _load_loop(ir: KernelIR, source: str, loop_var: str) -> int:
    """Return a named loop in the load block for ``source``."""
    return _block_loop(ir, _load_block(ir, source), loop_var)


def _copy_leaf(ir: KernelIR, source: str) -> int:
    """Return the tensor-copy drain reading ``source``."""
    return _leaf_with_tensor(ir, "NKITensorCopy", "src", source)


def _copy_block(ir: KernelIR, source: str) -> int:
    """Return the tensor-copy block reading ``source``."""
    return _block_for_leaf(ir, _copy_leaf(ir, source))


def _memset_leaf(ir: KernelIR, target: str) -> int:
    """Return the memset writing ``target``."""
    return _leaf_with_tensor(ir, "NKIMemset", "dst", target)


def _memset_block(ir: KernelIR, target: str) -> int:
    """Return the memset block writing ``target``."""
    return _block_for_leaf(ir, _memset_leaf(ir, target))


def _store_leaf(ir: KernelIR) -> int:
    """Return the sole output store."""
    leaves = _op_leaves(ir, "NKIStore")
    if len(leaves) != 1:
        raise ValueError(f"expected one NKIStore, found {leaves}")
    return leaves[0]


def _store_block(ir: KernelIR) -> int:
    """Return the output store block."""
    return _block_for_leaf(ir, _store_leaf(ir))


def _reorder_output_block(ir: KernelIR, block_nid: int) -> KernelIR:
    """Change one output block from M/N to N/M order."""
    return Reorder().apply(
        ir,
        ReorderOption(outer_nid=_block_loop(ir, block_nid, "i_d0_0"), inner_nid=_block_loop(ir, block_nid, "i_d2_0")),
    )


def _matmul_steps() -> list[Step]:
    """Return the transform recipe shared by the DMA-transpose ladder."""
    return [
        lambda ir: Reorder().apply(ir, ReorderOption(_matmul_loop(ir, "i_d0_0"), _matmul_loop(ir, "i_d2_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(_matmul_loop(ir, "i_d1_0"), _matmul_loop(ir, "i_d2_0"))),
        lambda ir: Split().apply(ir, SplitOption(_matmul_loop(ir, "i_d1_0"), (2, 8), None)),
        lambda ir: Split().apply(ir, SplitOption(_matmul_loop(ir, "i_d0_0"), (4, 4), None)),
        lambda ir: Reorder().apply(ir, ReorderOption(_matmul_loop(ir, "i_d1_1"), _matmul_loop(ir, "i_d0_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(_matmul_loop(ir, "i_d1_1"), _matmul_loop(ir, "i_d0_1"))),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption("psum_prod", 16)),
        lambda ir: Split().apply(ir, SplitOption(_copy_leaf(ir, "psum_prod"), (4, 512), "d2")),
        lambda ir: _reorder_output_block(ir, _copy_block(ir, "psum_prod")),
        lambda ir: CodeMotion().apply(
            ir, CodeMotionOption(_copy_block(ir, "psum_prod"), _matmul_loop(ir, "i_d2_0"), -1)
        ),
        lambda ir: Split().apply(ir, SplitOption(_store_leaf(ir), (4, 512), "d2")),
        lambda ir: _reorder_output_block(ir, _store_block(ir)),
        lambda ir: CodeMotion().apply(ir, CodeMotionOption(_store_block(ir), _matmul_loop(ir, "i_d2_0"), -1)),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption("sbuf_prod")),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption("sbuf_prod", 16)),
        lambda ir: Split().apply(ir, SplitOption(_memset_leaf(ir, "psum_prod"), (4, 512), "d2")),
        lambda ir: _reorder_output_block(ir, _memset_block(ir, "psum_prod")),
        lambda ir: CodeMotion().apply(
            ir, CodeMotionOption(_memset_block(ir, "psum_prod"), _matmul_loop(ir, "i_d2_0"), 0)
        ),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption("psum_prod")),
        lambda ir: Split().apply(ir, SplitOption(_load_loop(ir, "rhs", "i_d1_0"), (2, 8), None)),
        lambda ir: Split().apply(ir, SplitOption(_load_leaf(ir, "rhs"), (4, 512), "d2")),
        lambda ir: Reorder().apply(ir, ReorderOption(_load_loop(ir, "rhs", "i_d1_1"), _load_loop(ir, "rhs", "i_d2_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(_load_loop(ir, "rhs", "i_d1_0"), _load_loop(ir, "rhs", "i_d2_0"))),
        lambda ir: CodeMotion().apply(ir, CodeMotionOption(_load_block(ir, "rhs"), _matmul_loop(ir, "i_d1_0"), 0)),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption("sbuf_rhs")),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption("sbuf_rhs", 8)),
        lambda ir: RFactor().apply(ir, RFactorOption(_matmul_loop(ir, "i_d1_0"), 0)),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption("psum_prod")),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption("sbuf_rfactor")),
    ]


def _build_ladder() -> list[tuple[str, KernelIR]]:
    """Build the 32 states matching manual ladder kernels 0 through 31."""
    ir = build_initial_ir(f_lhs_matmul, LHS_INPUT_SPECS)
    ladder = [("kernel_0", ir)]

    load_transpose = LoadTranspose()
    options = load_transpose.analyze(ir)
    if len(options) != 1:
        raise ValueError(f"expected one canonical LoadTranspose option, found {options}")
    ir = load_transpose.apply(ir, options[0])
    ladder.append(("kernel_1", ir))

    steps = [*_matmul_steps(), lambda state: BufferLayout().apply(state, BufferLayoutOption("sbuf_lhs_T", 16))]
    for step in steps:
        ir = step(ir)
        ladder.append((f"kernel_{len(ladder)}", ir))
    return ladder
