"""Controller: drive Task-3 steps k0..k32, byte-check each vs manual_transforms."""
from __future__ import annotations
from examples import manual_transforms
from examples.kernel_transforms import (_load_blk, _load_for, _load_leaf, _loop, _op_blk,
    _op_leaf, _psum_memset_blk, _psum_memset_leaf, _reorder_blk_to_nm, f_nkigym, INPUT_SPECS)
from test.transforms._ladder_compare import assert_matches_hand
from nkigym.codegen import render
from nkigym.ir import build_initial_ir
from nkigym.transforms import (BufferCompaction, BufferCompactionOption, BufferLayout,
    BufferLayoutOption, CodeMotion, CodeMotionOption, Reorder, ReorderOption, Split, SplitOption)

def _steps():
    return [
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d1_0"), inner_nid=_loop(ir, "i_d2_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_0"), inner_nid=_loop(ir, "i_d2_0"))),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_loop(ir, "i_d0_0"), factors=(2, 8), target_axis=None)),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_loop(ir, "i_d1_0"), factors=(4, 4), target_axis=None)),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_1"), inner_nid=_loop(ir, "i_d1_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_loop(ir, "i_d0_1"), inner_nid=_loop(ir, "i_d1_1"))),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=16)),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_op_leaf(ir, "NKITensorCopy"), factors=(4, 512), target_axis="d2")),
        lambda ir: _reorder_blk_to_nm(ir, _op_blk(ir, "NKITensorCopy")),
        lambda ir: CodeMotion().apply(ir, CodeMotionOption(block_nid=_op_blk(ir, "NKITensorCopy"), target_loop_nid=_loop(ir, "i_d2_0"), index=-1)),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_op_leaf(ir, "NKIStore"), factors=(4, 512), target_axis="d2")),
        lambda ir: _reorder_blk_to_nm(ir, _op_blk(ir, "NKIStore")),
        lambda ir: CodeMotion().apply(ir, CodeMotionOption(block_nid=_op_blk(ir, "NKIStore"), target_loop_nid=_loop(ir, "i_d2_0"), index=-1)),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption(tensor="sbuf_prod")),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption(tensor="sbuf_prod", list_len=16)),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_psum_memset_leaf(ir), factors=(4, 512), target_axis="d2")),
        lambda ir: _reorder_blk_to_nm(ir, _psum_memset_blk(ir)),
        lambda ir: CodeMotion().apply(ir, CodeMotionOption(block_nid=_psum_memset_blk(ir), target_loop_nid=_loop(ir, "i_d2_0"), index=0)),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption(tensor="psum_prod")),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_load_for(ir, "rhs", "i_d0_0"), factors=(2, 8), target_axis=None)),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_load_leaf(ir, "rhs"), factors=(4, 512), target_axis="d2")),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_load_for(ir, "rhs", "i_d0_1"), inner_nid=_load_for(ir, "rhs", "i_d2_0"))),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_load_for(ir, "rhs", "i_d0_0"), inner_nid=_load_for(ir, "rhs", "i_d2_0"))),
        lambda ir: CodeMotion().apply(ir, CodeMotionOption(block_nid=_load_blk(ir, "rhs"), target_loop_nid=_loop(ir, "i_d0_0"), index=0)),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption(tensor="sbuf_rhs")),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption(tensor="sbuf_rhs", list_len=8)),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_load_leaf(ir, "lhs_T"), factors=(4, 512), target_axis="d1")),
        lambda ir: Split().apply(ir, SplitOption(target_nid=_load_for(ir, "lhs_T", "i_d0_0"), factors=(2, 8), target_axis=None)),
        lambda ir: Reorder().apply(ir, ReorderOption(outer_nid=_load_for(ir, "lhs_T", "i_d0_1"), inner_nid=_load_for(ir, "lhs_T", "i_d1_0"))),
        lambda ir: CodeMotion().apply(ir, CodeMotionOption(block_nid=_load_blk(ir, "lhs_T"), target_loop_nid=_loop(ir, "i_d1_0"), index=0)),
        lambda ir: BufferCompaction().apply(ir, BufferCompactionOption(tensor="sbuf_lhs_T")),
        lambda ir: BufferLayout().apply(ir, BufferLayoutOption(tensor="sbuf_lhs_T", list_len=8)),
    ]

def main():
    ir = build_initial_ir(f_nkigym, INPUT_SPECS)
    ok = 0
    for i, step in enumerate(_steps(), start=1):
        ir = step(ir)
        try:
            assert_matches_hand(render(ir), getattr(manual_transforms, f"kernel_{i}"))
            ok = i
        except AssertionError:
            print(f"kernel_{i}: MISMATCH (byte-exact through k{ok})")
            print(render(ir))
            return
    print(f"ALL BYTE-EXACT k0..k{ok}")

if __name__ == "__main__":
    main()
