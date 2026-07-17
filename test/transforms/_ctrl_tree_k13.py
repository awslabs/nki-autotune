"""Dump the k13 tree: where do sbuf_prod's touchers sit, and what is their LCA node?"""
from __future__ import annotations
from examples.kernel_transforms import (_loop, _op_blk, _op_leaf, _reorder_blk_to_nm, f_nkigym, INPUT_SPECS)
from nkigym.ir import build_initial_ir
from nkigym.ir.tree import BlockNode, ForNode, ISANode
from nkigym.transforms import (BufferLayout, BufferLayoutOption, CodeMotion, CodeMotionOption,
    Reorder, ReorderOption, Split, SplitOption)

def drive_to_13():
    steps = [
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
    ]
    ir = build_initial_ir(f_nkigym, INPUT_SPECS)
    for s in steps:
        ir = s(ir)
    return ir

def kind(tree, nid):
    d = tree.data(nid)
    if isinstance(d, ForNode): return f"For({d.loop_var},{d.extent})"
    if isinstance(d, BlockNode): return f"Block(alloc={[b.name for b in d.alloc_buffers]})"
    if isinstance(d, ISANode): return f"ISA({d.op_cls.__name__})"
    return "?"

def show(tree, nid, depth=0):
    print("  " * depth + f"#{nid} {kind(tree, nid)}")
    for c in tree.children(nid):
        show(tree, c, depth + 1)

def main():
    ir = drive_to_13()
    print("=== TREE ===")
    show(ir.tree, ir.tree.root)
    touchers = [n for n in ir.tree.preorder() if isinstance(ir.tree.data(n), ISANode)
                and any(r.tensor == "sbuf_prod" for r in ir.tree.data(n).operand_bindings.values())]
    print("=== sbuf_prod touchers:", touchers)
    for t in touchers:
        print(f"  #{t} ancestors: {ir.tree.ancestors(t)}")
    own = next(n for n in ir.tree.blocks() if any(b.name=="sbuf_prod" for b in ir.tree.data(n).alloc_buffers))
    print(f"=== sbuf_prod alloc'd on block #{own} (ancestors {ir.tree.ancestors(own)})")

if __name__ == "__main__":
    main()
