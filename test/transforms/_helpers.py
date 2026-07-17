"""Shared tree queries for transform tests."""

from __future__ import annotations

from nkigym.ir import KernelIR
from nkigym.ir.tree import ForNode, ISANode


def block_for_op(ir: KernelIR, op_name: str) -> int:
    """Return the single-leaf block containing the named operation."""
    for nid in ir.tree.blocks():
        leaves: list[ISANode] = []
        for descendant in ir.tree.descendants(nid):
            node = ir.tree.data(descendant)
            if isinstance(node, ISANode):
                leaves.append(node)
        if len(leaves) == 1 and leaves[0].op_cls.__name__ == op_name:
            return nid
    raise AssertionError(f"no leaf block for {op_name}")


def first_for_in(ir: KernelIR, block_nid: int) -> int:
    """Return the first loop nested in a block."""
    for descendant in ir.tree.preorder(block_nid):
        if isinstance(ir.tree.data(descendant), ForNode):
            return descendant
    raise AssertionError(f"no loop in block {block_nid}")


def leaf_for_op(ir: KernelIR, op_name: str, occurrence: int = 0) -> int:
    """Return one ISA leaf for an operation name."""
    leaves: list[int] = []
    for nid in ir.tree.preorder():
        node = ir.tree.data(nid)
        if isinstance(node, ISANode) and node.op_cls.__name__ == op_name:
            leaves.append(nid)
    return leaves[occurrence]


def load_block_reading(ir: KernelIR, tensor: str) -> int:
    """Return the single-leaf load block reading the named tensor."""
    for nid in ir.tree.blocks():
        leaves: list[ISANode] = []
        for descendant in ir.tree.descendants(nid):
            node = ir.tree.data(descendant)
            if isinstance(node, ISANode):
                leaves.append(node)
        if len(leaves) != 1:
            continue
        leaf = leaves[0]
        if leaf.op_cls.__name__ == "NKILoad" and leaf.operand_bindings["src"].tensor == tensor:
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
