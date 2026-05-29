"""Synthetic ``NKIOp`` with one ``SEQUENTIAL`` axis for Reorder legality tests.

Builds a minimal IR by hand: a root :class:`BlockNode` containing one
leaf :class:`BlockNode` whose body is a chain of two :class:`ForNode`s
ending in a single :class:`ISANode` of a synthetic op declaring one
``SEQUENTIAL`` axis. Used to exercise legality rules before any
production op carries ``SEQUENTIAL`` semantics.
"""

from __future__ import annotations

from typing import ClassVar

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.expr import Const, Var
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.base import AxisRole, NKIOp


class _SeqOp(NKIOp):
    """Minimal NKIOp with PARALLEL ('P') and SEQUENTIAL ('F') axes."""

    NAME: ClassVar[str] = "_seq_op_test"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.SEQUENTIAL}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset()
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}

    def _run(self, **kwargs):
        return None


def build_seq_ir(p_extent: int = 256, f_extent: int = 256) -> tuple[KernelIR, int, int, int]:
    """Build a minimal hand-rolled IR enclosing a SEQUENTIAL-role leaf.

    Returns ``(ir, outer_nid, inner_nid, leaf_nid)`` for assertions.
    """
    tree = KernelTree()
    root_block = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=())
    root_block_nid = tree.add_node(root_block, parent=tree.root)
    leaf_block = BlockNode(
        iter_vars=(
            IterVar(axis="P", dom=(0, p_extent), role=AxisRole.PARALLEL),
            IterVar(axis="F", dom=(0, f_extent), role=AxisRole.SEQUENTIAL),
        ),
        iter_values=(Var(name="i_P_0"), Var(name="i_F_0")),
        reads=(
            BufferRegion(
                tensor="x", ranges=((Var(name="i_P_0"), Const(value=128)), (Var(name="i_F_0"), Const(value=f_extent)))
            ),
        ),
        writes=(),
    )
    leaf_block_nid = tree.add_node(leaf_block, parent=root_block_nid)
    outer = tree.add_node(ForNode(loop_var="i_P_0", extent=2), parent=leaf_block_nid)
    inner = tree.add_node(ForNode(loop_var="i_F_0", extent=2), parent=outer)
    """Add another BlockNode under inner to test descendant block legality check."""
    nested_block = BlockNode(
        iter_vars=(
            IterVar(axis="P", dom=(0, p_extent), role=AxisRole.PARALLEL),
            IterVar(axis="F", dom=(0, f_extent), role=AxisRole.SEQUENTIAL),
        ),
        iter_values=(Var(name="i_P_0"), Var(name="i_F_0")),
        reads=(
            BufferRegion(
                tensor="x", ranges=((Var(name="i_P_0"), Const(value=128)), (Var(name="i_F_0"), Const(value=f_extent)))
            ),
        ),
        writes=(),
    )
    nested_block_nid = tree.add_node(nested_block, parent=inner)
    leaf = tree.add_node(
        ISANode(
            op_cls=_SeqOp,
            operand_bindings={
                "data": BufferRegion(
                    tensor="x",
                    ranges=((Var(name="i_P_0"), Const(value=128)), (Var(name="i_F_0"), Const(value=f_extent))),
                )
            },
        ),
        parent=nested_block_nid,
    )
    ir = KernelIR(
        func_name="_seq_fixture",
        param_names=["x"],
        return_name="x",
        tree=tree,
        dependency=Dependency(tree),
        param_buffers={"x": Buffer(name="x", shape=(p_extent, f_extent), dtype="bfloat16", location="shared_hbm")},
    )
    return ir, outer, inner, leaf


__all__ = ["build_seq_ir"]
