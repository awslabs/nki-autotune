"""Synthetic ``NKIOp`` with one ``SEQUENTIAL`` axis for Reorder legality tests.

No production op carries SEQUENTIAL today (matmul has ACCUMULATION on K;
RMSNorm/softmax prefix-scan ops will, when they ship). To verify the
legality rule before those ops land, we declare a minimal subclass and
build a hand-rolled IR that encloses one such leaf under two ForNodes
on its mapped dims.
"""

from __future__ import annotations

from typing import ClassVar

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import ForNode, ISANode, KernelTree
from nkigym.ops.base import AxisRole, NKIOp


class _SeqOp(NKIOp):
    """Minimal NKIOp with one PARALLEL ('P') and one SEQUENTIAL ('F') axis."""

    NAME: ClassVar[str] = "_seq_op_test"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.SEQUENTIAL}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset()
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}

    def _run(self, **kwargs):
        """Concrete stub so the class is instantiable; never called by legality checks."""
        return None


def build_seq_ir(p_extent: int = 256, f_extent: int = 256) -> tuple[KernelIR, int, int, int]:
    """Build a minimal IR: ``RootNode → ForNode(d0, trip=2) → ForNode(d1, trip=2) → _SeqOp leaf``.

    The leaf reads tensor 'x' with axis_map {'P': 'd0', 'F': 'd1'} so its
    SEQUENTIAL role on 'F' resolves to dim 'd1'. Returns ``(ir, outer_nid,
    inner_nid, leaf_nid)``. ``KernelIR``'s required dataclass fields
    (`func_name`, `param_names`, `return_name`) are populated with stubs;
    the legality check reads only ``ir.tree``.
    """
    tree = KernelTree()
    outer = tree.add_node(ForNode(dim="d0", trip=2), parent=tree.root)
    inner = tree.add_node(ForNode(dim="d1", trip=2), parent=outer)
    """Build the leaf payload. tensorize_sizes covers each axis (trip * tensorize == extent)."""
    leaf_data = ISANode(
        op_cls=_SeqOp,
        reads=("x",),
        writes=(),
        rmw=(),
        tensorize_sizes={"P": p_extent // 2, "F": f_extent // 2},
        axis_map={"P": "d0", "F": "d1"},
        kwargs={},
    )
    leaf_nid = tree.add_node(leaf_data, parent=inner)

    ir = KernelIR(
        func_name="_seq_fixture",
        param_names=[],
        return_name="",
        dim_sizes={"d0": p_extent, "d1": f_extent},
        tensors={},
        tree=tree,
        dependency=Dependency(tree),
    )
    return ir, outer, inner, leaf_nid


__all__ = ["build_seq_ir"]
