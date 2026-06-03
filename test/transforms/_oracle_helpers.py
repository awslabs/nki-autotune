"""Shared helpers for the Layer-B structural-oracle transform tests.

The Split (outer-trip + tensorize) and Reorder oracle guards all read back the loop
nest *enclosing a specific ISA leaf* — sibling blocks (loads, tensor_copy, store) each
carry their own block-local ``i_d{k}`` loops, so a full-tree preorder over-collects them.
This module centralizes that "restrict to the leaf's enclosing loops" predicate.
"""

from __future__ import annotations

from nkigym.ir import KernelIR
from nkigym.ir.tree import ForNode


def enclosing_for_nids(ir: KernelIR, leaf_nid: int, dim_prefix: str) -> list[int]:
    """Return the ForNode ancestors of ``leaf_nid`` whose loop_var starts ``dim_prefix``.

    Ordered outer -> inner (``ir.tree.ancestors`` returns root -> leaf order, which the
    enclosing loops inherit). Filtering by ``loop_var`` prefix (e.g. ``"i_d1"`` for a
    single dim or ``"i_d"`` for all dims) isolates exactly the loops the transform
    touched on the chosen leaf, skipping the leaf's siblings' identically named loops.

    Parameters
    ----------
    ir:
        The kernel IR to read.
    leaf_nid:
        The ISA-leaf node whose enclosing loops are wanted.
    dim_prefix:
        Loop-var name prefix selecting which dims to collect (e.g. ``"i_d1"``, ``"i_d"``).

    Returns
    -------
    list[int]
        The matching ForNode nids, outer -> inner.
    """
    return [
        a
        for a in ir.tree.ancestors(leaf_nid)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var.startswith(dim_prefix)
    ]
