"""Tests for new-loopnest detection inside :func:`nkigym.codegen.emit_body`.

A :class:`ForNode` whose parent is the tree root opens a fresh loopnest
(it sits at function-scope indent ``+1``); a :class:`ForNode` whose
parent is another :class:`ForNode` is nested inside that outer loop.
"""

from nkigym.codegen.body import is_loopnest_root
from nkigym.ir import build_initial_ir
from nkigym.ir.tree import ForNode
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy

K, M, N = 2048, 2048, 2048
_INPUT_SPECS: dict[str, tuple[int, ...]] = {"lhs_T": (K, M), "rhs": (K, N)}


@nkigym_kernel
def _matmul_fixture(lhs_T, rhs):
    """``lhs_T.T @ rhs`` — multiple per-op loop nests under the root."""
    sbuf_lhs_T = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    sbuf_rhs = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="shared_hbm", shape=(M, N), dtype="bfloat16")()

    NKILoad()(src=lhs_T, dst=sbuf_lhs_T)
    NKILoad()(src=rhs, dst=sbuf_rhs)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


def test_for_nodes_with_root_parent_are_loopnest_roots() -> None:
    """A ``ForNode`` whose parent is the tree root opens a fresh loopnest."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    for nid in ir.tree.preorder():
        if not isinstance(ir.tree.data(nid), ForNode):
            continue
        parent = ir.tree.parent(nid)
        if parent == ir.tree.root:
            assert is_loopnest_root(ir.tree, nid), f"node {nid} has root parent but is_loopnest_root=False"


def test_for_nodes_with_for_parent_are_nested() -> None:
    """A ``ForNode`` whose parent is another ``ForNode`` is nested, not a loopnest root."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    for nid in ir.tree.preorder():
        if not isinstance(ir.tree.data(nid), ForNode):
            continue
        parent = ir.tree.parent(nid)
        if parent is None or parent == ir.tree.root:
            continue
        if isinstance(ir.tree.data(parent), ForNode):
            assert not is_loopnest_root(ir.tree, nid), f"node {nid} nested under for-node {parent} but flagged as root"


def test_loopnest_roots_match_op_count() -> None:
    """One loopnest per compute op (allocs sit at root with no loops). The matmul fixture has 6 compute ops."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    roots = [
        nid for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ForNode) and is_loopnest_root(ir.tree, nid)
    ]
    assert len(roots) == 6, f"expected 6 loopnest roots (one per compute op), got {len(roots)}"
