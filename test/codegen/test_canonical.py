"""Tests for canonical module builder."""

import numpy as np

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import BodyLeaf, LoopNode
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy


@nkigym_kernel
def _matmul_k(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Simple matmul kernel (first-class buffers form) used as the canonical-module test fixture."""
    lhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs, dst=lhs_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


def test_builds_kernel_module_shape() -> None:
    """The builder returns a KernelModule with the expected surface shape."""
    input_specs = {
        "lhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
        "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
    }
    km = build_canonical_module(_matmul_k, input_specs)
    assert km.func_name == "_matmul_k"
    assert km.param_names == ["lhs", "rhs"]
    """First-class buffers: 11 leaves (5 allocs + 2 loads + 1 memset + 1 matmul + 1 copy + 1 store)."""
    assert len(km.body) == 11


def test_leaves_are_self_describing() -> None:
    """Matmul leaf must carry its full metadata directly.

    This is the headline Task 4 invariant: the renderer and legality
    checks must be able to operate on the leaf alone without a sidecar
    op graph, so every field described by BodyLeaf must be populated.
    """
    input_specs = {
        "lhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
        "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
    }
    km = build_canonical_module(_matmul_k, input_specs)

    def find_matmul_leaf(node: LoopNode | BodyLeaf) -> BodyLeaf | None:
        """Return the first matmul leaf found in ``node``."""
        if isinstance(node, BodyLeaf):
            return node if node.op_cls is NKIMatmul else None
        for c in node.children:
            found = find_matmul_leaf(c)
            if found is not None:
                return found
        return None

    """First-class buffers: single matmul leaf (no phases), should be in tree 8."""
    leaf = None
    for tree in km.body:
        leaf = find_matmul_leaf(tree)
        if leaf:
            break
    assert leaf is not None
    assert leaf.op_cls is NKIMatmul
    """First-class buffers: matmul RMW operand is in reads_writes."""
    assert leaf.reads == {"stationary": "lhs_sbuf", "moving": "rhs_sbuf"}
    assert leaf.reads_writes == ("psum_acc",)
    assert leaf.dim_role
    assert all(v is not None for v in leaf.dim_role.values())


def test_leaves_have_isolated_metadata() -> None:
    """Mutating one leaf's metadata must not bleed into other leaves."""
    input_specs = {
        "lhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
        "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
    }
    km = build_canonical_module(_matmul_k, input_specs)

    def find_all_load_leaves(node: LoopNode | BodyLeaf, acc: list[BodyLeaf]) -> None:
        """Append every load BodyLeaf reachable from ``node`` into ``acc``."""
        if isinstance(node, BodyLeaf):
            if node.op_cls is NKILoad:
                acc.append(node)
            return
        for c in node.children:
            find_all_load_leaves(c, acc)

    leaves: list[BodyLeaf] = []
    for tree in km.body:
        find_all_load_leaves(tree, leaves)
    assert len(leaves) == 2
    leaves[0].reads["bogus"] = "injected"
    for other in leaves[1:]:
        assert "bogus" not in other.reads


def test_canonical_loop_names_assigned() -> None:
    """Every LoopNode has a canonical ``i_<dim>_<ordinal>`` name."""
    input_specs = {
        "lhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
        "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
    }
    km = build_canonical_module(_matmul_k, input_specs)
    """Find the first LoopNode root (skip NKIAlloc leaves)."""
    loop_root = None
    for root in km.body:
        if isinstance(root, LoopNode):
            loop_root = root
            break
    assert loop_root is not None, "expected at least one LoopNode in body"
    assert loop_root.name is not None
    assert loop_root.name.startswith("i_")
