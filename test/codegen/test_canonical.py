"""Tests for canonical module builder."""

import numpy as np

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import BodyLeaf, LoopNode
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore


@nkigym_kernel
def _matmul_k(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Simple matmul kernel used as the canonical-module test fixture."""
    lhs_s = NKILoad()(data=lhs)
    rhs_s = NKILoad()(data=rhs)
    out_s = NKIMatmul()(stationary=lhs_s, moving=rhs_s)
    out = NKIStore()(data=out_s)
    return out


def test_builds_kernel_module_shape() -> None:
    """The builder returns a KernelModule with the expected surface shape."""
    input_specs = {
        "lhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
        "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
    }
    km = build_canonical_module(_matmul_k, input_specs)
    assert km.func_name == "_matmul_k"
    assert km.param_names == ["lhs", "rhs"]
    assert len(km.body) == 4


def test_leaves_are_self_describing() -> None:
    """Matmul compute-phase leaf must carry its full metadata directly.

    This is the headline Task 4 invariant: the renderer and legality
    checks must be able to operate on the leaf alone without a sidecar
    op graph, so every field described by BodyLeaf must be populated.
    """
    input_specs = {
        "lhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
        "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
    }
    km = build_canonical_module(_matmul_k, input_specs)

    def find_compute_leaf(node: LoopNode | BodyLeaf) -> BodyLeaf | None:
        """Return the first matmul compute-phase leaf found in ``node``."""
        if isinstance(node, BodyLeaf):
            return node if node.op_cls is NKIMatmul and node.phase == "compute" else None
        for c in node.children:
            found = find_compute_leaf(c)
            if found is not None:
                return found
        return None

    leaf = find_compute_leaf(km.body[2])
    assert leaf is not None
    assert leaf.op_cls is NKIMatmul
    assert leaf.phase == "compute"
    assert leaf.reads == {"stationary": "lhs_s", "moving": "rhs_s"}
    assert leaf.writes == ("out_s",)
    assert leaf.dim_role
    assert all(v is not None for v in leaf.dim_role.values())


def test_leaves_have_isolated_metadata() -> None:
    """Mutating one leaf's metadata must not bleed into other leaves."""
    input_specs = {
        "lhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
        "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
    }
    km = build_canonical_module(_matmul_k, input_specs)

    def find_all_matmul_leaves(node: LoopNode | BodyLeaf, acc: list[BodyLeaf]) -> None:
        """Append every matmul BodyLeaf reachable from ``node`` into ``acc``."""
        if isinstance(node, BodyLeaf):
            if node.op_cls is NKIMatmul:
                acc.append(node)
            return
        for c in node.children:
            find_all_matmul_leaves(c, acc)

    leaves: list[BodyLeaf] = []
    find_all_matmul_leaves(km.body[2], leaves)
    assert len(leaves) >= 2
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
    root = km.body[0]
    assert isinstance(root, LoopNode)
    assert root.name is not None
    assert root.name.startswith("i_")
