"""Unit tests for ReverseComputeAt atom."""

import numpy as np
import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.dep_cache import DepCache
from nkigym.codegen.ir import BodyLeaf, DimInfo, KernelModule, LoopNode, Tensor, leaves_under
from nkigym.ops import nkigym_kernel
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.tune import AtomLegalityError
from nkigym.tune.reverse_compute_at import ReverseComputeAt, enumerate_reverse_compute_at_atoms


def _mod_hand(body: list, dims: dict[str, DimInfo], tensors: dict[str, Tensor]) -> KernelModule:
    """Build a minimal KernelModule from hand-rolled body / dims / tensors."""
    return KernelModule(
        func_name="f",
        param_names=[],
        return_name=next(iter(tensors)) if tensors else "x",
        tensors=tensors,
        dims=dims,
        body=body,
        dep=DepCache(scopes={}),
    )


from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.memset import NKIMemset
from nkigym.ops.tensor_copy import NKITensorCopy


@nkigym_kernel
def _matmul_k(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """First-class buffers matmul fixture for ReverseComputeAt atom tests."""
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


_INPUT_SPECS: dict[str, dict] = {
    "lhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
    "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
}


@pytest.fixture
def module():
    """Build a canonical KernelModule for the matmul fixture."""
    return build_canonical_module(_matmul_k, _INPUT_SPECS)


def test_enumerator_emits_legal_atoms(module):
    """Enumerator returns only legal atoms for the matmul fixture."""
    atoms = enumerate_reverse_compute_at_atoms(module)
    assert atoms
    for atom in atoms:
        assert atom.is_legal(module)


def test_apply_changes_tree(module):
    """Applying a legal atom rewrites the tree and preserves the leaf count."""
    atoms = enumerate_reverse_compute_at_atoms(module)
    assert atoms
    atom = atoms[0]
    new_mod = atom.apply(module)
    assert new_mod.body is not module.body
    old_leaves = sum(1 for root in module.body for _ in leaves_under(root))
    new_leaves = sum(1 for root in new_mod.body for _ in leaves_under(root))
    assert old_leaves == new_leaves


def test_reverse_apply_regenerates_residual_trip_on_partial_ancestor() -> None:
    """Partial-coverage ancestor → residual inner loop regenerated.

    Forest body:
    * Subtree 0: ``L(d0, trip=2)`` wrapping producer ``writes={"p"}``.
    * Subtree 1: consumer ``reads={"p"}`` at the root.

    Move the consumer under subtree 0's root loop. ``num_tiles[d0]=16``,
    ancestor ``covered=2`` → regenerated ``L(d0, 8)``.
    """
    producer = BodyLeaf(op_cls=object, writes=("p",), dim_role={"d0": AxisRole.PARALLEL})
    producer_outer = LoopNode("d0", 2, AxisRole.PARALLEL, children=[producer])
    consumer = BodyLeaf(op_cls=object, reads={"data": "p"}, dim_role={"d0": AxisRole.PARALLEL})
    dims = {"d0": DimInfo(dim_id="d0", total_size=2048, tile_size=128, num_tiles=16)}
    tensors = {
        "p": Tensor(
            name="p",
            dim_ids=("d0",),
            shape=(2048,),
            dtype="float32",
            origin="intermediate",
            location="sbuf",
            buffer_degree={"d0": 1},
        )
    }
    module = _mod_hand(body=[producer_outer, consumer], dims=dims, tensors=tensors)
    atom = ReverseComputeAt(leaf_path=(1,), target_loop_path=(0,))
    assert atom.is_legal(module)
    new_module = atom.apply(module)
    new_target = new_module.body[0]
    assert isinstance(new_target, LoopNode) and new_target.trip_count == 2
    regen_wrappers = [c for c in new_target.children if isinstance(c, LoopNode) and c.dim_id == "d0"]
    assert len(regen_wrappers) == 1
    residual_loop = regen_wrappers[0]
    assert residual_loop.trip_count == 8
    residual_child = residual_loop.children[0]
    assert isinstance(residual_child, BodyLeaf)
    assert residual_child.reads == {"data": "p"}


def test_reverse_apply_skips_fully_covered_dim() -> None:
    """Ancestor covers full num_tiles → no regeneration.

    Body:
    * Subtree 0: ``L(d0, trip=16)`` wrapping producer.
    * Subtree 1: consumer at the root.
    ``num_tiles[d0]=16`` → ``covered==num_tiles`` → no inner d0 loop.
    """
    producer = BodyLeaf(op_cls=object, writes=("p",), dim_role={"d0": AxisRole.PARALLEL})
    producer_outer = LoopNode("d0", 16, AxisRole.PARALLEL, children=[producer])
    consumer = BodyLeaf(op_cls=object, reads={"data": "p"}, dim_role={"d0": AxisRole.PARALLEL})
    dims = {"d0": DimInfo(dim_id="d0", total_size=2048, tile_size=128, num_tiles=16)}
    tensors = {
        "p": Tensor(
            name="p",
            dim_ids=("d0",),
            shape=(2048,),
            dtype="float32",
            origin="intermediate",
            location="sbuf",
            buffer_degree={"d0": 1},
        )
    }
    module = _mod_hand(body=[producer_outer, consumer], dims=dims, tensors=tensors)
    atom = ReverseComputeAt(leaf_path=(1,), target_loop_path=(0,))
    assert atom.is_legal(module)
    new_module = atom.apply(module)
    new_target = new_module.body[0]
    assert isinstance(new_target, LoopNode) and new_target.trip_count == 16
    """Target children: the original producer + the moved consumer
    (directly, no regen wrapper)."""
    moved = [c for c in new_target.children if isinstance(c, BodyLeaf) and c.reads == {"data": "p"}]
    assert len(moved) == 1
    regen_wrappers = [c for c in new_target.children if isinstance(c, LoopNode) and c.dim_id == "d0"]
    assert not regen_wrappers


def test_reverse_apply_raises_on_indivisible_residual() -> None:
    """Ancestor trip does not divide num_tiles → AtomLegalityError."""
    producer = BodyLeaf(op_cls=object, writes=("p",), dim_role={"d0": AxisRole.PARALLEL})
    producer_outer = LoopNode("d0", 3, AxisRole.PARALLEL, children=[producer])
    consumer = BodyLeaf(op_cls=object, reads={"data": "p"}, dim_role={"d0": AxisRole.PARALLEL})
    dims = {"d0": DimInfo(dim_id="d0", total_size=17 * 128, tile_size=128, num_tiles=17)}
    tensors = {
        "p": Tensor(
            name="p",
            dim_ids=("d0",),
            shape=(17 * 128,),
            dtype="float32",
            origin="intermediate",
            location="sbuf",
            buffer_degree={"d0": 1},
        )
    }
    module = _mod_hand(body=[producer_outer, consumer], dims=dims, tensors=tensors)
    atom = ReverseComputeAt(leaf_path=(1,), target_loop_path=(0,))
    assert atom.is_legal(module)
    try:
        atom.apply(module)
    except AtomLegalityError:
        return
    raise AssertionError("expected AtomLegalityError for indivisible residual")
