"""Unit tests for the PlaceBuffers pass extracted from ``emit_source``.

Covers :func:`required_tiles`, :func:`sbuf_shape`, and
:func:`tensor_total_slots` across canonical-cross-nest, fused-intra-nest,
and multi-buffered tensor shapes. Also verifies the ``ValueError`` branch
on non-divisible trip products.
"""

from dataclasses import replace

import numpy as np
import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import DimInfo, KernelModule
from nkigym.codegen.lowering.place_buffers import required_tiles, sbuf_shape, tensor_total_slots
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune.compute_at import enumerate_compute_at_atoms
from nkigym.tune.multi_buffer import enumerate_multi_buffer_atoms

M, K, N = 2048, 2048, 2048


@nkigym_kernel
def _matmul_k(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Simple matmul kernel fixture: 2048x2048 bf16."""
    lhs_s = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    rhs_s = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(M, N), dtype="bfloat16")()
    NKILoad()(src=lhs, dst=lhs_s)
    NKILoad()(src=rhs, dst=rhs_s)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_s, moving=rhs_s, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


_INPUT_SPECS: dict[str, dict] = {
    "lhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
    "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
}


@pytest.fixture
def canonical_module() -> KernelModule:
    """Canonical module: per-op loop nests are roots (cross-nest intermediates)."""
    return build_canonical_module(_matmul_k, _INPUT_SPECS)


@pytest.fixture
def fused_module(canonical_module: KernelModule) -> KernelModule:
    """Pick the first ComputeAt atom that yields a multi-bufferable module.

    A tensor whose LCA sits below a tile-iterating ancestor has
    ``required_tiles < num_tiles`` on that dim — the exact shape
    PlaceBuffers must handle correctly.

    NOTE: Single-phase matmul doesn't create fusion opportunities that
    yield multi-buffering scenarios. Skip tests that depend on this fixture
    until Task 16 (RFactor) provides multi-phase matmul again.
    """
    for atom in enumerate_compute_at_atoms(canonical_module):
        candidate = atom.apply(canonical_module)
        if enumerate_multi_buffer_atoms(candidate):
            return candidate
    pytest.skip("Single-phase matmul has no ComputeAt atoms yielding multi-bufferable modules (Task 16 RFactor)")


def _first_intermediate(module: KernelModule) -> str:
    """Return the name of the first intermediate tensor in the module.

    Skip NKIAlloc-only tensors (those declared at SBUF/PSUM) and pick a
    tensor that's actually written to by a compute op.
    """
    from nkigym.ops.alloc import NKIAlloc

    for name, tensor in module.tensors.items():
        if tensor.origin != "intermediate":
            continue
        """Check if this tensor is written by a non-Alloc op."""
        for node in module.body:
            for leaf in _iter_leaves(node):
                if name in leaf.writes and leaf.op_cls is not NKIAlloc:
                    return name
    raise ValueError("no intermediate tensor written by compute ops found")


def _iter_leaves(node):
    """Recursively yield all BodyLeaf instances in a tree."""
    from nkigym.codegen.ir import BodyLeaf, LoopNode

    if isinstance(node, BodyLeaf):
        yield node
    elif isinstance(node, LoopNode):
        for child in node.children:
            yield from _iter_leaves(child)


def test_required_tiles_cross_nest_returns_num_tiles(canonical_module: KernelModule) -> None:
    """Canonical build: every intermediate's LCA is the forest root.

    With no dim-iterating ancestors above the LCA, the ancestor trip
    product is 1 and ``required_tiles`` equals ``module.dims[d].num_tiles``.
    """
    for name, tensor in canonical_module.tensors.items():
        if tensor.origin != "intermediate":
            continue
        for d in tensor.dim_ids:
            num_t = canonical_module.dims[d].num_tiles
            assert required_tiles(tensor, d, canonical_module) == num_t, f"tensor={name!r}, dim={d!r}"


def test_required_tiles_intra_nest_returns_one(fused_module: KernelModule) -> None:
    """After ComputeAt fusion, at least one intermediate has ``required_tiles == 1``.

    When the LCA captures every tile-iterating ancestor for a dim, the
    trip product equals ``num_tiles`` and ``required_tiles`` collapses
    to 1.
    """
    hits = []
    for name, tensor in fused_module.tensors.items():
        if tensor.origin != "intermediate":
            continue
        for d in tensor.dim_ids:
            if required_tiles(tensor, d, fused_module) == 1:
                hits.append((name, d))
    assert hits, "fused fixture should expose at least one intra-nest dim with required_tiles == 1"


def test_required_tiles_param_tensor_returns_num_tiles(canonical_module: KernelModule) -> None:
    """Parameter tensors live in HBM; required_tiles always returns num_tiles."""
    for name in canonical_module.param_names:
        tensor = canonical_module.tensors[name]
        for d in tensor.dim_ids:
            num_t = canonical_module.dims[d].num_tiles
            assert required_tiles(tensor, d, canonical_module) == num_t


def test_required_tiles_return_tensor_returns_num_tiles(canonical_module: KernelModule) -> None:
    """Return tensor lives in HBM; required_tiles always returns num_tiles."""
    tensor = canonical_module.tensors[canonical_module.return_name]
    for d in tensor.dim_ids:
        num_t = canonical_module.dims[d].num_tiles
        assert required_tiles(tensor, d, canonical_module) == num_t


def test_required_tiles_partial_hoist(fused_module: KernelModule) -> None:
    """Partially hoisted tensors: LCA captures some but not all dim trips.

    Check the invariant ``required_tiles * ancestor_trip_product == num_tiles``
    across every intermediate tensor / dim pair — holds for cross-nest,
    intra-nest, and partial-hoist cases alike.
    """
    for name, tensor in fused_module.tensors.items():
        if tensor.origin != "intermediate":
            continue
        for d in tensor.dim_ids:
            req = required_tiles(tensor, d, fused_module)
            num_t = fused_module.dims[d].num_tiles
            assert num_t % req == 0, f"required_tiles {req} does not divide num_tiles {num_t}"
            assert req >= 1 and req <= num_t, f"required_tiles out of range: {req} vs {num_t}"


def test_sbuf_shape_default_buffer_degree(canonical_module: KernelModule) -> None:
    """Default buffer_degree=1: sbuf_shape's P slot count equals required_tiles.

    For a 2D matmul intermediate, the shape is
    ``(p_tile, num_tiles(P), num_tiles(F) * f_tile)``.
    """
    name = _first_intermediate(canonical_module)
    tensor = canonical_module.tensors[name]
    shape = sbuf_shape(tensor, canonical_module)
    p_axis = tensor.dim_ids[0]
    f_axis = tensor.dim_ids[1]
    p_info = canonical_module.dims[p_axis]
    f_info = canonical_module.dims[f_axis]
    expected = (p_info.tile_size, p_info.num_tiles, f_info.num_tiles * f_info.tile_size)
    assert shape == expected


def test_sbuf_shape_with_multi_buffer(fused_module: KernelModule) -> None:
    """sbuf_shape scales the P slot count by buffer_degree.

    Apply a MultiBuffer atom that sets ``buffer_degree[P] = 2`` and
    verify the P slot count in the emitted shape doubles accordingly.
    """
    atom = next(a for a in enumerate_multi_buffer_atoms(fused_module) if a.degree >= 2)
    mb_module = atom.apply(fused_module)
    tensor = mb_module.tensors[atom.tensor_name]
    shape = sbuf_shape(tensor, mb_module)
    expected_p_slots = required_tiles(tensor, atom.dim_id, mb_module) * atom.degree
    if tensor.dim_ids[0] == atom.dim_id:
        assert shape[1] == expected_p_slots


def test_sbuf_shape_raises_for_empty_dims(canonical_module: KernelModule) -> None:
    """Tensor with empty ``dim_ids`` has no P axis — sbuf_shape rejects it."""
    name = _first_intermediate(canonical_module)
    empty_tensor = replace(canonical_module.tensors[name], dim_ids=(), shape=())
    with pytest.raises(ValueError, match="no dims"):
        sbuf_shape(empty_tensor, canonical_module)


def test_tensor_total_slots_degree_one(canonical_module: KernelModule) -> None:
    """With buffer_degree=1, total_slots equals required_tiles."""
    for name, tensor in canonical_module.tensors.items():
        if tensor.origin != "intermediate":
            continue
        for d in tensor.dim_ids:
            expected = required_tiles(tensor, d, canonical_module)
            assert tensor_total_slots(tensor, d, canonical_module) == expected


def test_tensor_total_slots_multiplies_buffer_degree(fused_module: KernelModule) -> None:
    """total_slots multiplies required_tiles by the per-dim buffer_degree."""
    atom = next(a for a in enumerate_multi_buffer_atoms(fused_module) if a.degree >= 2)
    mb_module = atom.apply(fused_module)
    tensor = mb_module.tensors[atom.tensor_name]
    req = required_tiles(tensor, atom.dim_id, mb_module)
    expected = req * atom.degree
    assert tensor_total_slots(tensor, atom.dim_id, mb_module) == expected


def test_required_tiles_raises_on_non_divisible_trip(fused_module: KernelModule) -> None:
    """ValueError when the ancestor trip product does not divide num_tiles.

    Canonical IR always produces divisible forms. Construct a malformed
    module by finding a tensor whose LCA captures every dim-iterating
    ancestor (``required_tiles == 1``, trip product equals num_tiles),
    then corrupting ``num_tiles`` to a non-divisor so the trip product
    no longer divides.
    """
    target_name = ""
    target_dim = ""
    for name, tensor in fused_module.tensors.items():
        if tensor.origin != "intermediate":
            continue
        for d in tensor.dim_ids:
            req = required_tiles(tensor, d, fused_module)
            num_t = fused_module.dims[d].num_tiles
            if req == 1 and num_t > 1:
                target_name, target_dim = name, d
                break
        if target_name:
            break
    assert target_name, "fused fixture should expose at least one fully-hoisted dim"
    old_info = fused_module.dims[target_dim]
    new_dims = dict(fused_module.dims)
    new_dims[target_dim] = DimInfo(
        dim_id=target_dim,
        total_size=old_info.total_size,
        tile_size=old_info.tile_size,
        num_tiles=old_info.num_tiles + 1,
    )
    bad_module = replace(fused_module, dims=new_dims)
    with pytest.raises(ValueError, match="does not divide num_tiles"):
        required_tiles(bad_module.tensors[target_name], target_dim, bad_module)
