"""Layer-2 tests: KernelRewrite protocol and FuseLoops."""

from typing import get_type_hints

from nkigym.codegen.dep_graph import DepGraph
from nkigym.codegen.graph import OpGraph, parse_and_resolve
from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, build_canonical_forest
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose
from nkigym.tune import KernelRewrite
from nkigym.tune.fuse_loops import FuseLoops, enumerate_fusion_atoms

EPS = 1e-6


@nkigym_kernel
def _rmsnorm_matmul_for_rewrites(lhs, rhs):
    """Rmsnorm+matmul kernel used to exercise FuseLoops."""
    lhs_sbuf = NKILoad()(data=lhs)
    rhs_sbuf = NKILoad()(data=rhs)
    sum_sq = NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf)
    rms_inv = NKIActivation(op="rsqrt", scale=1 / 256, bias=EPS)(data=sum_sq)
    lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
    lhs_T = NKITranspose()(data=lhs_rms)
    prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


_RMSNORM_MATMUL_SPECS = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}


def _canonical_rmsnorm_matmul():
    g = parse_and_resolve(_rmsnorm_matmul_for_rewrites, _RMSNORM_MATMUL_SPECS)
    forest = build_canonical_forest(g)
    return g, forest


def test_kernel_rewrite_is_runtime_checkable_protocol() -> None:
    """KernelRewrite is a typing.Protocol exposing is_legal + apply."""
    assert hasattr(KernelRewrite, "is_legal")
    assert hasattr(KernelRewrite, "apply")


def test_kernel_rewrite_protocol_accepts_minimal_impl() -> None:
    """A class with is_legal and apply satisfies KernelRewrite structurally."""

    class NoopRewrite:
        def is_legal(self, op_graph, forest) -> bool:
            _ = op_graph, forest
            return True

        def apply(self, op_graph, forest):
            return op_graph, forest

    r: KernelRewrite = NoopRewrite()
    assert r.is_legal(None, []) is True
    new_g, new_f = r.apply(None, [])
    assert new_g is None
    assert new_f == []


def test_kernel_rewrite_type_hints_resolvable() -> None:
    """Protocol methods have resolvable type hints (smoke test for future tooling)."""
    hints = get_type_hints(KernelRewrite.is_legal)
    assert "forest" in hints


def test_is_legal_accepts_parallel_pair_with_matching_dim_and_trip_count() -> None:
    """ActivationReduce and TensorScalar share d0 PARALLEL at forest root; fusing them is legal."""
    g, forest = _canonical_rmsnorm_matmul()
    atom = FuseLoops(path=(), boundary=(2, 3), dim_id="d0")
    assert atom.is_legal(g, forest) is True


def test_is_legal_rejects_non_adjacent_boundary() -> None:
    """Non-adjacent (i, i+2) boundary is illegal."""
    g, forest = _canonical_rmsnorm_matmul()
    atom = FuseLoops(path=(), boundary=(2, 4), dim_id="d0")
    assert atom.is_legal(g, forest) is False


def test_is_legal_rejects_boundary_out_of_range() -> None:
    """Boundary past end of forest is illegal."""
    g, forest = _canonical_rmsnorm_matmul()
    atom = FuseLoops(path=(), boundary=(len(forest) - 1, len(forest)), dim_id="d0")
    assert atom.is_legal(g, forest) is False


def test_is_legal_rejects_dim_mismatch_on_adjacent_loopnodes() -> None:
    """Adjacent LoopNode roots on different concrete dims reject fusion."""
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
    ]
    atom = FuseLoops(path=(), boundary=(0, 1), dim_id="d0")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_caller_supplied_dim_not_matching_nodes() -> None:
    """If caller names a different dim than the two roots actually iterate, reject."""
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
    ]
    atom = FuseLoops(path=(), boundary=(0, 1), dim_id="dZZZ")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_trip_count_mismatch() -> None:
    """Mismatched trip counts reject fusion (even if roles and dim match)."""
    forest = [
        LoopNode("d0", 8, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
    ]
    atom = FuseLoops(path=(), boundary=(0, 1), dim_id="d0")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_role_mismatch() -> None:
    """One PARALLEL + one ACCUMULATION rejects fusion."""
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d0", 4, AxisRole.ACCUMULATION, [BodyLeaf(op_idx=1)]),
    ]
    atom = FuseLoops(path=(), boundary=(0, 1), dim_id="d0")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_body_leaf_on_either_side() -> None:
    """A BodyLeaf root cannot be fused (not a LoopNode)."""
    forest = [BodyLeaf(op_idx=0), LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)])]
    atom = FuseLoops(path=(), boundary=(0, 1), dim_id="d0")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_invalid_path() -> None:
    """Paths that walk out of range or into a BodyLeaf are rejected."""
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
    ]
    """path=(0,) walks into tree 0's children list; that list contains one
    BodyLeaf, so there is no adjacent pair to fuse."""
    atom = FuseLoops(path=(0,), boundary=(0, 1), dim_id="d0")
    assert atom.is_legal(None, forest) is False
    """path=(99,) is out of range at the root level."""
    atom2 = FuseLoops(path=(99,), boundary=(0, 1), dim_id="d0")
    assert atom2.is_legal(None, forest) is False


def test_apply_merges_adjacent_roots_concatenating_children() -> None:
    """apply produces one LoopNode whose children are A.children ++ B.children, in order."""
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0, phase="A0")]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1, phase="B0")]),
    ]
    atom = FuseLoops(path=(), boundary=(0, 1), dim_id="d0")
    new_g, new_forest = atom.apply(None, forest)
    assert new_g is None
    assert len(new_forest) == 1
    merged = new_forest[0]
    assert isinstance(merged, LoopNode)
    assert merged.dim_id == "d0"
    assert merged.trip_count == 4
    assert merged.role is AxisRole.PARALLEL
    assert len(merged.children) == 2
    first, second = merged.children
    assert isinstance(first, BodyLeaf) and first.phase == "A0"
    assert isinstance(second, BodyLeaf) and second.phase == "B0"


def test_apply_at_nested_path_fuses_sibling_children() -> None:
    """apply with path=(0,) merges adjacent children inside forest[0]'s LoopNode."""
    inner_left = LoopNode("d0", 1, AxisRole.PARALLEL, [BodyLeaf(op_idx=0, phase="L")])
    inner_right = LoopNode("d0", 1, AxisRole.PARALLEL, [BodyLeaf(op_idx=0, phase="R")])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner_left, inner_right])
    forest = [outer]
    atom = FuseLoops(path=(0,), boundary=(0, 1), dim_id="d0")
    assert atom.is_legal(None, forest) is True
    _, new_forest = atom.apply(None, forest)
    assert len(new_forest) == 1
    new_outer = new_forest[0]
    assert isinstance(new_outer, LoopNode)
    assert len(new_outer.children) == 1
    merged_inner = new_outer.children[0]
    assert isinstance(merged_inner, LoopNode)
    assert merged_inner.dim_id == "d0"
    assert merged_inner.trip_count == 1
    assert [c.phase for c in merged_inner.children] == ["L", "R"]


def test_enumerate_finds_nested_sibling_pairs() -> None:
    """The enumerator walks every LoopNode.children list, not just the forest root."""
    inner_left = LoopNode("d0", 1, AxisRole.PARALLEL, [BodyLeaf(op_idx=0, phase="L")])
    inner_right = LoopNode("d0", 1, AxisRole.PARALLEL, [BodyLeaf(op_idx=1, phase="R")])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner_left, inner_right])
    forest = [outer]
    atoms = enumerate_fusion_atoms(None, forest)
    """Exactly one atom at path=(0,), boundary=(0, 1), dim=d0."""
    matching = [a for a in atoms if a.path == (0,) and a.boundary == (0, 1) and a.dim_id == "d0"]
    assert len(matching) == 1


def test_enumerate_fusion_atoms_rmsnorm_matmul_canonical_has_sensible_atoms() -> None:
    """Canonical rmsnorm+matmul yields atoms at boundaries where both roots are PARALLEL on same dim."""
    g, forest = _canonical_rmsnorm_matmul()
    atoms = enumerate_fusion_atoms(g, forest)
    """Op indices in _rmsnorm_matmul_for_rewrites:
      0=Load(lhs), 1=Load(rhs), 2=ActivationReduce, 3=Activation,
      4=TensorScalar, 5=Transpose, 6=Matmul, 7=Store.
    Forest-root atoms must include (2, 3), (3, 4), and (6, 7) on d0."""
    root_atoms = [(a.boundary, a.dim_id) for a in atoms if a.path == ()]
    """Literal-adjacent fuses on d0 remain; the topological generalisation
    adds (0, 2) — Load(lhs) ↔ ActivationReduce with independent Load(rhs)
    in between."""
    assert ((0, 2), "d0") in root_atoms
    assert ((2, 3), "d0") in root_atoms
    assert ((3, 4), "d0") in root_atoms
    assert ((6, 7), "d0") in root_atoms


def test_enumerate_returns_empty_on_empty_forest() -> None:
    """Enumeration on an empty forest returns an empty list."""
    assert enumerate_fusion_atoms(None, []) == []


def test_enumerate_returns_empty_when_no_adjacent_pair_matches() -> None:
    """Forest where adjacent trees differ on dim_id yields no atoms."""
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
    ]
    assert enumerate_fusion_atoms(None, forest) == []


def test_enumerate_skips_accumulation_pairs() -> None:
    """Adjacent ACCUMULATION roots on the same dim yield no atom."""
    forest = [
        LoopNode("d0", 4, AxisRole.ACCUMULATION, [BodyLeaf(op_idx=0)]),
        LoopNode("d0", 4, AxisRole.ACCUMULATION, [BodyLeaf(op_idx=1)]),
    ]
    assert enumerate_fusion_atoms(None, forest) == []


def test_enumerate_skips_trip_count_mismatch() -> None:
    """Adjacent PARALLEL roots on the same dim with different trip counts yield no atom."""
    forest = [
        LoopNode("d0", 8, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
    ]
    assert enumerate_fusion_atoms(None, forest) == []


def test_apply_shrinks_forest_length_by_one() -> None:
    """Fusing at root (i, i+1) removes one entry from the forest."""
    g, forest = _canonical_rmsnorm_matmul()
    original_len = len(forest)
    atom = FuseLoops(path=(), boundary=(2, 3), dim_id="d0")
    _, new_forest = atom.apply(g, forest)
    assert len(new_forest) == original_len - 1


def test_apply_preserves_invariant_on_rmsnorm_matmul() -> None:
    """After fusing activation_reduce ↔ tensor_scalar on d0, check_invariant still holds."""
    from nkigym.codegen.loop_forest import check_invariant, compute_phase_touched

    g, forest = _canonical_rmsnorm_matmul()
    atom = FuseLoops(path=(), boundary=(2, 3), dim_id="d0")
    _, new_forest = atom.apply(g, forest)
    num_tiles = {d: info.num_tiles for d, info in g.dims.items()}
    op_touched = {o.idx: o.touched_dims for o in g.ops}
    phase_touched = compute_phase_touched(g)
    check_invariant(new_forest, num_tiles, op_touched, phase_touched)


def test_apply_merged_root_holds_both_subtrees_at_same_depth() -> None:
    """The merged LoopNode's children = old-A.children ++ old-B.children in program order."""
    g, forest = _canonical_rmsnorm_matmul()
    a_children_count = len(forest[2].children)
    b_children_count = len(forest[3].children)
    atom = FuseLoops(path=(), boundary=(2, 3), dim_id="d0")
    _, new_forest = atom.apply(g, forest)
    merged = new_forest[2]
    assert isinstance(merged, LoopNode)
    assert merged.dim_id == "d0"
    assert merged.role is AxisRole.PARALLEL
    assert len(merged.children) == a_children_count + b_children_count


def test_re_enumeration_after_apply_finds_next_atom_at_shifted_boundary() -> None:
    """After fusing at the forest root, re-enumeration must not include stale root indices."""
    g, forest = _canonical_rmsnorm_matmul()
    atoms0 = enumerate_fusion_atoms(g, forest)
    assert any(a.path == () and a.boundary == (6, 7) and a.dim_id == "d0" for a in atoms0)
    forest1 = forest
    for a in atoms0:
        if a.path == () and a.boundary == (6, 7) and a.dim_id == "d0":
            _, forest1 = a.apply(g, forest1)
            break
    atoms1 = enumerate_fusion_atoms(g, forest1)
    for a in atoms1:
        if a.path == ():
            assert a.boundary[1] < len(forest1), f"Stale boundary {a.boundary} beyond forest length {len(forest1)}"


def test_compose_outer_then_inner_fusion() -> None:
    """Fuse outer d0 pair, then fuse the inner sibling d0 loops it exposed."""
    from nkigym.codegen.loop_forest import check_invariant, compute_phase_touched

    g, forest = _canonical_rmsnorm_matmul()
    outer = FuseLoops(path=(), boundary=(2, 3), dim_id="d0")
    _, forest1 = outer.apply(g, forest)
    """After the outer fuse, the merged tree sits at index 2. Its two direct
    children are the former outer tile-tier LoopNodes from activation_reduce
    and tensor_scalar — both LoopNode(d0, 1, PARALLEL). Re-enumerate and
    look for an atom at path=(2,) to fuse those inner siblings."""
    inner_atoms = [a for a in enumerate_fusion_atoms(g, forest1) if a.path == (2,)]
    assert inner_atoms, "expected an inner atom inside the merged d0 tree at path=(2,)"
    inner = inner_atoms[0]
    _, forest2 = inner.apply(g, forest1)
    num_tiles = {d: info.num_tiles for d, info in g.dims.items()}
    op_touched = {o.idx: o.touched_dims for o in g.ops}
    phase_touched = compute_phase_touched(g)
    check_invariant(forest2, num_tiles, op_touched, phase_touched)


def test_is_legal_accepts_topologically_adjacent_pair_with_independent_intervening() -> None:
    """A pair (i, j) with j > i+1 is legal when every intervening sibling commutes with both endpoints.

    On the canonical rmsnorm+matmul forest, siblings 2 (ActivationReduce
    writing sum_sq from lhs_sbuf) and 4 (TensorScalar writing lhs_rms
    from lhs_sbuf + rms_inv) share d0 PARALLEL. Sibling 3 (Activation
    writing rms_inv from sum_sq) reads sum_sq and writes rms_inv — the
    Activation depends on ActivationReduce (RAW via sum_sq), so the
    pair (2, 4) is NOT topologically adjacent. Use a simpler hand-built
    forest to exercise the acceptance path.
    """
    dep = DepGraph(
        producer={"a": 0, "b": 1, "d_out": 2},
        consumers={"a": (2,), "b": (), "d_out": ()},
        reads={0: frozenset(), 1: frozenset(), 2: frozenset({"a"})},
        writes={0: frozenset({"a"}), 1: frozenset({"b"}), 2: frozenset({"d_out"})},
    )
    og = OpGraph(func_name="f", param_names=[], return_name="d_out", tensors={}, dims={}, ops=[], dep=dep)
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=2)]),
    ]
    atom = FuseLoops(path=(), boundary=(0, 2), dim_id="d0")
    assert atom.is_legal(og, forest) is True


def test_is_legal_rejects_topologically_non_adjacent_pair() -> None:
    """A pair (i, j) with an intervening sibling that depends on i is illegal.

    Ops: 0 writes a. 1 reads a, writes b. 2 reads a, writes c.
    Sibling 1 has a RAW edge with sibling 0, so sibling 1 cannot pass
    the producer (sibling 0) to move left. Fuse (0, 2) must be rejected.
    """
    dep = DepGraph(
        producer={"a": 0, "b": 1, "c": 2},
        consumers={"a": (1, 2), "b": (), "c": ()},
        reads={0: frozenset(), 1: frozenset({"a"}), 2: frozenset({"a"})},
        writes={0: frozenset({"a"}), 1: frozenset({"b"}), 2: frozenset({"c"})},
    )
    og = OpGraph(func_name="f", param_names=[], return_name="c", tensors={}, dims={}, ops=[], dep=dep)
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=2)]),
    ]
    atom = FuseLoops(path=(), boundary=(0, 2), dim_id="d0")
    assert atom.is_legal(og, forest) is False


def test_is_legal_rejects_boundary_with_role_mismatch_on_endpoints() -> None:
    """Topological-adjacency check does not override the three-field rule.

    Siblings 0 (PARALLEL) and 2 (ACCUMULATION) on d0 with an independent
    sibling 1 in between. The pair fails the role check regardless of
    topology."""
    dep = DepGraph(
        producer={},
        consumers={},
        reads={0: frozenset(), 1: frozenset(), 2: frozenset()},
        writes={0: frozenset(), 1: frozenset(), 2: frozenset()},
    )
    og = OpGraph(func_name="f", param_names=[], return_name="x", tensors={}, dims={}, ops=[], dep=dep)
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
        LoopNode("d0", 4, AxisRole.ACCUMULATION, [BodyLeaf(op_idx=2)]),
    ]
    atom = FuseLoops(path=(), boundary=(0, 2), dim_id="d0")
    assert atom.is_legal(og, forest) is False


def test_apply_topological_fuse_pushes_survivors_left_and_lands_at_consumer_slot() -> None:
    """Fusing (0, 2) with an independent sibling at 1:
    forest goes from [A, B, C] to [B, fused(A ‖ C)].
    """
    dep = DepGraph(
        producer={"a": 0, "b": 1, "c": 2},
        consumers={"a": (2,), "b": (), "c": ()},
        reads={0: frozenset(), 1: frozenset(), 2: frozenset({"a"})},
        writes={0: frozenset({"a"}), 1: frozenset({"b"}), 2: frozenset({"c"})},
    )
    og = OpGraph(func_name="f", param_names=[], return_name="c", tensors={}, dims={}, ops=[], dep=dep)
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0, phase="A")]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1, phase="B")]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=2, phase="C")]),
    ]
    atom = FuseLoops(path=(), boundary=(0, 2), dim_id="d0")
    assert atom.is_legal(og, forest) is True
    _, new_forest = atom.apply(og, forest)
    assert len(new_forest) == 2
    """B slid left of the former producer position."""
    assert isinstance(new_forest[0], LoopNode)
    assert new_forest[0].children[0].phase == "B"
    """Fused nest lands at consumer slot with producer-body first."""
    assert isinstance(new_forest[1], LoopNode)
    assert new_forest[1].dim_id == "d0"
    assert new_forest[1].role is AxisRole.PARALLEL
    assert [child.phase for child in new_forest[1].children] == ["A", "C"]


def test_apply_topological_fuse_preserves_survivor_relative_order() -> None:
    """Two intervening siblings keep their relative order in the survivor list.

    Forest: [A, B, C, D] where A (writes a), D (reads a), B and C
    both independent. Fuse (0, 3) pushes [B, C] left of A's position
    preserving order — result: [B, C, fused(A ‖ D)].
    """
    dep = DepGraph(
        producer={"a": 0, "b": 1, "c": 2, "d_out": 3},
        consumers={"a": (3,)},
        reads={0: frozenset(), 1: frozenset(), 2: frozenset(), 3: frozenset({"a"})},
        writes={0: frozenset({"a"}), 1: frozenset({"b"}), 2: frozenset({"c"}), 3: frozenset({"d_out"})},
    )
    og = OpGraph(func_name="f", param_names=[], return_name="d_out", tensors={}, dims={}, ops=[], dep=dep)
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0, phase="A")]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1, phase="B")]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=2, phase="C")]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=3, phase="D")]),
    ]
    atom = FuseLoops(path=(), boundary=(0, 3), dim_id="d0")
    assert atom.is_legal(og, forest) is True
    _, new_forest = atom.apply(og, forest)
    phases = [n.children[0].phase for n in new_forest]
    assert phases[:2] == ["B", "C"]
    """Fused nest at new index 2 holds A then D."""
    assert [c.phase for c in new_forest[2].children] == ["A", "D"]


def test_apply_literal_adjacent_fuse_unchanged_behaviour() -> None:
    """For j == i+1 the generalized apply behaves identically to the previous implementation.

    Explicit regression: fused nest lands at j's slot with [*A.children,
    *B.children]; no siblings moved; length shrinks by one.
    """
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0, phase="A0")]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1, phase="B0")]),
    ]
    atom = FuseLoops(path=(), boundary=(0, 1), dim_id="d0")
    _, new_forest = atom.apply(None, forest)
    assert len(new_forest) == 1
    merged = new_forest[0]
    assert isinstance(merged, LoopNode)
    assert [c.phase for c in merged.children] == ["A0", "B0"]


def test_enumerate_emits_topological_non_adjacent_pair_when_independent_sibling_in_between() -> None:
    """Enumerator emits (0, 2) when sibling 1 is independent of 0 and 2."""
    dep = DepGraph(
        producer={"a": 0, "b": 1, "c": 2},
        consumers={"a": (2,)},
        reads={0: frozenset(), 1: frozenset(), 2: frozenset({"a"})},
        writes={0: frozenset({"a"}), 1: frozenset({"b"}), 2: frozenset({"c"})},
    )
    og = OpGraph(func_name="f", param_names=[], return_name="c", tensors={}, dims={}, ops=[], dep=dep)
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=2)]),
    ]
    atoms = enumerate_fusion_atoms(og, forest)
    root_atoms = [a for a in atoms if a.path == ()]
    """(0, 1), (1, 2), and (0, 2) are all legal."""
    boundaries = {a.boundary for a in root_atoms}
    assert (0, 1) in boundaries
    assert (1, 2) in boundaries
    assert (0, 2) in boundaries


def test_enumerate_omits_pair_blocked_by_intervening_dependency() -> None:
    """When sibling 1 depends on sibling 0 (RAW), (0, 2) is omitted."""
    dep = DepGraph(
        producer={"a": 0, "b": 1, "c": 2},
        consumers={"a": (1, 2)},
        reads={0: frozenset(), 1: frozenset({"a"}), 2: frozenset({"a"})},
        writes={0: frozenset({"a"}), 1: frozenset({"b"}), 2: frozenset({"c"})},
    )
    og = OpGraph(func_name="f", param_names=[], return_name="c", tensors={}, dims={}, ops=[], dep=dep)
    forest = [
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)]),
        LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=2)]),
    ]
    atoms = enumerate_fusion_atoms(og, forest)
    boundaries = {a.boundary for a in atoms if a.path == ()}
    assert (0, 2) not in boundaries
    """The literal-adjacent atoms remain legal — (0, 1) checks only the
    three-field rule; (1, 2) the same. Both independent of the
    intervening-sibling rule."""
    assert (0, 1) in boundaries
    assert (1, 2) in boundaries
