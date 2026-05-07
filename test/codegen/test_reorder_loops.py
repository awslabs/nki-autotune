"""Layer-2 tests: ReorderLoops atom and enumerator."""

from nkigym.codegen.loop_forest import BodyLeaf, LoopNode
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose


def test_reorder_loops_is_frozen_dataclass_with_path_outer_inner_dims() -> None:
    """ReorderLoops exposes path, outer_dim, inner_dim as frozen fields."""
    from nkigym.tune.reorder_loops import ReorderLoops

    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    assert atom.path == (0,)
    assert atom.outer_dim == "d0"
    assert atom.inner_dim == "d1"


def test_roles_commute_par_par_true() -> None:
    """Two PARALLEL loops always commute."""
    from nkigym.tune.reorder_loops import _roles_commute

    a = LoopNode("d0", 4, AxisRole.PARALLEL)
    b = LoopNode("d1", 4, AxisRole.PARALLEL)
    assert _roles_commute(a, b) is True


def test_roles_commute_par_acc_true_both_orderings() -> None:
    """PAR×ACC and ACC×PAR both commute (one is outer or the other)."""
    from nkigym.tune.reorder_loops import _roles_commute

    par = LoopNode("d0", 4, AxisRole.PARALLEL)
    acc = LoopNode("d1", 4, AxisRole.ACCUMULATION, reduce_op="add")
    assert _roles_commute(par, acc) is True
    assert _roles_commute(acc, par) is True


def test_roles_commute_acc_acc_same_reduce_op_true() -> None:
    """Two ACCs with the same reduce_op commute (associative+commutative)."""
    from nkigym.tune.reorder_loops import _roles_commute

    a = LoopNode("d0", 4, AxisRole.ACCUMULATION, reduce_op="add")
    b = LoopNode("d1", 4, AxisRole.ACCUMULATION, reduce_op="add")
    assert _roles_commute(a, b) is True


def test_roles_commute_acc_acc_different_reduce_op_false() -> None:
    """ACCs with different reduce_ops do not commute."""
    from nkigym.tune.reorder_loops import _roles_commute

    a = LoopNode("d0", 4, AxisRole.ACCUMULATION, reduce_op="add")
    b = LoopNode("d1", 4, AxisRole.ACCUMULATION, reduce_op="max")
    assert _roles_commute(a, b) is False


def test_roles_commute_acc_acc_missing_reduce_op_false() -> None:
    """An ACC node with reduce_op=None cannot commute even with another ACC."""
    from nkigym.tune.reorder_loops import _roles_commute

    a = LoopNode("d0", 4, AxisRole.ACCUMULATION, reduce_op=None)
    b = LoopNode("d1", 4, AxisRole.ACCUMULATION, reduce_op="add")
    assert _roles_commute(a, b) is False


def test_roles_commute_any_sequential_false() -> None:
    """A SEQUENTIAL loop never commutes."""
    from nkigym.tune.reorder_loops import _roles_commute

    seq = LoopNode("d0", 4, AxisRole.SEQUENTIAL)
    par = LoopNode("d1", 4, AxisRole.PARALLEL)
    acc = LoopNode("d2", 4, AxisRole.ACCUMULATION, reduce_op="add")
    assert _roles_commute(seq, par) is False
    assert _roles_commute(par, seq) is False
    assert _roles_commute(seq, acc) is False
    assert _roles_commute(acc, seq) is False


def test_is_legal_accepts_par_par_cross_dim_perfect_pair() -> None:
    """Classic positive case: outer PAR loop with one PAR LoopNode child on a different dim."""
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner])
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is True


def test_is_legal_accepts_par_par_same_dim_perfect_pair() -> None:
    """Same-dim swap is not excluded — future tiles_per_block uses it."""
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    outer = LoopNode("d0", 16, AxisRole.PARALLEL, [inner])
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d0")
    assert atom.is_legal(None, forest) is True


def test_is_legal_accepts_par_acc_perfect_pair() -> None:
    """Mixed PAR × ACC is legal (ordering affects footprint, not correctness)."""
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d1", 4, AxisRole.ACCUMULATION, [BodyLeaf(op_idx=0)], reduce_op="add")
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner])
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is True


def test_is_legal_accepts_acc_acc_same_reduce_op() -> None:
    """ACC × ACC with matching reduce_op is legal."""
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d1", 4, AxisRole.ACCUMULATION, [BodyLeaf(op_idx=0)], reduce_op="add")
    outer = LoopNode("d0", 4, AxisRole.ACCUMULATION, [inner], reduce_op="add")
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is True


def test_is_legal_rejects_acc_acc_different_reduce_op() -> None:
    """Different reduce_ops do not commute."""
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d1", 4, AxisRole.ACCUMULATION, [BodyLeaf(op_idx=0)], reduce_op="max")
    outer = LoopNode("d0", 4, AxisRole.ACCUMULATION, [inner], reduce_op="add")
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_any_sequential() -> None:
    """SEQUENTIAL never commutes."""
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d1", 4, AxisRole.SEQUENTIAL, [BodyLeaf(op_idx=0)])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner])
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_stale_outer_dim() -> None:
    """outer_dim not matching the resolved node → False."""
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner])
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="dZZZ", inner_dim="d1")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_stale_inner_dim() -> None:
    """inner_dim not matching the child → False."""
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner])
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="dZZZ")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_out_of_range_path() -> None:
    """Path indexing past the forest root → False."""
    from nkigym.tune.reorder_loops import ReorderLoops

    forest = [LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])]
    atom = ReorderLoops(path=(99,), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_path_terminating_at_body_leaf() -> None:
    """Path whose last step lands on a BodyLeaf → False (not a LoopNode)."""
    from nkigym.tune.reorder_loops import ReorderLoops

    forest = [LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])]
    """path=(0, 0) lands on the BodyLeaf."""
    atom = ReorderLoops(path=(0, 0), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_outer_with_multiple_children() -> None:
    """Local imperfect-nest (>1 child) → False."""
    from nkigym.tune.reorder_loops import ReorderLoops

    child_a = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    child_b = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [child_a, child_b])
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_outer_whose_child_is_body_leaf() -> None:
    """Locally-perfect shape must have child = LoopNode, not BodyLeaf."""
    from nkigym.tune.reorder_loops import ReorderLoops

    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is False


def test_is_legal_rejects_empty_path() -> None:
    """Empty path has no target node → False."""
    from nkigym.tune.reorder_loops import ReorderLoops

    forest = [LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])]
    atom = ReorderLoops(path=(), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(None, forest) is False


def test_apply_swaps_outer_and_inner_at_forest_root() -> None:
    """Apply swaps outer ↔ inner; grandchildren pass through by reference."""
    from nkigym.tune.reorder_loops import ReorderLoops

    leaf = BodyLeaf(op_idx=0)
    inner = LoopNode("d1", 4, AxisRole.PARALLEL, [leaf])
    outer = LoopNode("d0", 8, AxisRole.PARALLEL, [inner])
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    _, new_forest = atom.apply(None, forest)
    assert len(new_forest) == 1
    new_outer = new_forest[0]
    assert isinstance(new_outer, LoopNode)
    assert new_outer.dim_id == "d1"
    assert new_outer.trip_count == 4
    assert new_outer.role is AxisRole.PARALLEL
    assert len(new_outer.children) == 1
    new_inner = new_outer.children[0]
    assert isinstance(new_inner, LoopNode)
    assert new_inner.dim_id == "d0"
    assert new_inner.trip_count == 8
    assert new_inner.role is AxisRole.PARALLEL
    assert new_inner.children[0] is leaf, "grandchildren subtree must be reference-equal"


def test_apply_preserves_reduce_op_across_swap() -> None:
    """reduce_op field travels with each node through the swap."""
    from nkigym.tune.reorder_loops import ReorderLoops

    inner = LoopNode("d1", 4, AxisRole.ACCUMULATION, [BodyLeaf(op_idx=0)], reduce_op="add")
    outer = LoopNode("d0", 4, AxisRole.ACCUMULATION, [inner], reduce_op="add")
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    _, new_forest = atom.apply(None, forest)
    new_outer = new_forest[0]
    assert isinstance(new_outer, LoopNode)
    assert new_outer.reduce_op == "add"
    new_inner = new_outer.children[0]
    assert isinstance(new_inner, LoopNode)
    assert new_inner.reduce_op == "add"


def test_apply_at_nested_path() -> None:
    """path=(0, 0) swaps the inner two of three nested loops, preserving the outermost."""
    from nkigym.tune.reorder_loops import ReorderLoops

    innermost = LoopNode("d2", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    middle = LoopNode("d1", 4, AxisRole.PARALLEL, [innermost])
    outermost = LoopNode("d0", 4, AxisRole.PARALLEL, [middle])
    forest = [outermost]
    atom = ReorderLoops(path=(0, 0), outer_dim="d1", inner_dim="d2")
    _, new_forest = atom.apply(None, forest)
    top = new_forest[0]
    assert isinstance(top, LoopNode)
    assert top.dim_id == "d0"
    new_middle = top.children[0]
    assert isinstance(new_middle, LoopNode)
    assert new_middle.dim_id == "d2"
    new_inner = new_middle.children[0]
    assert isinstance(new_inner, LoopNode)
    assert new_inner.dim_id == "d1"
    assert new_inner.children[0] is innermost.children[0]


def test_apply_is_self_inverse_by_structural_hash() -> None:
    """Applying the same atom twice produces a forest with the starting hash."""
    from nkigym.codegen.graph import OpGraph
    from nkigym.codegen.loop_forest import hash_state
    from nkigym.tune.reorder_loops import ReorderLoops

    op_graph = OpGraph(func_name="t", param_names=[], return_name="", tensors={}, dims={}, ops=[])
    inner = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    outer = LoopNode("d0", 8, AxisRole.PARALLEL, [inner])
    forest = [outer]
    first = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")
    _, f1 = first.apply(op_graph, forest)
    """After the first swap the outer dim is d1, inner dim is d0.
    To swap back we need a fresh atom that matches the new state."""
    reverse = ReorderLoops(path=(0,), outer_dim="d1", inner_dim="d0")
    _, f2 = reverse.apply(op_graph, f1)
    assert hash_state(op_graph, forest) == hash_state(op_graph, f2)


def test_apply_rmsnorm_matmul_preserves_check_invariant() -> None:
    """Reordering d0(T=16) ↔ d1(T=16) inside tensor_scalar's chain keeps invariant."""
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import build_canonical_forest, check_invariant, compute_phase_touched
    from nkigym.tune.reorder_loops import ReorderLoops

    @nkigym_kernel
    def rmm(lhs, rhs):
        lhs_sbuf = NKILoad()(data=lhs)
        rhs_sbuf = NKILoad()(data=rhs)
        sum_sq = NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf)
        rms_inv = NKIActivation(op="rsqrt", scale=1 / 256, bias=1e-6)(data=sum_sq)
        lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
        lhs_T = NKITranspose()(data=lhs_rms)
        prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
        out = NKIStore()(data=prod)
        return out

    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    g = parse_and_resolve(rmm, specs)
    forest = build_canonical_forest(g)
    """tensor_scalar is op index 4 (0=Load, 1=Load, 2=ActivationReduce, 3=Activation, 4=TensorScalar).
    Its tree shape is d0-block/d0-tile/d1-block/d1-tile/leaf. Swap d0-tile (at path (4, 0)) with d1-block."""
    atom = ReorderLoops(path=(4, 0), outer_dim="d0", inner_dim="d1")
    assert atom.is_legal(g, forest) is True
    _, new_forest = atom.apply(g, forest)
    num_tiles = {d: info.num_tiles for d, info in g.dims.items()}
    op_touched = {o.idx: o.touched_dims for o in g.ops}
    phase_touched = compute_phase_touched(g)
    check_invariant(new_forest, num_tiles, op_touched, phase_touched)


def test_enumerate_empty_forest_returns_empty_list() -> None:
    """Empty forest has no atoms."""
    from nkigym.tune.reorder_loops import enumerate_reorder_atoms

    assert enumerate_reorder_atoms([]) == []


def test_enumerate_single_loop_node_over_leaf_returns_empty() -> None:
    """No parent→child LoopNode pair present → no atoms."""
    from nkigym.tune.reorder_loops import enumerate_reorder_atoms

    forest = [LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])]
    assert enumerate_reorder_atoms(forest) == []


def test_enumerate_finds_par_par_pair_at_forest_root() -> None:
    """A single perfect parent→child LoopNode pair yields one atom."""
    from nkigym.tune.reorder_loops import ReorderLoops, enumerate_reorder_atoms

    inner = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner])
    forest = [outer]
    atoms = enumerate_reorder_atoms(forest)
    assert atoms == [ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1")]


def test_enumerate_finds_nested_chain_yields_multiple_atoms() -> None:
    """3-deep chain yields two atoms (parent-child pair at each non-leaf level)."""
    from nkigym.tune.reorder_loops import ReorderLoops, enumerate_reorder_atoms

    innermost = LoopNode("d2", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    middle = LoopNode("d1", 4, AxisRole.PARALLEL, [innermost])
    outermost = LoopNode("d0", 4, AxisRole.PARALLEL, [middle])
    forest = [outermost]
    atoms = enumerate_reorder_atoms(forest)
    assert ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d1") in atoms
    assert ReorderLoops(path=(0, 0), outer_dim="d1", inner_dim="d2") in atoms
    assert len(atoms) == 2


def test_enumerate_skips_imperfect_pairs() -> None:
    """Outer with >1 child does not contribute a reorder atom at that level."""
    from nkigym.tune.reorder_loops import enumerate_reorder_atoms

    child_a = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    child_b = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [child_a, child_b])
    forest = [outer]
    """No swap at path=(0,) is legal. child_a/child_b are each locally
    perfect over a BodyLeaf, so no further atoms below either."""
    atoms = enumerate_reorder_atoms(forest)
    assert atoms == []


def test_enumerate_skips_role_incompatible_pairs() -> None:
    """PAR×SEQ is skipped even when locally perfect."""
    from nkigym.tune.reorder_loops import enumerate_reorder_atoms

    inner = LoopNode("d1", 4, AxisRole.SEQUENTIAL, [BodyLeaf(op_idx=0)])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner])
    forest = [outer]
    assert enumerate_reorder_atoms(forest) == []


def test_enumerate_rmsnorm_matmul_canonical_finds_expected_atoms() -> None:
    """Canonical rmsnorm+matmul yields the expected reorder set.

    Every op-tree is a 2N-deep chain of same-dim block+tile pairs, so
    the enumerator finds ``(path(k), outer_dim=d, inner_dim=d)`` atoms
    at every non-leaf level where roles commute. Inside matmul the K
    sub-chain is ACC×ACC-same-reduce-op ('add') — those same-dim pairs
    are also legal.
    """
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import build_canonical_forest
    from nkigym.tune.reorder_loops import enumerate_reorder_atoms

    @nkigym_kernel
    def rmm(lhs, rhs):
        lhs_sbuf = NKILoad()(data=lhs)
        rhs_sbuf = NKILoad()(data=rhs)
        sum_sq = NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf)
        rms_inv = NKIActivation(op="rsqrt", scale=1 / 256, bias=1e-6)(data=sum_sq)
        lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
        lhs_T = NKITranspose()(data=lhs_rms)
        prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
        out = NKIStore()(data=prod)
        return out

    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    g = parse_and_resolve(rmm, specs)
    forest = build_canonical_forest(g)
    atoms = enumerate_reorder_atoms(forest)
    """Sanity: at least one atom exists in every 2D-or-deeper op's chain
    (load=0, rhs_load=1, tensor_scalar=4, transpose=5, store=7 are 2D
    ops; each has block→tile same-dim pair legal as PAR×PAR)."""
    paths = {a.path for a in atoms}
    assert (0,) in paths
    assert (1,) in paths
    assert (4,) in paths


def test_canonical_forest_populates_loop_node_names() -> None:
    """build_canonical_forest assigns ``name`` to every LoopNode outermost->innermost.

    For each dim, the k-th same-dim ancestor on the root-to-leaf path
    gets ``i_<dim_id>_<k>``. Verified on a bare load/store kernel whose
    load tree is a d0-block/d0-tile/d1-block/d1-tile chain.
    """
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import build_canonical_forest

    @nkigym_kernel
    def _big_load_kernel(x):
        xs = NKILoad()(data=x)
        out = NKIStore()(data=xs)
        return out

    """Shape (2048, 2048) gives both d0 and d1 num_tiles > 1 (with the
    128-tile partition size and unbounded free-axis default, d0 has 16
    tiles; d1's tile size defaults to its total size so d1 has 1 tile).
    That's enough to exercise the d0 block and tile tiers; d1 is a
    trivial chain but the naming scheme still applies."""
    specs = {"x": ((2048, 2048), "bfloat16")}
    g = parse_and_resolve(_big_load_kernel, specs)
    forest = build_canonical_forest(g)
    load_tree = forest[0]
    assert isinstance(load_tree, LoopNode)
    assert load_tree.name == "i_d0_0"
    inner1 = load_tree.children[0]
    assert isinstance(inner1, LoopNode)
    assert inner1.name == "i_d0_1"
    inner2 = inner1.children[0]
    assert isinstance(inner2, LoopNode)
    assert inner2.name == "i_d1_0"
    inner3 = inner2.children[0]
    assert isinstance(inner3, LoopNode)
    assert inner3.name == "i_d1_1"


def test_apply_preserves_loop_names_across_same_dim_swap() -> None:
    """Post-swap, each LoopNode keeps its original name — loop identity is stable.

    Hand-builds a two-level d0 chain where block has trip=16 and tile has
    trip=1 (the canonical 2N shape). After swapping, the loop that was
    named ``i_d0_0`` still prints that name, now at the deeper tree
    position; likewise ``i_d0_1`` moves to the outer slot.
    """
    from nkigym.tune.reorder_loops import ReorderLoops

    leaf = BodyLeaf(op_idx=0)
    inner = LoopNode("d0", 1, AxisRole.PARALLEL, [leaf], name="i_d0_1")
    outer = LoopNode("d0", 16, AxisRole.PARALLEL, [inner], name="i_d0_0")
    forest = [outer]
    atom = ReorderLoops(path=(0,), outer_dim="d0", inner_dim="d0")
    _, new_forest = atom.apply(None, forest)
    new_outer = new_forest[0]
    assert isinstance(new_outer, LoopNode)
    assert new_outer.name == "i_d0_1"
    assert new_outer.trip_count == 1
    new_inner = new_outer.children[0]
    assert isinstance(new_inner, LoopNode)
    assert new_inner.name == "i_d0_0"
    assert new_inner.trip_count == 16
