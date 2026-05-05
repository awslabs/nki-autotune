"""Layer-2 tests: LoopForest IR, canonical forest, invariant."""

from nkigym.codegen.loop_forest import BodyLeaf, LoopForest, LoopNode, check_invariant
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose


def test_body_leaf_defaults_phase_to_main() -> None:
    """BodyLeaf defaults to phase='main' for single-phase ops."""
    leaf = BodyLeaf(op_idx=0)
    assert leaf.phase == "main"


def test_body_leaf_accepts_explicit_phase() -> None:
    """Multi-phase ops name their phase explicitly."""
    leaf = BodyLeaf(op_idx=3, phase="psum_init")
    assert leaf.phase == "psum_init"


def test_loop_node_stores_dim_trip_role_and_children() -> None:
    """LoopNode exposes dim_id, trip_count, role, and children."""
    node = LoopNode(dim_id="d0", trip_count=16, role=AxisRole.PARALLEL, children=[BodyLeaf(op_idx=0)])
    assert node.dim_id == "d0"
    assert node.trip_count == 16
    assert node.role is AxisRole.PARALLEL
    assert len(node.children) == 1


def test_check_invariant_passes_on_simple_2n_shape() -> None:
    """A 2-level block+tile chain (16 * 1 = 16) satisfies the invariant."""
    tree: LoopNode = LoopNode("d0", 16, AxisRole.PARALLEL, [LoopNode("d0", 1, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])])
    forest: LoopForest = [tree]
    num_tiles = {"d0": 16}
    op_touched = {0: ("d0",)}
    check_invariant(forest, num_tiles, op_touched)


def test_check_invariant_raises_on_product_mismatch() -> None:
    """A chain where product of same-dim trips != num_tiles(d) fails the invariant."""
    tree: LoopNode = LoopNode("d0", 8, AxisRole.PARALLEL, [LoopNode("d0", 1, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])])
    forest: LoopForest = [tree]
    num_tiles = {"d0": 16}
    op_touched = {0: ("d0",)}
    try:
        check_invariant(forest, num_tiles, op_touched)
    except ValueError as exc:
        assert "d0" in str(exc)
    else:
        raise AssertionError("check_invariant did not raise on product mismatch")


def test_check_invariant_raises_when_dim_missing_from_ancestors() -> None:
    """A BodyLeaf whose op references a dim with no ancestor LoopNodes fails."""
    tree: LoopNode = LoopNode("d0", 16, AxisRole.PARALLEL, [LoopNode("d0", 1, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])])
    forest: LoopForest = [tree]
    num_tiles = {"d0": 16, "d1": 4}
    op_touched = {0: ("d0", "d1")}
    try:
        check_invariant(forest, num_tiles, op_touched)
    except ValueError as exc:
        assert "d1" in str(exc)
    else:
        raise AssertionError("check_invariant did not raise on missing dim")


def _parse(kernel, specs):
    from nkigym.codegen.graph import parse_and_resolve

    return parse_and_resolve(kernel, specs)


@nkigym_kernel
def _load_store_kernel(x):
    """Test kernel: simple load + store."""
    y = NKILoad()(data=x)
    out = NKIStore()(data=y)
    return out


@nkigym_kernel
def _matmul_lhsT_rhs_for_forest_tests(lhs_T, rhs):
    """Test kernel: matmul lhs_T @ rhs."""
    lhs_T_sbuf = NKILoad()(data=lhs_T)
    rhs_sbuf = NKILoad()(data=rhs)
    prod = NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


@nkigym_kernel
def _rms_kernel_with_post_op(x):
    """Test kernel: activation_reduce with post_op=rsqrt."""
    xs = NKILoad()(data=x)
    m = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt")(data=xs)
    out = NKIStore()(data=m)
    return out


@nkigym_kernel
def _rms_kernel_without_post_op(x):
    """Test kernel: activation_reduce with no post_op."""
    xs = NKILoad()(data=x)
    m = NKIActivationReduce(op="square", reduce_op="add")(data=xs)
    out = NKIStore()(data=m)
    return out


EPS = 1e-6


@nkigym_kernel
def _rmsnorm_matmul(lhs, rhs):
    """Full rmsnorm+matmul DAG for invariant coverage."""
    lhs_sbuf = NKILoad()(data=lhs)
    rhs_sbuf = NKILoad()(data=rhs)
    rms_inv = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt")(data=lhs_sbuf)
    lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
    lhs_T = NKITranspose()(data=lhs_rms)
    prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


_MATMUL_SPECS = {"lhs_T": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")}
_RMS_SPECS = {"x": ((128, 256), "bfloat16")}
_RMSNORM_MATMUL_SPECS = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}


def test_canonical_forest_load_kernel_shape() -> None:
    """A 2D NKILoad op produces a 4-deep chain: d0 block / d0 tile / d1 block / d1 tile / BodyLeaf."""
    from nkigym.codegen.loop_forest import build_canonical_forest

    specs = {"x": ((128, 256), "bfloat16")}
    g = _parse(_load_store_kernel, specs)
    forest = build_canonical_forest(g)
    """Two roots — one per op."""
    assert len(forest) == 2
    load_tree = forest[0]
    assert isinstance(load_tree, LoopNode)
    assert load_tree.dim_id == "d0"
    assert load_tree.trip_count == g.dims["d0"].num_tiles
    assert load_tree.role is AxisRole.PARALLEL
    inner_d0 = load_tree.children[0]
    assert isinstance(inner_d0, LoopNode)
    assert inner_d0.dim_id == "d0"
    assert inner_d0.trip_count == 1
    outer_d1 = inner_d0.children[0]
    assert isinstance(outer_d1, LoopNode)
    assert outer_d1.dim_id == "d1"
    assert outer_d1.trip_count == g.dims["d1"].num_tiles
    inner_d1 = outer_d1.children[0]
    assert isinstance(inner_d1, LoopNode)
    assert inner_d1.dim_id == "d1"
    assert inner_d1.trip_count == 1
    leaf = inner_d1.children[0]
    assert isinstance(leaf, BodyLeaf)
    assert leaf.op_idx == 0
    assert leaf.phase == "main"


def test_canonical_forest_invariant_holds_on_load_store() -> None:
    """check_invariant passes on the canonical forest of a load+store kernel."""
    from nkigym.codegen.loop_forest import build_canonical_forest

    specs = {"x": ((128, 256), "bfloat16")}
    g = _parse(_load_store_kernel, specs)
    forest = build_canonical_forest(g)
    num_tiles = {d: info.num_tiles for d, info in g.dims.items()}
    op_touched = {o.idx: o.touched_dims for o in g.ops}
    check_invariant(forest, num_tiles, op_touched)


def test_canonical_forest_matmul_has_three_phase_leaves() -> None:
    """Matmul's innermost N-tile node contains psum_init, K chain ending in compute, drain."""
    from nkigym.codegen.loop_forest import build_canonical_forest

    g = _parse(_matmul_lhsT_rhs_for_forest_tests, _MATMUL_SPECS)
    forest = build_canonical_forest(g)
    matmul_idx = next(i for i, op in enumerate(g.ops) if op.op_cls.__name__ == "NKIMatmul")
    tree = forest[matmul_idx]
    """Drill in: M-block, M-tile, N-block, N-tile."""
    n_tile = tree.children[0].children[0].children[0]
    children = n_tile.children
    assert isinstance(children[0], BodyLeaf) and children[0].phase == "psum_init"
    """Children[1] is the K chain (block -> tile -> compute leaf)."""
    k_block = children[1]
    assert isinstance(k_block, LoopNode) and k_block.trip_count > 1
    k_tile = k_block.children[0]
    compute_leaf = k_tile.children[0]
    assert isinstance(compute_leaf, BodyLeaf) and compute_leaf.phase == "compute"
    """Children[2] is the drain leaf at the same depth as psum_init."""
    assert isinstance(children[2], BodyLeaf) and children[2].phase == "drain"


def test_canonical_forest_matmul_invariant_holds() -> None:
    """Matmul's canonical forest satisfies the per-dim product invariant."""
    from nkigym.codegen.loop_forest import build_canonical_forest, compute_phase_touched

    g = _parse(_matmul_lhsT_rhs_for_forest_tests, _MATMUL_SPECS)
    forest = build_canonical_forest(g)
    num_tiles = {d: info.num_tiles for d, info in g.dims.items()}
    op_touched = {o.idx: o.touched_dims for o in g.ops}
    phase_touched = compute_phase_touched(g)
    check_invariant(forest, num_tiles, op_touched, phase_touched)


def test_canonical_forest_activation_reduce_with_post_op_has_three_leaves() -> None:
    """ActivationReduce with post_op emits reducer_init + F chain ending in reduce_step + post_op."""
    from nkigym.codegen.loop_forest import build_canonical_forest

    g = _parse(_rms_kernel_with_post_op, _RMS_SPECS)
    forest = build_canonical_forest(g)
    ar_idx = next(i for i, op in enumerate(g.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    tree = forest[ar_idx]
    p_tile = tree.children[0]
    children = p_tile.children
    """Three children: reducer_init, F chain, post_op."""
    assert isinstance(children[0], BodyLeaf) and children[0].phase == "reducer_init"
    f_block = children[1]
    assert isinstance(f_block, LoopNode)
    f_tile = f_block.children[0]
    reduce_leaf = f_tile.children[0]
    assert isinstance(reduce_leaf, BodyLeaf) and reduce_leaf.phase == "reduce_step"
    assert isinstance(children[2], BodyLeaf) and children[2].phase == "post_op"


def test_canonical_forest_activation_reduce_no_post_op_omits_post_op_leaf() -> None:
    """When the op has no post_op, the post_op leaf is omitted."""
    from nkigym.codegen.loop_forest import build_canonical_forest

    g = _parse(_rms_kernel_without_post_op, _RMS_SPECS)
    forest = build_canonical_forest(g)
    ar_idx = next(i for i, op in enumerate(g.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    tree = forest[ar_idx]
    p_tile = tree.children[0]
    children = p_tile.children
    """Two children: reducer_init, F chain. No post_op."""
    assert len(children) == 2
    assert isinstance(children[0], BodyLeaf) and children[0].phase == "reducer_init"
    assert isinstance(children[1], LoopNode)
    for c in children:
        if isinstance(c, BodyLeaf):
            assert c.phase != "post_op"


def test_canonical_forest_rmsnorm_matmul_invariant_holds() -> None:
    """The full rmsnorm+matmul canonical forest satisfies the invariant."""
    from nkigym.codegen.loop_forest import build_canonical_forest, compute_phase_touched

    g = _parse(_rmsnorm_matmul, _RMSNORM_MATMUL_SPECS)
    forest = build_canonical_forest(g)
    num_tiles = {d: info.num_tiles for d, info in g.dims.items()}
    op_touched = {o.idx: o.touched_dims for o in g.ops}
    phase_touched = compute_phase_touched(g)
    check_invariant(forest, num_tiles, op_touched, phase_touched)


def test_loop_node_reduce_op_defaults_to_none() -> None:
    """LoopNode without an explicit reduce_op defaults to None."""
    from nkigym.codegen.loop_forest import LoopNode
    from nkigym.ops.base import AxisRole

    node = LoopNode(dim_id="d0", trip_count=4, role=AxisRole.PARALLEL)
    assert node.reduce_op is None


def test_loop_node_reduce_op_accepts_explicit_value() -> None:
    """LoopNode can be constructed with reduce_op='add' for ACC loops."""
    from nkigym.codegen.loop_forest import LoopNode
    from nkigym.ops.base import AxisRole

    node = LoopNode(dim_id="d0", trip_count=4, role=AxisRole.ACCUMULATION, reduce_op="add")
    assert node.reduce_op == "add"


def test_canonical_forest_matmul_k_loops_carry_reduce_op_add() -> None:
    """Matmul's K-chain LoopNodes (block + tile) carry reduce_op='add'."""
    from nkigym.codegen.loop_forest import LoopNode, build_canonical_forest

    g = _parse(_matmul_lhsT_rhs_for_forest_tests, _MATMUL_SPECS)
    forest = build_canonical_forest(g)
    matmul_idx = next(i for i, op in enumerate(g.ops) if op.op_cls.__name__ == "NKIMatmul")
    tree = forest[matmul_idx]
    n_tile = tree.children[0].children[0].children[0]
    k_block = n_tile.children[1]
    assert isinstance(k_block, LoopNode)
    assert k_block.reduce_op == "add"
    k_tile = k_block.children[0]
    assert isinstance(k_tile, LoopNode)
    assert k_tile.reduce_op == "add"


def test_canonical_forest_matmul_m_n_loops_have_no_reduce_op() -> None:
    """Matmul's M and N loops are PARALLEL and carry reduce_op=None."""
    from nkigym.codegen.loop_forest import LoopNode, build_canonical_forest

    g = _parse(_matmul_lhsT_rhs_for_forest_tests, _MATMUL_SPECS)
    forest = build_canonical_forest(g)
    matmul_idx = next(i for i, op in enumerate(g.ops) if op.op_cls.__name__ == "NKIMatmul")
    tree = forest[matmul_idx]
    m_block = tree
    m_tile = m_block.children[0]
    n_block = m_tile.children[0]
    n_tile = n_block.children[0]
    for node in (m_block, m_tile, n_block, n_tile):
        assert isinstance(node, LoopNode)
        assert node.reduce_op is None


def test_canonical_forest_activation_reduce_f_loops_carry_kwargs_reduce_op() -> None:
    """ActivationReduce F-chain LoopNodes carry reduce_op from op_kwargs."""
    from nkigym.codegen.loop_forest import LoopNode, build_canonical_forest

    g = _parse(_rms_kernel_with_post_op, _RMS_SPECS)
    forest = build_canonical_forest(g)
    ar_idx = next(i for i, op in enumerate(g.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    tree = forest[ar_idx]
    p_tile = tree.children[0]
    f_block = p_tile.children[1]
    assert isinstance(f_block, LoopNode)
    assert f_block.reduce_op == "add"
    f_tile = f_block.children[0]
    assert isinstance(f_tile, LoopNode)
    assert f_tile.reduce_op == "add"


def test_hash_forest_is_deterministic_across_invocations() -> None:
    """Two calls on the same forest produce the same hash."""
    from nkigym.codegen.loop_forest import build_canonical_forest, hash_forest

    g = _parse(_rmsnorm_matmul, _RMSNORM_MATMUL_SPECS)
    forest = build_canonical_forest(g)
    assert hash_forest(forest) == hash_forest(forest)


def test_hash_forest_equals_for_independently_built_canonical_forests() -> None:
    """Two independent canonical forests of the same op_graph hash equal."""
    from nkigym.codegen.loop_forest import build_canonical_forest, hash_forest

    g = _parse(_rmsnorm_matmul, _RMSNORM_MATMUL_SPECS)
    f1 = build_canonical_forest(g)
    f2 = build_canonical_forest(g)
    assert hash_forest(f1) == hash_forest(f2)


def test_hash_forest_distinguishes_trip_count_change() -> None:
    """Forests differing only in a trip count hash differently."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, hash_forest
    from nkigym.ops.base import AxisRole

    f1 = [LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])]
    f2 = [LoopNode("d0", 8, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])]
    assert hash_forest(f1) != hash_forest(f2)


def test_hash_forest_distinguishes_dim_id_change() -> None:
    """Forests differing only in dim_id at a node hash differently."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, hash_forest
    from nkigym.ops.base import AxisRole

    f1 = [LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])]
    f2 = [LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])]
    assert hash_forest(f1) != hash_forest(f2)


def test_hash_forest_distinguishes_reduce_op_change() -> None:
    """Forests differing only in reduce_op hash differently."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, hash_forest
    from nkigym.ops.base import AxisRole

    f1 = [LoopNode("d0", 4, AxisRole.ACCUMULATION, [BodyLeaf(op_idx=0)], reduce_op="add")]
    f2 = [LoopNode("d0", 4, AxisRole.ACCUMULATION, [BodyLeaf(op_idx=0)], reduce_op="max")]
    assert hash_forest(f1) != hash_forest(f2)


def test_hash_forest_distinguishes_leaf_phase_change() -> None:
    """Leaves differing only in phase hash differently."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, hash_forest
    from nkigym.ops.base import AxisRole

    f1 = [LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0, phase="a")])]
    f2 = [LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0, phase="b")])]
    assert hash_forest(f1) != hash_forest(f2)


def test_resolve_node_returns_forest_root_entry_for_single_index_path() -> None:
    """path=(0,) returns forest[0]."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, _resolve_node
    from nkigym.ops.base import AxisRole

    node_a = LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])
    node_b = LoopNode("d1", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=1)])
    forest = [node_a, node_b]
    assert _resolve_node(forest, (0,)) is node_a
    assert _resolve_node(forest, (1,)) is node_b


def test_resolve_node_walks_nested_children() -> None:
    """path=(0, 0) returns forest[0].children[0]."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, _resolve_node
    from nkigym.ops.base import AxisRole

    leaf = BodyLeaf(op_idx=0)
    inner = LoopNode("d1", 1, AxisRole.PARALLEL, [leaf])
    outer = LoopNode("d0", 4, AxisRole.PARALLEL, [inner])
    forest = [outer]
    assert _resolve_node(forest, (0, 0)) is inner
    assert _resolve_node(forest, (0, 0, 0)) is leaf


def test_resolve_node_returns_none_for_empty_path() -> None:
    """Empty path is invalid (no target node)."""
    from nkigym.codegen.loop_forest import _resolve_node

    assert _resolve_node([], ()) is None


def test_resolve_node_returns_none_for_out_of_range_index() -> None:
    """Path index beyond children length returns None."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, _resolve_node
    from nkigym.ops.base import AxisRole

    forest = [LoopNode("d0", 4, AxisRole.PARALLEL, [BodyLeaf(op_idx=0)])]
    assert _resolve_node(forest, (99,)) is None
    assert _resolve_node(forest, (0, 99)) is None


def test_resolve_node_returns_none_when_traversing_through_body_leaf() -> None:
    """Can't descend into a BodyLeaf's 'children'; returns None."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, _resolve_node
    from nkigym.ops.base import AxisRole

    leaf = BodyLeaf(op_idx=0)
    forest = [LoopNode("d0", 4, AxisRole.PARALLEL, [leaf])]
    """path=(0, 0, 0) would try to descend into leaf.children."""
    assert _resolve_node(forest, (0, 0, 0)) is None
