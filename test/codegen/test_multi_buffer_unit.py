"""Unit tests for MultiBuffer atom mechanics."""

from copy import deepcopy
from test.codegen._rmsnorm_matmul_fixture import INPUT_SPECS, f_nkigym

from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, build_canonical_forest
from nkigym.ops.base import AxisRole
from nkigym.tune.multi_buffer import MultiBuffer, enumerate_multi_buffer_atoms


def _fresh_state():
    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    forest = build_canonical_forest(graph)
    return graph, forest


def _fused_state():
    """Return a state where AR's reduce_close + activation's main share a d0 LoopNode."""
    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    ar_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    act_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivation")
    fused = LoopNode(
        dim_id="d0",
        trip_count=graph.dims["d0"].num_tiles,
        role=AxisRole.PARALLEL,
        children=[
            BodyLeaf(op_idx=graph.ops[ar_idx].idx, phase="reduce_close"),
            BodyLeaf(op_idx=graph.ops[act_idx].idx, phase="main"),
        ],
        name="i_d0_0_fused",
    )
    canonical = build_canonical_forest(graph)
    new_forest = [r for i, r in enumerate(canonical) if i not in (ar_idx, act_idx)]
    new_forest.insert(ar_idx, fused)
    return graph, new_forest, graph.ops[ar_idx].output_names[0]


def test_multi_buffer_illegal_on_missing_tensor() -> None:
    graph, forest = _fresh_state()
    atom = MultiBuffer(tensor_name="does_not_exist", dim_id="d0", degree=1)
    assert atom.is_legal(graph, forest) is False


def test_multi_buffer_illegal_on_missing_dim() -> None:
    graph, forest = _fresh_state()
    tname = next(t.name for t in graph.tensors.values() if t.origin == "intermediate")
    atom = MultiBuffer(tensor_name=tname, dim_id="no_such_dim", degree=1)
    assert atom.is_legal(graph, forest) is False


def test_multi_buffer_illegal_on_nonpositive_degree() -> None:
    graph, forest = _fresh_state()
    tname = next(t.name for t in graph.tensors.values() if t.origin == "intermediate")
    d = graph.tensors[tname].dim_ids[0]
    assert MultiBuffer(tensor_name=tname, dim_id=d, degree=0).is_legal(graph, forest) is False


def test_multi_buffer_illegal_when_degree_exceeds_lca_trip_product() -> None:
    """Cross-loopnest tensor has lca_trip_product=1; degree>1 is illegal."""
    graph, forest = _fresh_state()
    tname = next(t.name for t in graph.tensors.values() if t.origin == "intermediate")
    d = graph.tensors[tname].dim_ids[0]
    assert MultiBuffer(tensor_name=tname, dim_id=d, degree=2).is_legal(graph, forest) is False


def test_multi_buffer_legal_on_intra_loopnest_tensor() -> None:
    graph, forest, sum_sq = _fused_state()
    """lca_trip_product(d0) = 16, so degree in {1, 2, 4, 8, 16} is legal."""
    for degree in (1, 2, 4, 8, 16):
        atom = MultiBuffer(tensor_name=sum_sq, dim_id="d0", degree=degree)
        assert atom.is_legal(graph, forest), f"degree {degree} should be legal"
    """Non-divisors are illegal."""
    for degree in (3, 5, 7, 32):
        atom = MultiBuffer(tensor_name=sum_sq, dim_id="d0", degree=degree)
        assert not atom.is_legal(graph, forest), f"degree {degree} should be illegal"


def test_multi_buffer_apply_updates_only_targeted_tensor() -> None:
    graph, forest, sum_sq = _fused_state()
    before = deepcopy(graph)
    atom = MultiBuffer(tensor_name=sum_sq, dim_id="d0", degree=2)
    new_graph, new_forest = atom.apply(graph, forest)
    assert new_graph.tensors[sum_sq].buffer_degree["d0"] == 2
    for name, tensor in new_graph.tensors.items():
        if name == sum_sq:
            continue
        assert tensor.buffer_degree == before.tensors[name].buffer_degree
    assert new_forest is forest


def test_enumerate_multi_buffer_atoms_skips_cross_loopnest() -> None:
    """In the starting canonical forest every intermediate is cross-loopnest on every dim →
    no useful atoms emitted."""
    graph, forest = _fresh_state()
    atoms = enumerate_multi_buffer_atoms(graph, forest)
    assert atoms == [], f"Expected empty list; got {atoms}"


def test_enumerate_multi_buffer_atoms_finds_intra_loopnest() -> None:
    graph, forest, sum_sq = _fused_state()
    atoms = enumerate_multi_buffer_atoms(graph, forest)
    """Every divisor of 16 except 1 (current degree) for sum_sq on d0."""
    d0_atoms = [a for a in atoms if a.tensor_name == sum_sq and a.dim_id == "d0"]
    degrees = sorted(a.degree for a in d0_atoms)
    assert degrees == [2, 4, 8, 16], f"got degrees {degrees}"
