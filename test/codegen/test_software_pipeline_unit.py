"""Unit tests for SoftwarePipeline atom mechanics."""

from test.codegen._rmsnorm_matmul_fixture import INPUT_SPECS, f_nkigym

from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, build_canonical_forest
from nkigym.ops.base import AxisRole
from nkigym.tune.software_pipeline import SoftwarePipeline, enumerate_software_pipeline_atoms


def _fused_state():
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
    return graph, new_forest, ar_idx, graph.ops[ar_idx].output_names[0]


def test_software_pipeline_illegal_on_non_looppath() -> None:
    graph, forest, _, _ = _fused_state()
    atom = SoftwarePipeline(loop_path=(), depth=2)
    assert atom.is_legal(graph, forest) is False


def test_software_pipeline_illegal_on_depth_one_without_prior_depth() -> None:
    graph, forest, ar_idx, _ = _fused_state()
    atom = SoftwarePipeline(loop_path=(ar_idx,), depth=1)
    """Self-move when current pipeline_depth already 1."""
    assert atom.is_legal(graph, forest) is False


def test_software_pipeline_illegal_when_buffer_degree_insufficient() -> None:
    graph, forest, ar_idx, _ = _fused_state()
    atom = SoftwarePipeline(loop_path=(ar_idx,), depth=2)
    """sum_sq buffer_degree default is 1, required_tiles=1 → total_slots=1 < skew+1=2."""
    assert atom.is_legal(graph, forest) is False


def test_software_pipeline_legal_when_buffer_degree_sufficient() -> None:
    graph, forest, ar_idx, sum_sq = _fused_state()
    graph.tensors[sum_sq].buffer_degree["d0"] = 2
    atom = SoftwarePipeline(loop_path=(ar_idx,), depth=2)
    assert atom.is_legal(graph, forest) is True


def test_software_pipeline_apply_sets_pipeline_depth() -> None:
    graph, forest, ar_idx, sum_sq = _fused_state()
    graph.tensors[sum_sq].buffer_degree["d0"] = 2
    atom = SoftwarePipeline(loop_path=(ar_idx,), depth=2)
    _, new_forest = atom.apply(graph, forest)
    updated = new_forest[ar_idx]
    assert isinstance(updated, LoopNode)
    assert updated.pipeline_depth == 2
    assert updated.dim_id == "d0"
    assert updated.trip_count == graph.dims["d0"].num_tiles
    assert updated.name == "i_d0_0_fused"


def test_enumerate_software_pipeline_atoms_respects_chain_length() -> None:
    graph, forest, ar_idx, sum_sq = _fused_state()
    graph.tensors[sum_sq].buffer_degree["d0"] = 16
    atoms = enumerate_software_pipeline_atoms(graph, forest)
    """Depth can be in {2} for a 2-stage chain, excluding current depth=1."""
    d0_atoms = [a for a in atoms if a.loop_path == (ar_idx,)]
    depths = sorted(a.depth for a in d0_atoms)
    assert depths == [2], f"got depths {depths}"


def test_software_pipeline_illegal_when_depth_less_than_chain_length() -> None:
    """Pipeline must cover every stage — depth < chain_length is illegal even if buffer_degree suffices."""
    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    ar_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    act_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivation")
    ts_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKITensorScalar")
    """Build a 3-stage chain (AR -> activation -> tensor_scalar), give all
    intermediates enough buffer_degree, and assert depth=2 is illegal."""
    chain = LoopNode(
        dim_id="d0",
        trip_count=graph.dims["d0"].num_tiles,
        role=AxisRole.PARALLEL,
        children=[
            BodyLeaf(op_idx=graph.ops[ar_idx].idx, phase="reduce_close"),
            BodyLeaf(op_idx=graph.ops[act_idx].idx, phase="main"),
            BodyLeaf(op_idx=graph.ops[ts_idx].idx, phase="main"),
        ],
    )
    canonical = build_canonical_forest(graph)
    new_forest = [r for i, r in enumerate(canonical) if i not in (ar_idx, act_idx, ts_idx)]
    new_forest.insert(ar_idx, chain)
    """Give every intermediate enough slots."""
    for t in graph.tensors.values():
        if "d0" in t.dim_ids:
            t.buffer_degree["d0"] = graph.dims["d0"].num_tiles

    atom_shallow = SoftwarePipeline(loop_path=(ar_idx,), depth=2)
    assert atom_shallow.is_legal(graph, new_forest) is False, "depth=2 on 3-stage chain must be illegal"
    atom_exact = SoftwarePipeline(loop_path=(ar_idx,), depth=3)
    assert atom_exact.is_legal(graph, new_forest) is True, "depth=3 on 3-stage chain must be legal"
