"""Tests for render-time derivations (required_tiles) and new atoms' emission."""

from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import build_canonical_forest


def test_required_tiles_cross_loopnest_returns_num_tiles() -> None:
    """Starting state: every intermediate is cross-loopnest, so required_tiles = num_tiles."""
    from test.codegen._rmsnorm_matmul_fixture import INPUT_SPECS, f_nkigym

    from nkigym.codegen.render import required_tiles

    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    forest = build_canonical_forest(graph)
    for tensor in graph.tensors.values():
        if tensor.origin != "intermediate":
            continue
        for d in tensor.dim_ids:
            want = graph.dims[d].num_tiles
            got = required_tiles(tensor, d, graph, forest)
            assert got == want, f"Tensor {tensor.name!r} dim {d!r}: required_tiles={got}, num_tiles={want}"


def test_required_tiles_intra_loopnest_returns_one() -> None:
    """After fusing AR + activation under a shared d0, sbuf_squared_sum's d0
    required_tiles drops to 1."""
    from test.codegen._rmsnorm_matmul_fixture import INPUT_SPECS, f_nkigym

    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode
    from nkigym.codegen.render import required_tiles
    from nkigym.ops.base import AxisRole

    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)

    """Hand-build a tiny forest with AR's reduce_close leaf and the
    activation leaf under the same d0 LoopNode. We only need these two
    ops under the shared d0; other ops can stay in their canonical
    loopnests in the forest to the side."""
    reduce_close_idx = next(op.idx for op in graph.ops if op.op_cls.__name__ == "NKIActivationReduce")
    activation_idx = next(op.idx for op in graph.ops if op.op_cls.__name__ == "NKIActivation")
    fused = LoopNode(
        dim_id="d0",
        trip_count=graph.dims["d0"].num_tiles,
        role=AxisRole.PARALLEL,
        children=[
            BodyLeaf(op_idx=reduce_close_idx, phase="reduce_close"),
            BodyLeaf(op_idx=activation_idx, phase="main"),
        ],
    )
    forest = [fused]

    sum_sq_name = graph.ops[reduce_close_idx].output_names[0]
    tensor = graph.tensors[sum_sq_name]
    got = required_tiles(tensor, "d0", graph, forest)
    assert got == 1, f"required_tiles(sum_sq, d0) = {got}, expected 1"


def test_sbuf_allocation_shrinks_after_fusion_of_ar_and_activation() -> None:
    """After fusing AR + activation under shared d0, sbuf_squared_sum allocates (128, 1, 1)."""
    from test.codegen._rmsnorm_matmul_fixture import INPUT_SPECS, f_nkigym

    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, build_canonical_forest
    from nkigym.codegen.render import render
    from nkigym.ops.base import AxisRole

    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    canonical = build_canonical_forest(graph)

    """Replace the AR tree and the activation tree in the canonical
    forest with a single fused d0 LoopNode that contains AR's
    reduce_close leaf + activation's main leaf directly. This is a
    synthetic forest for testing the derivation only — a proper fusion
    via FuseLoops atoms lands in Task 10."""
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
    """Splice: keep other roots, drop the AR and activation trees, add the fused node."""
    new_forest = []
    for idx, root in enumerate(canonical):
        if idx in (ar_idx, act_idx):
            continue
        new_forest.append(root)
    new_forest.insert(ar_idx, fused)

    sum_sq_name = graph.ops[ar_idx].output_names[0]
    src = render(graph, new_forest)
    expected_alloc = f"sbuf_{sum_sq_name} = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.sbuf)"
    assert expected_alloc in src, f"Did not find expected allocation; source:\n{src}"


def test_slot_expression_degenerates_to_zero_when_total_slots_equals_one() -> None:
    """Intra-loopnest tensor with buffer_degree=1: slot expression degrades to '0'."""
    from test.codegen._rmsnorm_matmul_fixture import INPUT_SPECS, f_nkigym

    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, build_canonical_forest
    from nkigym.codegen.render import render
    from nkigym.ops.base import AxisRole

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

    src = render(graph, new_forest)
    sum_sq_name = graph.ops[ar_idx].output_names[0]
    """sum_sq slot should index with a constant 0 at the P position, no modulo clutter."""
    assert (
        f"sbuf_{sum_sq_name}[0:128, 0, 0:1]" in src
    ), f"Expected collapsed slot 'sbuf_{sum_sq_name}[0:128, 0, 0:1]' in source; got:\n{src}"


def test_assign_stages_linear_chain() -> None:
    """Linear chain A->B->C under one LoopNode gives stages {A:0, B:1, C:2}."""
    from test.codegen._rmsnorm_matmul_fixture import INPUT_SPECS, f_nkigym

    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode
    from nkigym.codegen.render import assign_stages
    from nkigym.ops.base import AxisRole

    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    ar_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivationReduce")
    act_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKIActivation")
    ts_idx = next(i for i, op in enumerate(graph.ops) if op.op_cls.__name__ == "NKITensorScalar")
    chain = LoopNode(
        dim_id="d0",
        trip_count=16,
        role=AxisRole.PARALLEL,
        children=[
            BodyLeaf(op_idx=graph.ops[ar_idx].idx, phase="reduce_close"),
            BodyLeaf(op_idx=graph.ops[act_idx].idx, phase="main"),
            BodyLeaf(op_idx=graph.ops[ts_idx].idx, phase="main"),
        ],
    )
    stages = assign_stages(chain, graph.dep)
    assert stages[(graph.ops[ar_idx].idx, "reduce_close")] == 0
    assert stages[(graph.ops[act_idx].idx, "main")] == 1
    assert stages[(graph.ops[ts_idx].idx, "main")] == 2


def test_render_pipelined_loop_depth_two_emits_prologue_body_epilogue() -> None:
    """A d0 loop with pipeline_depth=2 and a legal buffer_degree emits 3 phases."""
    from test.codegen._rmsnorm_matmul_fixture import INPUT_SPECS, f_nkigym

    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, build_canonical_forest
    from nkigym.codegen.render import render
    from nkigym.ops.base import AxisRole

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
        pipeline_depth=2,
    )

    canonical = build_canonical_forest(graph)
    new_forest = [r for i, r in enumerate(canonical) if i not in (ar_idx, act_idx)]
    new_forest.insert(ar_idx, fused)

    """Before rendering we need enough buffer_degree for sum_sq to
    satisfy the skew legality (1 < required_tiles * buffer_degree).
    required_tiles is 1 inside the fused loop; buffer_degree must be >=
    2."""
    sum_sq_name = graph.ops[ar_idx].output_names[0]
    graph.tensors[sum_sq_name].buffer_degree["d0"] = 2

    src = render(graph, new_forest)
    """Prologue fires stage-0 (tensor_reduce) once at iter 0."""
    assert (
        "nisa.tensor_reduce(sbuf_sum_sq[0:128, (0) % 2, 0:1]" in src
    ), f"Prologue tensor_reduce(literal=0) not found; source:\n{src}"
    """Pipelined body spans iters [1, N)."""
    assert (
        f"for i_d0_0_fused in range(1, {graph.dims['d0'].num_tiles}):" in src
    ), f"Pipelined body for-header not found; source:\n{src}"
    """Epilogue fires stage-1 (activation) once at iter N-1 = 15."""
    assert (
        "nisa.activation(dst=sbuf_rms_inv[0:128, 15, 0:1]" in src
    ), f"Epilogue activation(literal=15) not found; source:\n{src}"
