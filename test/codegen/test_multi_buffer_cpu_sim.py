"""CPU-sim correctness tests for MultiBuffer + SoftwarePipeline.

Mirrors ``test_fuse_loops_cpu_sim.py``'s fp32 contract: rewrite
bf16/fp16 dtypes to fp32 in the rendered source, feed fp32 inputs,
assert numpy-golden match within fp32 tolerance.
"""

import numpy as np
import pytest

try:
    import nki
except ImportError:
    nki = None

from test.codegen._rmsnorm_matmul_fixture import INPUT_SPECS, f_nkigym, f_numpy

from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import BodyLeaf, LoopNode, build_canonical_forest
from nkigym.codegen.render import render
from nkigym.tune.fuse_loops import enumerate_fusion_atoms
from nkigym.tune.multi_buffer import MultiBuffer
from nkigym.tune.software_pipeline import SoftwarePipeline

_SEED = 0
_ATOL = 1e-3
_RTOL = 1e-3


def _cpu_sim(kernel_source: str, func_name: str, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Execute ``kernel_source`` under ``nki.simulate`` and return its output array.

    Matches the fp32 contract from ``nkigym.compile._cpu_sim_check``:
    rewrite bf16/fp16 dtypes to fp32 throughout the rendered source.
    """
    sim_source = kernel_source.replace("nl.bfloat16", "nl.float32").replace("nl.float16", "nl.float32")
    ns: dict = {}
    exec(sim_source, ns)  # noqa: S102
    actual = nki.simulate(ns[func_name])(**inputs)
    if isinstance(actual, tuple):
        actual = actual[0]
    return actual


def _fp32_inputs():
    """Return fp32-cast ``INPUT_SPECS`` tensors for the fp32 sim contract."""
    rng = np.random.default_rng(_SEED)
    return {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _) in INPUT_SPECS.items()}


def _subtree_has_all(node, op_idxs):
    """Return True when every op_idx in ``op_idxs`` has a BodyLeaf under ``node``."""
    found = set()

    def walk(n):
        """Recursive DFS accumulator."""
        if isinstance(n, BodyLeaf):
            if n.op_idx in op_idxs:
                found.add(n.op_idx)
        else:
            for c in n.children:
                walk(c)

    walk(node)
    return set(op_idxs).issubset(found)


def _find_lca_loop_path(forest, op_idxs):
    """Return the forest path of the shallowest LoopNode whose subtree contains every op_idx."""

    def find(node, path):
        """Return the deepest LoopNode path whose subtree still contains all target op_idxs."""
        if isinstance(node, BodyLeaf):
            return None
        for idx, child in enumerate(node.children):
            if not isinstance(child, LoopNode):
                continue
            if _subtree_has_all(child, op_idxs):
                deeper = find(child, path + (idx,))
                return deeper if deeper is not None else path + (idx,)
        return None

    for i, root in enumerate(forest):
        if _subtree_has_all(root, op_idxs):
            deeper = find(root, (i,))
            return deeper if deeper is not None else (i,)
    raise RuntimeError("no single LoopNode contains all target op indices")


def _build_fused_state():
    """Apply real ``FuseLoops`` atoms until AR and activation share a d0 LoopNode.

    The resulting fused LoopNode retains the canonical 2N form
    (Loop(d0, trip=N) with trip=1 child Loops wrapping inner
    non-pipelined loops). The MultiBuffer + SoftwarePipeline rewrites
    must handle this form.

    Returns:
        ``(graph, forest, loop_path, sum_sq_name)`` where ``loop_path``
        targets the fused d0 LoopNode that contains both AR and
        activation subtrees, and ``sum_sq_name`` is the AR output
        tensor (candidate for MultiBuffer).
    """
    graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    forest = build_canonical_forest(graph)
    ar_op_idx = next(op.idx for op in graph.ops if op.op_cls.__name__ == "NKIActivationReduce")
    act_op_idx = next(op.idx for op in graph.ops if op.op_cls.__name__ == "NKIActivation")
    target_ops = [ar_op_idx, act_op_idx]
    while not any(_subtree_has_all(r, target_ops) for r in forest):
        atoms = [a for a in enumerate_fusion_atoms(graph, forest) if a.dim_id == "d0"]
        if not atoms:
            raise RuntimeError("no d0 fusion atoms available but AR+activation still disjoint")
        _, forest = atoms[0].apply(graph, forest)
    loop_path = _find_lca_loop_path(forest, target_ops)
    sum_sq = graph.ops[ar_op_idx].output_names[0]
    return graph, forest, loop_path, sum_sq


@pytest.mark.skipif(nki is None, reason="nki runtime not available")
def test_cpu_sim_after_fusion_only() -> None:
    """Fused AR+activation state must produce a correct kernel."""
    graph, forest, _, _ = _build_fused_state()
    src = render(graph, forest)
    inputs = _fp32_inputs()
    actual = _cpu_sim(src, graph.func_name, inputs)
    expected = f_numpy(**inputs)
    assert np.allclose(
        actual, expected, atol=_ATOL, rtol=_RTOL
    ), f"CPU-sim mismatch after fusion only: max_abs_diff={float(np.abs(actual - expected).max()):.3e}"


@pytest.mark.skipif(nki is None, reason="nki runtime not available")
def test_cpu_sim_after_fusion_plus_multi_buffer() -> None:
    """Fused state + MultiBuffer(sum_sq, d0, 2) must still produce a correct kernel."""
    graph, forest, _, sum_sq = _build_fused_state()
    mb = MultiBuffer(tensor_name=sum_sq, dim_id="d0", degree=2)
    assert mb.is_legal(graph, forest)
    graph_after, forest_after = mb.apply(graph, forest)
    src = render(graph_after, forest_after)
    inputs = _fp32_inputs()
    actual = _cpu_sim(src, graph_after.func_name, inputs)
    expected = f_numpy(**inputs)
    assert np.allclose(
        actual, expected, atol=_ATOL, rtol=_RTOL
    ), f"CPU-sim mismatch after MultiBuffer: max_abs_diff={float(np.abs(actual - expected).max()):.3e}"


@pytest.mark.skipif(nki is None, reason="nki runtime not available")
def test_cpu_sim_after_fusion_plus_multi_buffer_plus_pipeline() -> None:
    """After fusion + MultiBuffer(degree>=chain_length) + SoftwarePipeline(depth=chain_length)."""
    from nkigym.codegen.loop_forest import _resolve_node
    from nkigym.codegen.render import assign_stages
    from nkigym.tune.multi_buffer import _divisors, _lca_trip_product

    graph, forest, loop_path, sum_sq = _build_fused_state()

    """Derive chain_length from the real fused LoopNode — this is the
    depth that covers every stage in the pipeline.
    """
    target_loop = _resolve_node(forest, loop_path)
    assert target_loop is not None, f"no LoopNode at {loop_path=}"
    stages = assign_stages(target_loop, graph.dep)
    chain_length = max(stages.values()) + 1

    """buffer_degree must satisfy: required_tiles * degree >= max_skew + 1
    (<= chain_length) AND degree must divide lca_trip_product. Pick the
    smallest divisor of lca_trip_product that is >= chain_length — this
    is the tightest legal degree that covers every tensor in the subtree
    at required_tiles == 1.
    """
    lca_prod = _lca_trip_product(sum_sq, "d0", graph, forest)
    legal_degrees = [d for d in _divisors(lca_prod) if d >= chain_length]
    assert legal_degrees, f"no divisor of {lca_prod} satisfies >= {chain_length}"
    degree = legal_degrees[0]
    mb = MultiBuffer(tensor_name=sum_sq, dim_id="d0", degree=degree)
    assert mb.is_legal(graph, forest), f"MultiBuffer(degree={degree}) not legal"
    graph_after, forest_after = mb.apply(graph, forest)

    sp = SoftwarePipeline(loop_path=loop_path, depth=chain_length)
    assert sp.is_legal(graph_after, forest_after), f"SoftwarePipeline(depth={chain_length}) not legal at {loop_path=}"
    graph_after2, forest_after2 = sp.apply(graph_after, forest_after)

    src = render(graph_after2, forest_after2)
    inputs = _fp32_inputs()
    actual = _cpu_sim(src, graph_after2.func_name, inputs)
    expected = f_numpy(**inputs)
    assert np.allclose(actual, expected, atol=_ATOL, rtol=_RTOL), (
        f"CPU-sim mismatch at chain_length={chain_length}, degree={degree}: "
        f"max_abs_diff={float(np.abs(actual - expected).max()):.3e}"
    )
