"""Tests for graph-based transform sampling search."""

import numpy as np
from golden.analyses import MATMUL_256_ANALYSIS, MATMUL_256_OP_CALLS, MATMUL_256_PARAMS
from golden.search_data import (
    GRAPH_MATMUL_DEFAULT_LOOP_ORDER,
    GRAPH_MATMUL_SEED_COUNT,
    GRAPH_MATMUL_TARGET_5_MAX_NODES,
    GRAPH_MATMUL_TARGET_5_MIN_NODES,
    GRAPH_MATMUL_TRANSFORM_NAMES,
)

from nkigym.codegen.passes import assign_passes
from nkigym.schedule.enumerate import default_schedule
from nkigym.schedule.render import render_schedule
from nkigym.search.graph import TransformGraph, _schedule_hash


def _make_graph() -> TransformGraph:
    """Create a TransformGraph for 256x256 matmul."""
    pa = assign_passes(MATMUL_256_OP_CALLS, MATMUL_256_ANALYSIS)
    return TransformGraph(
        analysis=MATMUL_256_ANALYSIS,
        op_calls=MATMUL_256_OP_CALLS,
        params=MATMUL_256_PARAMS,
        func_name="matmul_kernel",
        pa=pa,
    )


def _seed_graph(graph: TransformGraph) -> str:
    """Add default schedule as seed and return the kernel source."""
    pa = assign_passes(MATMUL_256_OP_CALLS, MATMUL_256_ANALYSIS)
    sched = default_schedule(MATMUL_256_ANALYSIS, MATMUL_256_OP_CALLS, MATMUL_256_PARAMS, pa.passes_per_dim)
    source = render_schedule(MATMUL_256_ANALYSIS, sched, MATMUL_256_OP_CALLS, MATMUL_256_PARAMS, "matmul_kernel", pa)
    graph.add_seed(source, sched)
    return source


def test_add_seed() -> None:
    """Adding a seed produces exactly 1 node."""
    graph = _make_graph()
    _seed_graph(graph)
    assert len(graph) == GRAPH_MATMUL_SEED_COUNT


def test_seed_loop_order() -> None:
    """Seed node schedule has the expected default loop order."""
    graph = _make_graph()
    _seed_graph(graph)
    seed_schedule = graph.schedules()[0]
    assert seed_schedule.loop_order == GRAPH_MATMUL_DEFAULT_LOOP_ORDER


def test_random_grow_adds_variants() -> None:
    """Growing to target_size=5 adds between 1 and 4 new variants."""
    graph = _make_graph()
    _seed_graph(graph)
    rng = np.random.default_rng(42)
    graph.random_grow(GRAPH_MATMUL_TARGET_5_MAX_NODES, rng)
    assert GRAPH_MATMUL_TARGET_5_MIN_NODES <= len(graph) <= GRAPH_MATMUL_TARGET_5_MAX_NODES


def test_all_variants_unique() -> None:
    """All variant kernel sources in the graph are unique."""
    graph = _make_graph()
    _seed_graph(graph)
    rng = np.random.default_rng(42)
    graph.random_grow(10, rng)
    sources = graph.variants()
    assert len(sources) == len(set(sources))


def test_transform_names_valid() -> None:
    """All nodes have valid transform names from the known set."""
    graph = _make_graph()
    _seed_graph(graph)
    rng = np.random.default_rng(42)
    graph.random_grow(10, rng)
    for sched in graph.schedules():
        """
        Schedules are just NamedTuples; the transform names are
        on the VariantNode objects, accessed via the internal list.
        """
    for node in graph._nodes:
        assert node.transform_name in GRAPH_MATMUL_TRANSFORM_NAMES


def test_schedule_hash_deterministic() -> None:
    """Same schedule produces the same hash."""
    pa = assign_passes(MATMUL_256_OP_CALLS, MATMUL_256_ANALYSIS)
    sched = default_schedule(MATMUL_256_ANALYSIS, MATMUL_256_OP_CALLS, MATMUL_256_PARAMS, pa.passes_per_dim)
    h1 = _schedule_hash(sched)
    h2 = _schedule_hash(sched)
    assert h1 == h2


def test_variants_returns_sources() -> None:
    """variants() returns kernel source strings."""
    graph = _make_graph()
    source = _seed_graph(graph)
    sources = graph.variants()
    assert len(sources) == 1
    assert sources[0] == source


def test_grow_respects_max_attempts() -> None:
    """Growth stops after max_attempts consecutive failures."""
    graph = _make_graph()
    _seed_graph(graph)
    rng = np.random.default_rng(42)
    graph.random_grow(1000, rng, max_attempts=5)
    assert len(graph) <= 1000


def test_cpu_simulation_valid() -> None:
    """All graph variants simulate correctly against numpy reference."""
    from nkigym.simulate import simulate_kernel

    graph = _make_graph()
    _seed_graph(graph)
    rng_graph = np.random.default_rng(42)
    graph.random_grow(5, rng_graph)
    a = np.random.RandomState(42).randn(256, 256)
    b = np.random.RandomState(43).randn(256, 256)
    expected = a.T @ b
    for source in graph.variants():
        actual = simulate_kernel(source, "matmul_kernel", {"a": a, "b": b})
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
