"""Golden data for graph search tests.

Hardcoded expected values for transform graph operations on a
256x256 matmul workload with seed=42.
"""

GRAPH_MATMUL_SEED_COUNT = 1
"""Number of seed nodes in the matmul transform graph."""

GRAPH_MATMUL_TARGET_5_MIN_NODES = 2
"""Minimum number of nodes after growing to target_size=5.

The graph always has at least 2 (1 seed + at least 1 valid
transform), though it may reach 5 depending on random state.
"""

GRAPH_MATMUL_TARGET_5_MAX_NODES = 5
"""Maximum number of nodes after growing to target_size=5."""

GRAPH_MATMUL_TRANSFORM_NAMES = ("seed", "reorder_loop", "change_placement", "change_blocking")
"""Valid transform names that can appear in the graph."""

GRAPH_MATMUL_DEFAULT_LOOP_ORDER = (("d1", 0), ("d3", 0), ("d0", 0))
"""Expected loop order of the default (seed) schedule for 256x256 matmul."""
