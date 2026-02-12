"""Tests for transform graph search module.

Tests search() with exhaustive (num_targets=inf) and targeted
(num_targets=N) modes using the original matmul function.

Run with: pytest test/test_search.py -v
"""

import math
from pathlib import Path

import numpy as np
from conftest import assert_arrays_close, make_random_array

import nkigym
from nkigym.ir import Program, ir_to_callable, ir_to_source
from nkigym.search import search
from nkigym.transforms import DataReuseTransform, OperandMergeTransform


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute matrix multiplication."""
    return nkigym.nc_matmul(a, b)


_KERNEL_KWARGS = {
    "1x1": {"a": make_random_array((128, 128), seed=10), "b": make_random_array((128, 128), seed=11)},
    "2x1": {"a": make_random_array((128, 256), seed=10), "b": make_random_array((128, 128), seed=11)},
    "2x2": {"a": make_random_array((128, 256), seed=10), "b": make_random_array((128, 256), seed=11)},
    "1x2": {"a": make_random_array((128, 128), seed=10), "b": make_random_array((128, 256), seed=11)},
}


def _get_unique_sources(programs: list[Program]) -> set[str]:
    """Get the set of unique sources from a list of programs.

    Args:
        programs: List of Program tuples.

    Returns:
        Set of source strings.
    """
    return {ir_to_source(p) for p in programs}


class TestExhaustiveSearch:
    """Tests for search() in exhaustive mode (num_targets=inf)."""

    def test_no_opportunities_returns_root(self, tmp_path: Path) -> None:
        """When no transforms have opportunities, return [root program].

        A 1x1 matmul has a single subgraph so DataReuseTransform
        finds nothing to merge.
        """
        transforms = [DataReuseTransform()]
        leaves = search(_matmul, transforms, math.inf, None, 0, tmp_path, _KERNEL_KWARGS["1x1"])

        assert len(leaves) == 1

    def test_leaves_are_unique_by_source(self, tmp_path: Path) -> None:
        """All returned leaves have distinct source code.

        Uses 2x2 matmul with DataReuseTransform which has
        multiple convergent paths.
        """
        transforms = [DataReuseTransform()]
        leaves = search(_matmul, transforms, math.inf, None, 0, tmp_path, _KERNEL_KWARGS["2x2"])

        sources = [ir_to_source(p) for p in leaves]
        assert len(sources) == len(set(sources))

    def test_numerical_correctness(self, tmp_path: Path) -> None:
        """All leaves preserve numerical correctness.

        Applies exhaustive search to 2x2 matmul with DataReuseTransform,
        then verifies each leaf produces the same output as the original.
        """
        transforms = [DataReuseTransform()]
        leaves = search(_matmul, transforms, math.inf, None, 0, tmp_path, _KERNEL_KWARGS["2x2"])

        a = make_random_array((128, 256), seed=42)
        b = make_random_array((128, 256), seed=43)
        expected = _matmul(a, b)

        for leaf in leaves:
            actual = ir_to_callable(leaf)(a, b)
            assert_arrays_close(actual, expected)

    def test_dedup_prunes_convergent_paths(self, tmp_path: Path) -> None:
        """Source-based dedup prunes states reached via different orderings.

        A 2x2 matmul has 4 merge pairs. Different orderings can reach
        the same merged state. Without dedup the number of visited nodes
        would be larger.
        """
        transforms = [DataReuseTransform()]
        leaves = search(_matmul, transforms, math.inf, None, 0, tmp_path, _KERNEL_KWARGS["2x2"])

        sources = _get_unique_sources(leaves)
        assert len(sources) == len(leaves)

    def test_single_transform_single_opportunity(self, tmp_path: Path) -> None:
        """Single transform with a single opportunity produces root and transformed node.

        A 2x1 matmul with DataReuseTransform has exactly one pair,
        yielding two nodes: the root and the transformed state.
        """
        transforms = [DataReuseTransform()]
        results = search(_matmul, transforms, math.inf, None, 0, tmp_path, _KERNEL_KWARGS["2x1"])

        assert len(results) == 2

    def test_multiple_transforms(self, tmp_path: Path) -> None:
        """Exhaustive search with both DataReuse and OperandMerge transforms.

        Uses 1x2 matmul which has data reuse opportunities,
        and the post-reuse state has operand merge opportunities.
        Returns all unique nodes including intermediate states.
        """
        transforms = [DataReuseTransform(), OperandMergeTransform()]
        results = search(_matmul, transforms, math.inf, None, 0, tmp_path, _KERNEL_KWARGS["1x2"])

        assert len(results) >= 1
        sources = _get_unique_sources(results)
        assert len(sources) == len(results)

    def test_multiple_transforms_numerical_correctness(self, tmp_path: Path) -> None:
        """Leaves from multi-transform search preserve numerical correctness."""
        transforms = [DataReuseTransform(), OperandMergeTransform()]
        leaves = search(_matmul, transforms, math.inf, None, 0, tmp_path, _KERNEL_KWARGS["1x2"])

        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 256), seed=43)
        expected = _matmul(a, b)

        for leaf in leaves:
            actual = ir_to_callable(leaf)(a, b)
            assert_arrays_close(actual, expected)

    def test_operand_merge_only(self, tmp_path: Path) -> None:
        """Exhaustive search with OperandMergeTransform on 1x2 matmul."""
        transforms = [OperandMergeTransform()]
        leaves = search(_matmul, transforms, math.inf, None, 0, tmp_path, _KERNEL_KWARGS["1x2"])

        assert len(leaves) >= 1

        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 256), seed=43)
        expected = _matmul(a, b)

        for leaf in leaves:
            actual = ir_to_callable(leaf)(a, b)
            assert_arrays_close(actual, expected)


class TestTargetedSearch:
    """Tests for search() with finite num_targets."""

    def test_seed_reproducibility(self, tmp_path: Path) -> None:
        """Same seed produces identical results across runs."""
        transforms = [DataReuseTransform()]

        result1 = search(_matmul, transforms, 5, 42, 0, tmp_path, _KERNEL_KWARGS["2x2"])
        result2 = search(_matmul, transforms, 5, 42, 0, tmp_path, _KERNEL_KWARGS["2x2"])

        sources1 = [ir_to_source(p) for p in result1]
        sources2 = [ir_to_source(p) for p in result2]
        assert sources1 == sources2

    def test_unique_dedup(self, tmp_path: Path) -> None:
        """All leaves are unique by source (graph dedup)."""
        transforms = [DataReuseTransform()]
        leaves = search(_matmul, transforms, 20, 42, 0, tmp_path, _KERNEL_KWARGS["2x2"])

        sources = [ir_to_source(p) for p in leaves]
        assert len(sources) == len(set(sources))
        assert len(leaves) <= 20

    def test_num_targets_bounds(self, tmp_path: Path) -> None:
        """Returns at most num_targets unique leaves."""
        transforms = [DataReuseTransform()]
        leaves = search(_matmul, transforms, 10, 42, 0, tmp_path, _KERNEL_KWARGS["2x2"])

        assert len(leaves) <= 10
        sources = [ir_to_source(p) for p in leaves]
        assert len(sources) == len(set(sources))

    def test_no_opportunities_returns_root(self, tmp_path: Path) -> None:
        """When no opportunities exist, return [root program].

        A 1x1 matmul has no opportunities, so the root is the only
        leaf regardless of num_targets.
        """
        transforms = [DataReuseTransform()]
        leaves = search(_matmul, transforms, 5, 42, 0, tmp_path, _KERNEL_KWARGS["1x1"])

        assert len(leaves) == 1

    def test_numerical_correctness(self, tmp_path: Path) -> None:
        """All targeted search leaves preserve numerical correctness."""
        transforms = [DataReuseTransform()]
        leaves = search(_matmul, transforms, 10, 42, 0, tmp_path, _KERNEL_KWARGS["2x2"])

        a = make_random_array((128, 256), seed=42)
        b = make_random_array((128, 256), seed=43)
        expected = _matmul(a, b)

        for leaf in leaves:
            actual = ir_to_callable(leaf)(a, b)
            assert_arrays_close(actual, expected)

    def test_multiple_transforms(self, tmp_path: Path) -> None:
        """Targeted search with both transforms returns unique nodes."""
        transforms = [DataReuseTransform(), OperandMergeTransform()]
        results = search(_matmul, transforms, 10, 42, 0, tmp_path, _KERNEL_KWARGS["1x2"])

        assert len(results) >= 1
        assert len(results) <= 10
        sources = _get_unique_sources(results)
        assert len(sources) == len(results)

    def test_multiple_transforms_numerical(self, tmp_path: Path) -> None:
        """Targeted search with both transforms preserves numerical correctness."""
        transforms = [DataReuseTransform(), OperandMergeTransform()]
        leaves = search(_matmul, transforms, 5, 42, 0, tmp_path, _KERNEL_KWARGS["1x2"])

        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 256), seed=43)
        expected = _matmul(a, b)

        for leaf in leaves:
            actual = ir_to_callable(leaf)(a, b)
            assert_arrays_close(actual, expected)

    def test_seed_none_is_unseeded(self, tmp_path: Path) -> None:
        """seed=None does not crash and produces results."""
        transforms = [DataReuseTransform()]
        leaves = search(_matmul, transforms, 3, None, 0, tmp_path, _KERNEL_KWARGS["2x1"])

        assert len(leaves) >= 1

    def test_post_reuse_operand_merge(self, tmp_path: Path) -> None:
        """Targeted search on 1x2 matmul with OperandMergeTransform."""
        transforms = [OperandMergeTransform()]
        leaves = search(_matmul, transforms, 5, 42, 0, tmp_path, _KERNEL_KWARGS["1x2"])

        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 256), seed=43)
        expected = _matmul(a, b)

        for leaf in leaves:
            actual = ir_to_callable(leaf)(a, b)
            assert_arrays_close(actual, expected)


class TestFrontierDepletion:
    """Tests for search behavior when frontier is exhausted."""

    def test_exhaustive_depletes_frontier(self, tmp_path: Path) -> None:
        """Exhaustive search depletes the frontier completely.

        With num_targets=inf, the search must expand all frontier nodes
        until none remain, returning all unique reachable states.
        """
        transforms = [DataReuseTransform()]
        results = search(_matmul, transforms, math.inf, None, 0, tmp_path, _KERNEL_KWARGS["2x1"])

        assert len(results) == 2

    def test_high_num_targets_returns_all_when_frontier_exhausted(self, tmp_path: Path) -> None:
        """When num_targets exceeds reachable states, search stops at frontier depletion.

        A 2x1 matmul has exactly 2 reachable states (root + 1 transform).
        Requesting 100 targets should return only 2.
        """
        transforms = [DataReuseTransform()]
        results = search(_matmul, transforms, 100, 42, 0, tmp_path, _KERNEL_KWARGS["2x1"])

        assert len(results) == 2

    def test_min_depth_filters_shallow_nodes(self, tmp_path: Path) -> None:
        """Nodes shallower than min_depth are excluded from results.

        With min_depth=1, the root node (depth 0) should be excluded.
        """
        transforms = [DataReuseTransform()]
        results = search(_matmul, transforms, math.inf, None, 1, tmp_path, _KERNEL_KWARGS["2x1"])

        assert len(results) == 1

    def test_min_depth_too_high_returns_empty(self, tmp_path: Path) -> None:
        """When min_depth exceeds maximum graph depth, no results are returned.

        A 1x1 matmul has no opportunities (depth stays 0).
        min_depth=1 means the root is excluded, yielding empty results.
        """
        transforms = [DataReuseTransform()]
        results = search(_matmul, transforms, math.inf, None, 1, tmp_path, _KERNEL_KWARGS["1x1"])

        assert len(results) == 0
