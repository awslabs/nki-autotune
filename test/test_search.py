"""Tests for transform graph search module.

Tests search() with exhaustive (num_targets=inf) and targeted
(num_targets=N) modes using golden fixtures from data_reuse_golden.py
and operand_merge_golden.py.

Run with: pytest test/test_search.py -v
"""

import math
from collections.abc import Callable

from conftest import assert_arrays_close, make_random_array
from data_reuse_golden import tiled_matmul_1x1, tiled_matmul_1x2, tiled_matmul_2x1, tiled_matmul_2x2
from operand_merge_golden import tiled_matmul_post_reuse_1x2

from nkigym.search import search
from nkigym.transforms import DataReuseTransform, OperandMergeTransform
from nkigym.utils.source import get_source


def _get_unique_sources(funcs: list[Callable]) -> set[str]:
    """Get the set of unique sources from a list of callables.

    Args:
        funcs: List of callables with source.

    Returns:
        Set of source strings.
    """
    return {get_source(f) for f in funcs}


class TestExhaustiveSearch:
    """Tests for search() in exhaustive mode (num_targets=inf)."""

    def test_no_opportunities_returns_root(self) -> None:
        """When no transforms have opportunities, return [func].

        tiled_matmul_1x1 has a single subgraph so DataReuseTransform
        finds nothing to merge.
        """
        transforms = [DataReuseTransform()]
        leaves = search(tiled_matmul_1x1, transforms, math.inf, None, 0, None, False, None)

        assert len(leaves) == 1
        assert get_source(leaves[0]) == get_source(tiled_matmul_1x1)

    def test_leaves_are_unique_by_source(self) -> None:
        """All returned leaves have distinct source code.

        Uses tiled_matmul_2x2 with DataReuseTransform which has
        multiple convergent paths.
        """
        transforms = [DataReuseTransform()]
        leaves = search(tiled_matmul_2x2, transforms, math.inf, None, 0, None, False, None)

        sources = [get_source(f) for f in leaves]
        assert len(sources) == len(set(sources))

    def test_numerical_correctness(self) -> None:
        """All leaves preserve numerical correctness.

        Applies exhaustive search to tiled_matmul_2x2 with DataReuseTransform,
        then verifies each leaf produces the same output as the original.
        """
        transforms = [DataReuseTransform()]
        leaves = search(tiled_matmul_2x2, transforms, math.inf, None, 0, None, False, None)

        a = make_random_array((256, 128), seed=42)
        b = make_random_array((128, 256), seed=43)
        expected = tiled_matmul_2x2(a, b)

        for leaf in leaves:
            actual = leaf(a, b)
            assert_arrays_close(actual, expected)

    def test_dedup_prunes_convergent_paths(self) -> None:
        """Source-based dedup prunes states reached via different orderings.

        tiled_matmul_2x2 has 4 merge pairs. Different orderings can reach
        the same merged state. Without dedup the number of visited nodes
        would be larger.
        """
        transforms = [DataReuseTransform()]
        leaves = search(tiled_matmul_2x2, transforms, math.inf, None, 0, None, False, None)

        sources = _get_unique_sources(leaves)
        assert len(sources) == len(leaves)

    def test_single_transform_single_opportunity(self) -> None:
        """Single transform with a single opportunity produces root and transformed node.

        tiled_matmul_2x1 with DataReuseTransform has exactly one pair,
        yielding two nodes: the root and the transformed state.
        """
        transforms = [DataReuseTransform()]
        results = search(tiled_matmul_2x1, transforms, math.inf, None, 0, None, False, None)

        assert len(results) == 2
        sources = _get_unique_sources(results)
        assert get_source(tiled_matmul_2x1) in sources

    def test_multiple_transforms(self) -> None:
        """Exhaustive search with both DataReuse and OperandMerge transforms.

        Uses tiled_matmul_1x2 which has data reuse opportunities,
        and the post-reuse state has operand merge opportunities.
        Returns all unique nodes including intermediate states.
        """
        transforms = [DataReuseTransform(), OperandMergeTransform()]
        results = search(tiled_matmul_1x2, transforms, math.inf, None, 0, None, False, None)

        assert len(results) >= 1
        sources = _get_unique_sources(results)
        assert len(sources) == len(results)

    def test_multiple_transforms_numerical_correctness(self) -> None:
        """Leaves from multi-transform search preserve numerical correctness."""
        transforms = [DataReuseTransform(), OperandMergeTransform()]
        leaves = search(tiled_matmul_1x2, transforms, math.inf, None, 0, None, False, None)

        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 256), seed=43)
        expected = tiled_matmul_1x2(a, b)

        for leaf in leaves:
            actual = leaf(a, b)
            assert_arrays_close(actual, expected)

    def test_operand_merge_only(self) -> None:
        """Exhaustive search with OperandMergeTransform on post-reuse fixture.

        tiled_matmul_post_reuse_1x2 has adjacent b loads and nc_matmul ops
        that can be merged.
        """
        transforms = [OperandMergeTransform()]
        leaves = search(tiled_matmul_post_reuse_1x2, transforms, math.inf, None, 0, None, False, None)

        assert len(leaves) >= 1

        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 256), seed=43)
        expected = tiled_matmul_post_reuse_1x2(a, b)

        for leaf in leaves:
            actual = leaf(a, b)
            assert_arrays_close(actual, expected)


class TestTargetedSearch:
    """Tests for search() with finite num_targets."""

    def test_seed_reproducibility(self) -> None:
        """Same seed produces identical results across runs."""
        transforms = [DataReuseTransform()]

        result1 = search(tiled_matmul_2x2, transforms, 5, 42, 0, None, False, None)
        result2 = search(tiled_matmul_2x2, transforms, 5, 42, 0, None, False, None)

        sources1 = [get_source(f) for f in result1]
        sources2 = [get_source(f) for f in result2]
        assert sources1 == sources2

    def test_unique_dedup(self) -> None:
        """All leaves are unique by source (graph dedup)."""
        transforms = [DataReuseTransform()]
        leaves = search(tiled_matmul_2x2, transforms, 20, 42, 0, None, False, None)

        sources = [get_source(f) for f in leaves]
        assert len(sources) == len(set(sources))
        assert len(leaves) <= 20

    def test_num_targets_bounds(self) -> None:
        """Returns at most num_targets unique leaves."""
        transforms = [DataReuseTransform()]
        leaves = search(tiled_matmul_2x2, transforms, 10, 42, 0, None, False, None)

        assert len(leaves) <= 10
        sources = [get_source(f) for f in leaves]
        assert len(sources) == len(set(sources))

    def test_no_opportunities_returns_root(self) -> None:
        """When no opportunities exist, return [func].

        tiled_matmul_1x1 has no opportunities, so the root is the only
        leaf regardless of num_targets.
        """
        transforms = [DataReuseTransform()]
        leaves = search(tiled_matmul_1x1, transforms, 5, 42, 0, None, False, None)

        assert len(leaves) == 1
        assert get_source(leaves[0]) == get_source(tiled_matmul_1x1)

    def test_numerical_correctness(self) -> None:
        """All targeted search leaves preserve numerical correctness."""
        transforms = [DataReuseTransform()]
        leaves = search(tiled_matmul_2x2, transforms, 10, 42, 0, None, False, None)

        a = make_random_array((256, 128), seed=42)
        b = make_random_array((128, 256), seed=43)
        expected = tiled_matmul_2x2(a, b)

        for leaf in leaves:
            actual = leaf(a, b)
            assert_arrays_close(actual, expected)

    def test_multiple_transforms(self) -> None:
        """Targeted search with both transforms returns unique nodes."""
        transforms = [DataReuseTransform(), OperandMergeTransform()]
        results = search(tiled_matmul_1x2, transforms, 10, 42, 0, None, False, None)

        assert len(results) >= 1
        assert len(results) <= 10
        sources = _get_unique_sources(results)
        assert len(sources) == len(results)

    def test_multiple_transforms_numerical(self) -> None:
        """Targeted search with both transforms preserves numerical correctness."""
        transforms = [DataReuseTransform(), OperandMergeTransform()]
        leaves = search(tiled_matmul_1x2, transforms, 5, 42, 0, None, False, None)

        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 256), seed=43)
        expected = tiled_matmul_1x2(a, b)

        for leaf in leaves:
            actual = leaf(a, b)
            assert_arrays_close(actual, expected)

    def test_seed_none_is_unseeded(self) -> None:
        """seed=None does not crash and produces results."""
        transforms = [DataReuseTransform()]
        leaves = search(tiled_matmul_2x1, transforms, 3, None, 0, None, False, None)

        assert len(leaves) >= 1

    def test_post_reuse_operand_merge(self) -> None:
        """Targeted search on a post-reuse fixture with OperandMergeTransform."""
        transforms = [OperandMergeTransform()]
        leaves = search(tiled_matmul_post_reuse_1x2, transforms, 5, 42, 0, None, False, None)

        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 256), seed=43)
        expected = tiled_matmul_post_reuse_1x2(a, b)

        for leaf in leaves:
            actual = leaf(a, b)
            assert_arrays_close(actual, expected)
