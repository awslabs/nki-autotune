"""Tests for DataReuseTransform: analyze, apply, and numerical correctness."""

import numpy as np
from golden.nki_data_reuse import (
    DIFFERENT_SLICES_KERNEL,
    THREE_BLOCK_AFTER,
    THREE_BLOCK_BEFORE,
    THREE_BLOCK_OPTIONS,
    WITHIN_BLOCK_AFTER,
    WITHIN_BLOCK_BEFORE,
    WITHIN_BLOCK_OPTIONS,
)
from golden.nki_transforms import DATA_REUSE_AFTER, DATA_REUSE_BEFORE, DATA_REUSE_NO_MATCH, DATA_REUSE_OPTIONS

from nkigym.transforms.data_reuse import DataReuseTransform


class TestDataReuse:
    """Tests for DataReuseTransform with two-block golden data."""

    def test_analyze(self) -> None:
        """Duplicate DMA loads across blocks are detected."""
        assert DataReuseTransform().analyze(DATA_REUSE_BEFORE) == DATA_REUSE_OPTIONS

    def test_apply(self) -> None:
        """Applying reuse removes the duplicate DMA load."""
        result = DataReuseTransform().apply(DATA_REUSE_BEFORE, DATA_REUSE_OPTIONS[0])
        assert result == DATA_REUSE_AFTER

    def test_simulate_correctness(self) -> None:
        """Both reuse options preserve simulation output."""
        rng = np.random.default_rng(99)
        kwargs = {
            "a": rng.standard_normal((128, 128)).astype(np.float16),
            "b": rng.standard_normal((128, 128)).astype(np.float16),
        }
        expected = DATA_REUSE_BEFORE.simulate(kwargs)
        transform = DataReuseTransform()
        for option in DATA_REUSE_OPTIONS:
            result = transform.apply(DATA_REUSE_BEFORE, option).simulate(kwargs)
            np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_no_false_positives(self) -> None:
        """Single-load kernel produces no options."""
        assert DataReuseTransform().analyze(DATA_REUSE_NO_MATCH) == []


class TestThreeBlocks:
    """Three-block data reuse with identical loads."""

    def test_analyze(self) -> None:
        """Three identical DMA loads produce 3 pair options."""
        assert DataReuseTransform().analyze(THREE_BLOCK_BEFORE) == THREE_BLOCK_OPTIONS

    def test_apply(self) -> None:
        """Applying first option merges blocks 0 and 1, removes duplicate DMA."""
        result = DataReuseTransform().apply(THREE_BLOCK_BEFORE, THREE_BLOCK_OPTIONS[0])
        assert result == THREE_BLOCK_AFTER

    def test_simulate_correctness(self) -> None:
        """All 3 reuse options preserve simulation output."""
        rng = np.random.default_rng(99)
        kwargs = {"a": rng.standard_normal((128, 128)).astype(np.float16)}
        expected = THREE_BLOCK_BEFORE.simulate(kwargs)
        transform = DataReuseTransform()
        for option in THREE_BLOCK_OPTIONS:
            result = transform.apply(THREE_BLOCK_BEFORE, option).simulate(kwargs)
            np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_no_match_different_slices(self) -> None:
        """Different source slices produce no options."""
        assert DataReuseTransform().analyze(DIFFERENT_SLICES_KERNEL) == []


class TestWithinBlock:
    """Within-block data reuse deduplication."""

    def test_analyze(self) -> None:
        """Two identical DMA loads in same block produce 1 option."""
        assert DataReuseTransform().analyze(WITHIN_BLOCK_BEFORE) == WITHIN_BLOCK_OPTIONS

    def test_apply(self) -> None:
        """Applying within-block reuse removes duplicate and renames consumers."""
        result = DataReuseTransform().apply(WITHIN_BLOCK_BEFORE, WITHIN_BLOCK_OPTIONS[0])
        assert result == WITHIN_BLOCK_AFTER

    def test_simulate_correctness(self) -> None:
        """Within-block reuse preserves simulation output."""
        rng = np.random.default_rng(99)
        kwargs = {"a": rng.standard_normal((128, 128)).astype(np.float16)}
        expected = WITHIN_BLOCK_BEFORE.simulate(kwargs)
        result = DataReuseTransform().apply(WITHIN_BLOCK_BEFORE, WITHIN_BLOCK_OPTIONS[0]).simulate(kwargs)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
