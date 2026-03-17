"""Tests for OperandMergeTransform: analyze, apply, and numerical correctness."""

import numpy as np
from golden.nki_transforms import (
    ACCUMULATION_KERNEL,
    ACTIVATION_DIFFERENT_OPS_KERNEL,
    ACTIVATION_MERGE_AFTER,
    ACTIVATION_MERGE_BEFORE,
    ACTIVATION_MERGE_OPTIONS,
    COMPUTE_MERGE_AFTER,
    COMPUTE_MERGE_BEFORE,
    COMPUTE_MERGE_OPTIONS,
    DMA_MERGE_AFTER,
    DMA_MERGE_BEFORE,
    DMA_MERGE_OPTIONS,
    DMA_PARTITION_MERGE_KERNEL,
    M_AXIS_KERNEL,
)

from nkigym.transforms.operand_merge import OperandMergeTransform


class TestDmaMerge:
    """Tests for OperandMergeTransform DMA merge."""

    def test_analyze(self) -> None:
        """Two DMA loads with adjacent HBM sources are detected."""
        assert OperandMergeTransform().analyze(DMA_MERGE_BEFORE) == DMA_MERGE_OPTIONS

    def test_apply(self) -> None:
        """After DMA merge, loads are merged and allocs widened."""
        result = OperandMergeTransform().apply(DMA_MERGE_BEFORE, DMA_MERGE_OPTIONS[0])
        assert result == DMA_MERGE_AFTER

    def test_simulate_correctness(self) -> None:
        """DMA merge preserves simulation output."""
        rng = np.random.default_rng(99)
        kwargs = {"a": rng.standard_normal((128, 256)).astype(np.float16)}
        expected = DMA_MERGE_BEFORE.simulate(kwargs)
        result = OperandMergeTransform().apply(DMA_MERGE_BEFORE, DMA_MERGE_OPTIONS[0]).simulate(kwargs)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_partition_axis_rejected(self) -> None:
        """Partition-axis DMA merge to 256 exceeds buffer limit, rejected."""
        assert OperandMergeTransform().analyze(DMA_PARTITION_MERGE_KERNEL) == []


class TestComputeMerge:
    """Tests for OperandMergeTransform compute merge."""

    def test_analyze(self) -> None:
        """Two matmuls with adjacent moving on N axis are detected."""
        assert OperandMergeTransform().analyze(COMPUTE_MERGE_BEFORE) == COMPUTE_MERGE_OPTIONS

    def test_apply(self) -> None:
        """After compute merge, matmuls are merged with widened operands."""
        result = OperandMergeTransform().apply(COMPUTE_MERGE_BEFORE, COMPUTE_MERGE_OPTIONS[0])
        assert result == COMPUTE_MERGE_AFTER

    def test_simulate_correctness(self) -> None:
        """Compute merge preserves simulation output."""
        rng = np.random.default_rng(99)
        kwargs = {
            "a": rng.standard_normal((128, 128)).astype(np.float16),
            "b": rng.standard_normal((128, 256)).astype(np.float16),
        }
        expected = COMPUTE_MERGE_BEFORE.simulate(kwargs)
        result = OperandMergeTransform().apply(COMPUTE_MERGE_BEFORE, COMPUTE_MERGE_OPTIONS[0]).simulate(kwargs)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_m_axis_rejected(self) -> None:
        """M-axis merge to 256 exceeds TILE_LIMITS[M]=128, rejected."""
        assert OperandMergeTransform().analyze(M_AXIS_KERNEL) == []

    def test_accumulation_rejected(self) -> None:
        """Two matmuls accumulating to same dst are not merge candidates."""
        assert OperandMergeTransform().analyze(ACCUMULATION_KERNEL) == []


class TestActivationMerge:
    """Tests for OperandMergeTransform activation merge."""

    def test_analyze_same_op(self) -> None:
        """Two activations with same op and adjacent src are detected."""
        assert OperandMergeTransform().analyze(ACTIVATION_MERGE_BEFORE) == ACTIVATION_MERGE_OPTIONS

    def test_apply_same_op(self) -> None:
        """After activation merge, activations are merged with widened operands."""
        result = OperandMergeTransform().apply(ACTIVATION_MERGE_BEFORE, ACTIVATION_MERGE_OPTIONS[0])
        assert result == ACTIVATION_MERGE_AFTER

    def test_simulate_correctness(self) -> None:
        """Activation merge preserves simulation output."""
        rng = np.random.default_rng(99)
        kwargs = {"a": rng.standard_normal((128, 256)).astype(np.float16)}
        expected = ACTIVATION_MERGE_BEFORE.simulate(kwargs)
        result = OperandMergeTransform().apply(ACTIVATION_MERGE_BEFORE, ACTIVATION_MERGE_OPTIONS[0]).simulate(kwargs)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_different_ops_rejected(self) -> None:
        """Activations with different ops produce no merge options."""
        assert OperandMergeTransform().analyze(ACTIVATION_DIFFERENT_OPS_KERNEL) == []
