"""Tests for OperandMergeTransform: DMA merge and compute merge with hardcoded golden kernels."""

from golden.nki_transforms import (
    ACCUMULATION_KERNEL,
    COMPUTE_MERGE_AFTER,
    COMPUTE_MERGE_BEFORE,
    COMPUTE_MERGE_OPTIONS,
    DMA_MERGE_AFTER,
    DMA_MERGE_BEFORE,
    DMA_MERGE_OPTIONS,
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


class TestComputeMerge:
    """Tests for OperandMergeTransform compute merge."""

    def test_analyze(self) -> None:
        """Two matmuls with adjacent moving on N axis are detected."""
        assert OperandMergeTransform().analyze(COMPUTE_MERGE_BEFORE) == COMPUTE_MERGE_OPTIONS

    def test_apply(self) -> None:
        """After compute merge, matmuls are merged with widened operands."""
        result = OperandMergeTransform().apply(COMPUTE_MERGE_BEFORE, COMPUTE_MERGE_OPTIONS[0])
        assert result == COMPUTE_MERGE_AFTER

    def test_m_axis_rejected(self) -> None:
        """M-axis merge to 256 exceeds TILE_LIMITS[M]=128, rejected."""
        assert OperandMergeTransform().analyze(M_AXIS_KERNEL) == []

    def test_accumulation_rejected(self) -> None:
        """Two matmuls accumulating to same dst are not merge candidates."""
        assert OperandMergeTransform().analyze(ACCUMULATION_KERNEL) == []
