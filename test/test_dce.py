"""Tests for dead code elimination with hardcoded golden kernels."""

from golden.nki_dce import (
    DCE_DEAD_AFTER,
    DCE_DEAD_BEFORE,
    DCE_EMPTY_BLOCK_AFTER,
    DCE_EMPTY_BLOCK_BEFORE,
    DCE_LIVE_KERNEL,
)

from nkigym.codegen.dce import dce


class TestDCE:
    """Dead code elimination on NKIKernel."""

    def test_no_dead_code(self) -> None:
        """All-live kernel is unchanged."""
        assert dce(DCE_LIVE_KERNEL) == DCE_LIVE_KERNEL

    def test_removes_dead_code(self) -> None:
        """Dead alloc + DMA load pair is removed."""
        assert dce(DCE_DEAD_BEFORE) == DCE_DEAD_AFTER

    def test_empty_block_dropped(self) -> None:
        """Block with only dead code is removed entirely."""
        assert dce(DCE_EMPTY_BLOCK_BEFORE) == DCE_EMPTY_BLOCK_AFTER
