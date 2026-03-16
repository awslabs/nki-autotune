"""Tests for DataReuseTransform: analyze and apply with hardcoded golden kernels."""

from golden.nki_transforms import DATA_REUSE_AFTER, DATA_REUSE_BEFORE, DATA_REUSE_NO_MATCH, DATA_REUSE_OPTIONS

from nkigym.transforms.data_reuse import DataReuseTransform


class TestDataReuse:
    """Tests for DataReuseTransform."""

    def test_analyze(self) -> None:
        """Duplicate DMA loads across blocks are detected."""
        assert DataReuseTransform().analyze(DATA_REUSE_BEFORE) == DATA_REUSE_OPTIONS

    def test_apply(self) -> None:
        """Applying reuse removes the duplicate DMA load."""
        result = DataReuseTransform().apply(DATA_REUSE_BEFORE, DATA_REUSE_OPTIONS[0])
        assert result == DATA_REUSE_AFTER

    def test_no_false_positives(self) -> None:
        """Single-load kernel produces no options."""
        assert DataReuseTransform().analyze(DATA_REUSE_NO_MATCH) == []
