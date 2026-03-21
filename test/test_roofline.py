"""Tests for roofline cost model analysis."""

import pytest
from golden.nki_roofline import (
    ROOFLINE_COMPUTE_KERNEL,
    ROOFLINE_MATMUL_KERNEL,
    ROOFLINE_NO_MATMUL_KERNEL,
    ROOFLINE_ZERO_BYTES_KERNEL,
)

from nkigym.codegen.roofline import RooflineAnalysis, _dtype_bytes, analyze_roofline


class TestDtypeBytes:
    """Verify _dtype_bytes returns correct byte widths."""

    @pytest.mark.parametrize(
        "dtype_str,expected",
        [
            ("nl.float16", 2),
            ("nl.bfloat16", 2),
            ("nl.float32", 4),
            ("nl.float8_e4m3fn", 1),
            ("nl.float8_e5m2", 1),
            ("nl.int8", 1),
            ("nl.int32", 4),
        ],
    )
    def test_known_dtypes(self, dtype_str: str, expected: int) -> None:
        """Known NKI dtypes return correct byte widths."""
        assert _dtype_bytes(dtype_str) == expected

    def test_unknown_dtype_raises(self) -> None:
        """Unknown dtype raises ValueError."""
        with pytest.raises(ValueError, match="Unknown NKI dtype"):
            _dtype_bytes("nl.complex128")


class TestAnalyzeRoofline:
    """Verify roofline analysis on golden kernels."""

    def test_returns_named_tuple(self) -> None:
        """Result is a RooflineAnalysis NamedTuple."""
        r = analyze_roofline(ROOFLINE_MATMUL_KERNEL)
        assert isinstance(r, RooflineAnalysis)

    def test_matmul_kernel_flops(self) -> None:
        """256x256 BF16 matmul: 4 blocks * 2 matmuls * 128^3 MACs * 2."""
        r = analyze_roofline(ROOFLINE_MATMUL_KERNEL)
        assert r.total_flops == 33_554_432

    def test_matmul_kernel_hbm_bytes(self) -> None:
        """4 blocks * (4 loads + 1 store) * 128*128*2 bytes."""
        r = analyze_roofline(ROOFLINE_MATMUL_KERNEL)
        assert r.total_hbm_bytes == 655_360

    def test_matmul_kernel_arithmetic_intensity(self) -> None:
        """33,554,432 / 655,360 = 51.2 FLOP/byte."""
        r = analyze_roofline(ROOFLINE_MATMUL_KERNEL)
        assert r.arithmetic_intensity == pytest.approx(51.2)

    def test_matmul_kernel_bound(self) -> None:
        """51.2 < 210.67 ridge point -> memory-bound."""
        r = analyze_roofline(ROOFLINE_MATMUL_KERNEL)
        assert r.bound == "memory"

    def test_matmul_kernel_roofline_peak(self) -> None:
        """51.2 * 375e9 / 1e12 = 19.2 TFLOPS."""
        r = analyze_roofline(ROOFLINE_MATMUL_KERNEL)
        assert r.roofline_peak_tflops == pytest.approx(19.2)

    def test_matmul_kernel_ridge_point(self) -> None:
        """78.6432e12 / 375e9 = 209.7152 FLOP/byte."""
        r = analyze_roofline(ROOFLINE_MATMUL_KERNEL)
        assert r.ridge_point == pytest.approx(209.7152, rel=1e-6)

    def test_matmul_kernel_peak_tflops(self) -> None:
        """BF16 peak = 2.4e9 * 2*128*128 / 1e12 = 78.6432 TFLOPS."""
        r = analyze_roofline(ROOFLINE_MATMUL_KERNEL)
        assert r.peak_tflops == pytest.approx(78.6432)

    def test_compute_bound_kernel(self) -> None:
        """Kernel with many matmuls and few DMA ops is compute-bound."""
        r = analyze_roofline(ROOFLINE_COMPUTE_KERNEL)
        assert r.bound == "compute"
        assert r.arithmetic_intensity > r.ridge_point
        assert r.roofline_peak_tflops == pytest.approx(r.peak_tflops)

    def test_no_matmul_kernel(self) -> None:
        """DMA-only kernel has zero flops and is memory-bound."""
        r = analyze_roofline(ROOFLINE_NO_MATMUL_KERNEL)
        assert r.total_flops == 0
        assert r.arithmetic_intensity == 0.0
        assert r.bound == "memory"
        assert r.roofline_peak_tflops == 0.0

    def test_zero_hbm_bytes(self) -> None:
        """Kernel with no HBM DMA has infinite arithmetic intensity."""
        r = analyze_roofline(ROOFLINE_ZERO_BYTES_KERNEL)
        assert r.arithmetic_intensity == float("inf")
        assert r.bound == "compute"
        assert r.roofline_peak_tflops == pytest.approx(r.peak_tflops)


class TestRooflineEfficiency:
    """Verify post-benchmark efficiency calculation logic."""

    def test_efficiency_calculation(self) -> None:
        """Given known latency and roofline peak, verify efficiency."""
        r = analyze_roofline(ROOFLINE_MATMUL_KERNEL)
        min_ms = 0.001
        achieved_tflops = r.total_flops / (min_ms / 1000) / 1e12
        efficiency = achieved_tflops / r.roofline_peak_tflops
        assert achieved_tflops == pytest.approx(33.554432)
        assert efficiency == pytest.approx(33.554432 / 19.2, rel=1e-3)
