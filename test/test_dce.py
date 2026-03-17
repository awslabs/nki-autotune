"""Tests for dead code elimination with hardcoded golden kernels."""

from collections.abc import Callable

import numpy as np
import pytest
from golden.nki_dce import (
    DCE_DEAD_AFTER,
    DCE_DEAD_BEFORE,
    DCE_DEAD_CHAIN_AFTER,
    DCE_DEAD_CHAIN_BEFORE,
    DCE_EMPTY_BLOCK_AFTER,
    DCE_EMPTY_BLOCK_BEFORE,
    DCE_LIVE_KERNEL,
    DCE_TRANSITIVE_AFTER,
    DCE_TRANSITIVE_BEFORE,
)

import nkigym
from nkigym.codegen import codegen
from nkigym.codegen.dce import dce
from nkigym.transforms.data_reuse import DataReuseTransform


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


class TestDCEDeepChain:
    """DCE on deep dead code chains including matmul pipelines."""

    def test_dead_matmul_chain(self) -> None:
        """Full dead matmul pipeline (alloc+dma+matmul+tensor_copy) is removed."""
        assert dce(DCE_DEAD_CHAIN_BEFORE) == DCE_DEAD_CHAIN_AFTER

    def test_transitive(self) -> None:
        """Dead matmul chain with all feeders is transitively removed."""
        assert dce(DCE_TRANSITIVE_BEFORE) == DCE_TRANSITIVE_AFTER


def matmul_tanh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matmul + tanh test workload."""
    c = nkigym.nc_matmul(a, b)
    return nkigym.activation(c, op=np.tanh)


def matmul_only(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matmul without activation."""
    return nkigym.nc_matmul(a, b)


_DCE_SIMULATE_PARAMS = [
    pytest.param(matmul_tanh, (256, 256), (256, 256), np.float16, id="matmul_tanh_256"),
    pytest.param(matmul_only, (128, 128), (128, 128), np.float16, id="matmul_only_128"),
]


class TestDCESimulate:
    """Verify DCE preserves numerical semantics after transforms."""

    @pytest.mark.parametrize("func,a_shape,b_shape,dtype", _DCE_SIMULATE_PARAMS)
    def test_dce_preserves_semantics(
        self, func: Callable, a_shape: tuple[int, int], b_shape: tuple[int, int], dtype: type
    ) -> None:
        """DCE after transform preserves simulation correctness."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(a_shape).astype(dtype)
        b = rng.standard_normal(b_shape).astype(dtype)
        kwargs = {"a": a, "b": b}
        kernel = codegen(func, kwargs)
        expected = kernel.simulate(kwargs)
        transform = DataReuseTransform()
        options = transform.analyze(kernel)
        transformed = transform.apply(kernel, options[0]) if options else kernel
        cleaned = dce(transformed)
        result = cleaned.simulate(kwargs)
        np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)
