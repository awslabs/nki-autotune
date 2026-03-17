"""Tests for codegen: codegen(func, kwargs) -> NKIKernel, normalize, frozen/hashable, axis metadata."""

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest
from golden.nki_codegen import MATMUL_TANH_KERNEL, NORMALIZE_AFTER, NORMALIZE_BEFORE

import nkigym
from nkigym.codegen import NKIActivation, NKIAlloc, NKIDmaCopy, codegen, normalize
from nkigym.codegen.types import NKIKernel
from nkigym.ir.tensor import TensorRef
from nkigym.ops.matmul import NKIMatmul as NKIMatmulOp


def matmul_tanh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """256x256 matmul + tanh test workload."""
    c = nkigym.nc_matmul(a, b)
    return nkigym.activation(c, op=np.tanh)


def matmul_only(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matmul without activation."""
    return nkigym.nc_matmul(a, b)


def matmul_exp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matmul with exp activation."""
    c = nkigym.nc_matmul(a, b)
    return nkigym.activation(c, op=np.exp)


_RNG = np.random.default_rng(42)
_KWARGS = {
    "a": _RNG.standard_normal((256, 256)).astype(np.float16),
    "b": _RNG.standard_normal((256, 256)).astype(np.float16),
}


class TestCodegen:
    """End-to-end codegen(func, kwargs) -> NKIKernel."""

    def test_full_kernel(self) -> None:
        """Codegen produces expected golden kernel and matches numpy semantics."""
        kernel = codegen(matmul_tanh, _KWARGS)
        assert kernel == MATMUL_TANH_KERNEL
        kwargs_f64 = {k: v.astype(np.float64) for k, v in _KWARGS.items()}
        expected = matmul_tanh(**kwargs_f64)
        actual = kernel.simulate(_KWARGS)
        np.testing.assert_allclose(actual, expected)


class TestNormalize:
    """Variable renaming to canonical order."""

    def test_identity(self) -> None:
        """Already-canonical kernel is unchanged."""
        assert normalize(MATMUL_TANH_KERNEL) == MATMUL_TANH_KERNEL

    def test_renumber(self) -> None:
        """Non-canonical tensor_5 gets renamed to tensor_0."""
        assert normalize(NORMALIZE_BEFORE) == NORMALIZE_AFTER


class TestNKIOpFrozen:
    """Immutability and hashability."""

    def test_frozen(self) -> None:
        """NKIOp subclasses raise on mutation."""
        alloc = NKIAlloc(dst="t", shape=(128, 128), dtype="nl.float32", buffer="psum")
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(alloc, "dst", "x")

    def test_hashable(self) -> None:
        """NKIOp subclasses can be put in sets."""
        alloc = NKIAlloc(dst="t", shape=(128, 128), dtype="nl.float32", buffer="psum")
        ref = TensorRef("t", (128, 128), ((0, 128), (0, 128)))
        dma = NKIDmaCopy(dst=ref, src=ref)
        assert {alloc, alloc, dma} == {alloc, dma}


class TestNKIOpAxisMetadata:
    """Class-level axis metadata."""

    def test_matmul_axes(self) -> None:
        """NKIMatmul has correct operand axes."""
        assert NKIMatmulOp.OPERAND_AXES == {"stationary": ("K", "M"), "moving": ("K", "N")}

    def test_matmul_output(self) -> None:
        """NKIMatmul has correct output axes."""
        assert NKIMatmulOp.OUTPUT_AXES == ("M", "N")

    def test_activation_axes(self) -> None:
        """NKIActivation has correct operand axes."""
        assert NKIActivation.OPERAND_AXES == {"data": ("P", "F")}


_SIMULATE_PARAMS = [
    pytest.param(matmul_only, (128, 128), (128, 128), np.float16, id="matmul_only_128"),
    pytest.param(matmul_only, (256, 256), (256, 256), np.float16, id="matmul_only_256"),
    pytest.param(matmul_tanh, (128, 128), (128, 128), np.float16, id="matmul_tanh_128"),
    pytest.param(matmul_exp, (256, 256), (256, 256), np.float16, id="matmul_exp_256"),
    pytest.param(matmul_tanh, (384, 128), (384, 128), np.float16, id="reduction_3_steps"),
    pytest.param(matmul_tanh, (128, 256), (128, 128), np.float16, id="asymmetric_grid"),
]

_STRUCTURE_PARAMS = [
    pytest.param(matmul_only, (128, 128), (128, 128), 1, 1, id="single_tile"),
    pytest.param(matmul_tanh, (256, 256), (256, 256), 4, 2, id="grid_2x2"),
    pytest.param(matmul_tanh, (384, 128), (384, 128), 1, 3, id="deep_reduction"),
    pytest.param(matmul_tanh, (128, 256), (128, 128), 2, 1, id="asymmetric"),
]


def _make_kwargs(a_shape: tuple[int, int], b_shape: tuple[int, int], dtype: type) -> dict:
    """Create random input kwargs for codegen."""
    rng = np.random.default_rng(123)
    return {"a": rng.standard_normal(a_shape).astype(dtype), "b": rng.standard_normal(b_shape).astype(dtype)}


def _count_matmuls(kernel: NKIKernel) -> list[int]:
    """Count NKIMatmul statements per block in a kernel."""
    counts = []
    for block in kernel.blocks:
        count = sum(1 for stmt in block.body if isinstance(stmt, NKIMatmulOp))
        counts.append(count)
    return counts


class TestCodegenSimulate:
    """Parametrized codegen + simulate correctness tests."""

    @pytest.mark.parametrize("func,a_shape,b_shape,dtype", _SIMULATE_PARAMS)
    def test_simulate_correctness(
        self, func: Callable, a_shape: tuple[int, int], b_shape: tuple[int, int], dtype: type
    ) -> None:
        """Codegen + simulate matches numpy reference for various shapes and dtypes."""
        kwargs = _make_kwargs(a_shape, b_shape, dtype)
        kernel = codegen(func, kwargs)
        kwargs_f64 = {k: v.astype(np.float64) for k, v in kwargs.items()}
        expected = func(**kwargs_f64)
        actual = kernel.simulate(kwargs)
        np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)


class TestCodegenStructure:
    """Parametrized codegen structure tests: block count and matmuls per block."""

    @pytest.mark.parametrize("func,a_shape,b_shape,expected_blocks,matmuls_per_block", _STRUCTURE_PARAMS)
    def test_block_structure(
        self,
        func: Callable,
        a_shape: tuple[int, int],
        b_shape: tuple[int, int],
        expected_blocks: int,
        matmuls_per_block: int,
    ) -> None:
        """Codegen produces expected number of blocks and matmuls per block."""
        kwargs = _make_kwargs(a_shape, b_shape, np.float16)
        kernel = codegen(func, kwargs)
        assert len(kernel.blocks) == expected_blocks
        matmul_counts = _count_matmuls(kernel)
        assert all(c == matmuls_per_block for c in matmul_counts)
