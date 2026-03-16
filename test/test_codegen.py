"""Tests for codegen: codegen(func, kwargs) -> NKIKernel, normalize, frozen/hashable, axis metadata."""

import dataclasses

import numpy as np
import pytest
from golden.nki_codegen import MATMUL_TANH_KERNEL, NORMALIZE_AFTER, NORMALIZE_BEFORE

import nkigym
from nkigym.codegen import NKIActivation, NKIAlloc, NKIDmaCopy, NKIMatmul, codegen, normalize
from nkigym.ir.tensor import TensorRef


def matmul_tanh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """256x256 matmul + tanh test workload."""
    c = nkigym.nc_matmul(a, b)
    return nkigym.activation(c, op=np.tanh)


_KWARGS = {"a": np.zeros((256, 256), dtype=np.float16), "b": np.zeros((256, 256), dtype=np.float16)}


class TestCodegen:
    """End-to-end codegen(func, kwargs) -> NKIKernel."""

    def test_full_kernel(self) -> None:
        """Codegen produces expected golden kernel."""
        assert codegen(matmul_tanh, _KWARGS) == MATMUL_TANH_KERNEL


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
        assert NKIMatmul.OPERAND_AXES == {"stationary": ("K", "M"), "moving": ("K", "N")}

    def test_matmul_output(self) -> None:
        """NKIMatmul has correct output axes."""
        assert NKIMatmul.OUTPUT_AXES == ("M", "N")

    def test_activation_axes(self) -> None:
        """NKIActivation has correct operand axes."""
        assert NKIActivation.OPERAND_AXES == {"data": ("P", "F")}
