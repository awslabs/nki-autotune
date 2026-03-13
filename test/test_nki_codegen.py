"""Tests for codegen package: codegen, render, normalize, stmt types."""

import ast
import dataclasses

import numpy as np
import pytest
from golden.nki_codegen import (
    MATMUL_TANH_BLOCK_0,
    MATMUL_TANH_KERNEL,
    MATMUL_TANH_RENDERED,
    NORMALIZE_AFTER,
    NORMALIZE_BEFORE,
    STMT_RENDER_CASES,
)

import nkigym
from nkigym.codegen import NKIActivation, NKIAlloc, NKIDmaCopy, NKIMatmul, codegen, normalize
from nkigym.ir.tensor import TensorRef
from nkigym.ops.base import NKIOp


def matmul_tanh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """256x256 matmul + tanh test workload."""
    c = nkigym.nc_matmul(a, b)
    return nkigym.activation(c, op=np.tanh)


_KWARGS = {"a": np.zeros((256, 256), dtype=np.float16), "b": np.zeros((256, 256), dtype=np.float16)}


class TestNKIOpRender:
    """Statement render methods."""

    @pytest.mark.parametrize("stmt,expected", STMT_RENDER_CASES)
    def test_render(self, stmt: NKIOp, expected: str) -> None:
        """Verify stmt.render() == expected string."""
        assert stmt.render() == expected


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
        assert len({alloc, alloc, dma}) == 2


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


class TestNormalize:
    """Variable renaming to canonical order."""

    def test_identity(self) -> None:
        """Already-canonical kernel is unchanged."""
        assert normalize(MATMUL_TANH_KERNEL) == MATMUL_TANH_KERNEL

    def test_renumber(self) -> None:
        """Non-canonical tensor_5 gets renamed to tensor_0."""
        assert normalize(NORMALIZE_BEFORE) == NORMALIZE_AFTER


class TestCodegen:
    """End-to-end codegen(func, kwargs) -> NKIKernel."""

    def test_full_kernel(self) -> None:
        """Codegen produces expected golden kernel."""
        assert codegen(matmul_tanh, _KWARGS) == MATMUL_TANH_KERNEL

    def test_block_count(self) -> None:
        """256x256 matmul produces 4 parallel blocks."""
        result = codegen(matmul_tanh, _KWARGS)
        assert len(result.blocks) == 4

    def test_stmts_per_block(self) -> None:
        """Each block has 16 statements (2 red positions)."""
        result = codegen(matmul_tanh, _KWARGS)
        assert len(result.blocks[0].body) == 16

    def test_kernel_metadata(self) -> None:
        """Kernel metadata matches expected values."""
        result = codegen(matmul_tanh, _KWARGS)
        assert result.name == "matmul_tanh"
        assert result.params == ("a", "b")
        assert result.input_shapes == ((256, 256), (256, 256))
        assert result.dtype == "nl.float16"
        assert result.output_shape == (256, 256)

    def test_first_block(self) -> None:
        """First block matches golden block_0."""
        result = codegen(matmul_tanh, _KWARGS)
        assert result.blocks[0] == MATMUL_TANH_BLOCK_0


class TestRender:
    """Source generation from NKIKernel."""

    def test_full_render(self) -> None:
        """Rendered source matches golden string."""
        assert MATMUL_TANH_KERNEL.render() == MATMUL_TANH_RENDERED

    def test_round_trip_parseable(self) -> None:
        """Rendered source is valid Python."""
        ast.parse(MATMUL_TANH_KERNEL.render())
