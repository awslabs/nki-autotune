"""Tests for NKIOp.render() and NKIKernel.render() source generation."""

import pytest
from golden.nki_codegen import MATMUL_TANH_KERNEL, MATMUL_TANH_RENDERED, STMT_RENDER_CASES

from nkigym.ops.base import NKIOp


class TestNKIOpRender:
    """Statement render methods."""

    @pytest.mark.parametrize("stmt,expected", STMT_RENDER_CASES)
    def test_render(self, stmt: NKIOp, expected: str) -> None:
        """Verify stmt.render() == expected string."""
        assert stmt.render() == expected


class TestRender:
    """Source generation from NKIKernel."""

    def test_full_render(self) -> None:
        """Rendered source matches golden string."""
        assert MATMUL_TANH_KERNEL.render() == MATMUL_TANH_RENDERED
