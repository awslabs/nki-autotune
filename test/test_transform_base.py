"""Unit tests for nkigym.transforms.base.Transform IR methods.

Tests the analyze_ir() and transform_ir() abstract interface using
concrete test transforms (IdentityTransform and NoOpportunityTransform).
"""

from typing import Any

import numpy as np
from conftest import assert_arrays_close, make_random_array

from nkigym.ir import Program, callable_to_ir, ir_to_callable
from nkigym.transforms.base import Transform


class IdentityTransform(Transform):
    """Transform that returns the program unchanged.

    Attributes:
        name: Transform name for diagnostics.
    """

    name = "identity"

    def analyze_ir(self, program: Program) -> list[Any]:
        """Return a single dummy opportunity.

        Args:
            program: Program tuple.

        Returns:
            List with one None element (single opportunity).
        """
        return [None]

    def transform_ir(self, program: Program, option: Any) -> Program:
        """Return the program unchanged.

        Args:
            program: Program tuple.
            option: Unused.

        Returns:
            The same program tuple.
        """
        return program


class NoOpportunityTransform(Transform):
    """Transform that never finds opportunities.

    Attributes:
        name: Transform name for diagnostics.
    """

    name = "no_opportunity"

    def analyze_ir(self, program: Program) -> list[Any]:
        """Return no opportunities.

        Args:
            program: Program tuple.

        Returns:
            Empty list.
        """
        return []

    def transform_ir(self, program: Program, option: Any) -> Program:
        """Raise since this should never be called.

        Args:
            program: Program tuple.
            option: Unused.

        Raises:
            RuntimeError: Always, since no opportunities should exist.
        """
        raise RuntimeError("transform_ir should not be called with no opportunities")


def _simple_func(a: np.ndarray) -> np.ndarray:
    """Load a slice and return it.

    Args:
        a: Input array of shape (128, 128).

    Returns:
        Loaded tensor.
    """
    tensor_0 = a[0:128, 0:128]
    return tensor_0


class TestAnalyzeIR:
    """Tests for Transform.analyze_ir() method."""

    def test_analyze_ir_finds_opportunities(self) -> None:
        """analyze_ir() returns opportunities from the program tuple."""
        transform = IdentityTransform()
        program = callable_to_ir(_simple_func)
        result = transform.analyze_ir(program)
        assert result == [None]

    def test_analyze_ir_with_no_opportunities(self) -> None:
        """analyze_ir() returns empty list when no opportunities exist."""
        transform = NoOpportunityTransform()
        program = callable_to_ir(_simple_func)
        result = transform.analyze_ir(program)
        assert result == []


class TestTransformIR:
    """Tests for Transform.transform_ir() method."""

    def test_transform_ir_returns_program(self) -> None:
        """transform_ir() returns a Program tuple."""
        transform = IdentityTransform()
        program = callable_to_ir(_simple_func)
        options = transform.analyze_ir(program)
        result = transform.transform_ir(program, options[0])
        assert isinstance(result, Program)

    def test_transform_roundtrip_preserves_behavior(self) -> None:
        """callable_to_ir -> transform_ir -> ir_to_callable preserves semantics."""
        transform = IdentityTransform()
        program = callable_to_ir(_simple_func)
        options = transform.analyze_ir(program)
        new_program = transform.transform_ir(program, options[0])
        result_func = ir_to_callable(new_program)

        a = make_random_array((128, 128), seed=42)
        expected = _simple_func(a)
        actual = result_func(a)
        assert_arrays_close(actual, expected)

    def test_transform_result_has_source(self) -> None:
        """ir_to_callable result has __source__ attribute."""
        transform = IdentityTransform()
        program = callable_to_ir(_simple_func)
        options = transform.analyze_ir(program)
        new_program = transform.transform_ir(program, options[0])
        result_func = ir_to_callable(new_program)
        assert hasattr(result_func, "__source__")
        assert "def _simple_func" in result_func.__source__
