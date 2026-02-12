"""Tests for data reuse analysis and transform passes.

This module contains tests organized by functionality:
- TestAnalyzeDataReuse: Tests for reuse group identification
- TestMergeReusableTensorsSource: Tests for merge transform source code
- TestMergeReusableTensorsErrors: Tests for merge error handling
- TestMergeReusableTensorsNumerical: Tests for merge numerical correctness
- TestIterativeMerge: Tests for chained merge operations

Run with: pytest test/test_data_reuse.py -v
"""

from collections.abc import Callable

import numpy as np
import pytest
from conftest import make_random_array, matmul_input_shapes, normalize_source
from data_reuse_golden import (
    EXPECTED_MERGE_TRANSFORMS,
    EXPECTED_REUSE,
    MERGE_ERROR_CASES,
    MERGED_MATMUL_2X2_ALL_GROUPS,
    MERGED_MATMUL_4X1_ALL_B,
    tiled_double_matmul_2x1,
    tiled_matmul_1x2,
    tiled_matmul_2x1,
    tiled_matmul_2x2,
    tiled_matmul_4x1,
)
from hypothesis import given, settings

import nkigym
from nkigym.ir import callable_to_ir, ir_to_callable
from nkigym.tiling import generate_tiled_ir
from nkigym.transforms import DataReuseTransform, normalize_reuse_groups
from nkigym.utils import get_source

_reuse = DataReuseTransform()


def _analyze_data_reuse(func: Callable) -> list[tuple[str, str]]:
    """Analyze data reuse on a callable by converting to IR first."""
    return _reuse.analyze_ir(callable_to_ir(func))


def _merge_reusable_tensors(func: Callable, tensor_a: str, tensor_b: str) -> Callable:
    """Merge reusable tensors on a callable by converting to/from IR."""
    program = callable_to_ir(func)
    new_program = _reuse.transform_ir(program, (tensor_a, tensor_b))
    return ir_to_callable(new_program)


def assert_reuse_groups_equal(actual: list[tuple[str, ...]], expected: list[tuple[str, ...]]) -> None:
    """Assert two reuse group lists are equal, ignoring order.

    Normalizes both lists by sorting tensors within groups and sorting groups
    lexicographically before comparison. This ensures that equivalent reuse
    groups are considered equal regardless of discovery order.

    Args:
        actual: Result from analyze_data_reuse.
        expected: Expected golden value.

    Raises:
        AssertionError: If groups differ after normalization.

    Example:
        >>> assert_reuse_groups_equal(
        ...     [("b_sg1", "b_sg0")],
        ...     [("b_sg0", "b_sg1")]
        ... )  # Passes - same group, different order
    """
    actual_normalized = normalize_reuse_groups(actual)
    expected_normalized = normalize_reuse_groups(expected)

    assert actual_normalized == expected_normalized, (
        f"Reuse groups mismatch:\n" f"  actual:   {actual_normalized}\n" f"  expected: {expected_normalized}"
    )


class TestAnalyzeDataReuse:
    """Tests for analyze_data_reuse on pre-tiled functions.

    This class verifies that analyze_data_reuse correctly identifies tensor
    slices that can be merged across subgraphs. Tensors are considered reusable
    when they share identical slice parameters (same offset and size).

    Attributes:
        None

    Example:
        Run data reuse analysis tests::

            pytest test/test_data_reuse.py::TestAnalyzeDataReuse -v
    """

    @pytest.mark.parametrize("func", list(EXPECTED_REUSE.keys()), ids=lambda f: f.__name__)
    def test_data_reuse_identifies_correct_groups(self, func: Callable) -> None:
        """Verify data reuse analysis identifies correct reuse groups.

        Tests that analyze_data_reuse correctly identifies all tensor slices
        that share identical slice parameters and can be merged.

        **Validates: Requirements 2.1**

        Args:
            func: Pre-tiled function to analyze for data reuse opportunities.
        """
        reuse_groups = _analyze_data_reuse(func)
        expected = EXPECTED_REUSE[func]
        assert_reuse_groups_equal(reuse_groups, expected)


class TestMergeReusableTensorsSource:
    """Tests for merge_reusable_tensors source code transformation.

    This class verifies that merge_reusable_tensors correctly transforms
    tiled function source code by replacing two tensor slice variables
    with a single shared variable.

    Attributes:
        None

    Example:
        Run merge source transformation tests::

            pytest test/test_data_reuse.py::TestMergeReusableTensorsSource -v
    """

    @pytest.mark.parametrize(
        "func,tensor_a,tensor_b",
        list(EXPECTED_MERGE_TRANSFORMS.keys()),
        ids=[f"{func.__name__}-{a}+{b}" for func, a, b in EXPECTED_MERGE_TRANSFORMS.keys()],
    )
    def test_merge_produces_expected_source(self, func: Callable, tensor_a: str, tensor_b: str) -> None:
        """Verify merge transform produces expected source code.

        Tests that merging two tensor variables produces source code that
        matches the expected golden value.

        **Validates: Requirements 2.2**

        Args:
            func: Pre-tiled function to transform.
            tensor_a: Name of first tensor to merge.
            tensor_b: Name of second tensor to merge into tensor_a.
        """
        merged_func = _merge_reusable_tensors(func, tensor_a, tensor_b)
        actual_source = get_source(merged_func)
        expected_source = EXPECTED_MERGE_TRANSFORMS[(func, tensor_a, tensor_b)]
        assert normalize_source(actual_source) == normalize_source(expected_source)


class TestMergeReusableTensorsErrors:
    """Tests for merge_reusable_tensors error handling.

    This class verifies that merge_reusable_tensors raises appropriate errors
    for invalid inputs, including non-existent tensor names, non-matching
    slices, and attempts to merge a tensor with itself.

    Attributes:
        None

    Example:
        Run merge error handling tests::

            pytest test/test_data_reuse.py::TestMergeReusableTensorsErrors -v
    """

    @pytest.mark.parametrize(
        "func,tensor_a,tensor_b,expected_error,error_match",
        MERGE_ERROR_CASES,
        ids=[f"{func.__name__}-{a}+{b}" for func, a, b, _, _ in MERGE_ERROR_CASES],
    )
    def test_merge_raises_expected_error(
        self, func: Callable, tensor_a: str, tensor_b: str, expected_error: type, error_match: str
    ) -> None:
        """Verify merge transform raises appropriate errors for invalid inputs.

        Tests that merge_reusable_tensors raises the expected exception type
        with a message matching the expected pattern for various error cases.

        **Validates: Requirements 2.3, 2.4**

        Args:
            func: Pre-tiled function to transform.
            tensor_a: Name of first tensor to merge.
            tensor_b: Name of second tensor to merge.
            expected_error: Expected exception type (ValueError, etc.).
            error_match: Regex pattern that error message should match.
        """
        with pytest.raises(expected_error, match=error_match):
            _merge_reusable_tensors(func, tensor_a, tensor_b)


class TestMergeReusableTensorsNumerical:
    """Numerical correctness tests for merge_reusable_tensors transform.

    These tests verify that merging reusable tensor slices produces
    numerically equivalent results to the original tiled function.
    This validates that the merge transform is semantically correct.

    **Validates: Requirements 2.2, 5.3**
    """

    @pytest.mark.parametrize(
        "fixture,tensor_a,tensor_b,a_shape,b_shape",
        [
            (tiled_matmul_2x1, "b_sg0", "b_sg1", (256, 128), (128, 128)),
            (tiled_matmul_1x2, "a_sg0", "a_sg1", (128, 128), (128, 256)),
            (tiled_matmul_2x2, "a_sg0", "a_sg1", (256, 128), (128, 256)),
        ],
        ids=["merge_b_2x1", "merge_a_1x2", "merge_partial_2x2"],
    )
    def test_merge_numerical(
        self, fixture: Callable, tensor_a: str, tensor_b: str, a_shape: tuple[int, int], b_shape: tuple[int, int]
    ) -> None:
        """Verify merging reusable tensors preserves numerical correctness.

        **Validates: Requirements 2.2, 5.3**
        """
        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)

        expected = fixture(a, b)

        merged_func = _merge_reusable_tensors(fixture, tensor_a, tensor_b)
        actual = merged_func(a, b)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    def test_merge_double_matmul_numerical(self) -> None:
        """Verify merging tensors in double matmul preserves numerical correctness.

        Tests that merging b_sg0 and b_sg1 in a double matmul (A @ B @ C)
        produces the same output as the original function.

        **Validates: Requirements 2.2, 5.3**
        """
        a = make_random_array((256, 128), seed=42)
        b = make_random_array((128, 128), seed=43)
        c = make_random_array((128, 128), seed=44)

        expected = tiled_double_matmul_2x1(a, b, c)

        merged_func = _merge_reusable_tensors(tiled_double_matmul_2x1, "b_sg0", "b_sg1")
        actual = merged_func(a, b, c)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    @settings(max_examples=100)
    @given(input_shapes=matmul_input_shapes())
    def test_merge_operation_equivalence_property(self, input_shapes: dict[str, tuple[int, int]]) -> None:
        """Property 6: Merge Operation Equivalence.

        For any valid tensor merge operation (where two tensors share identical
        slices), the merged function SHALL produce output numerically equivalent
        to the original function.

        This property test:
        1. Generates random nc_matmul shapes (k, m, n as multiples of 128)
        2. Creates a tiled matmul function using generate_tiled_ir + ir_to_callable
        3. Identifies reuse groups using analyze_data_reuse
        4. If reuse groups exist, merges tensors from the first group
        5. Verifies numerical equivalence between original and merged functions

        **Feature: test-coverage-improvement, Property 6: Merge Operation Equivalence**

        **Validates: Requirements 2.2, 5.3**
        """
        a_shape = input_shapes["a"]
        b_shape = input_shapes["b"]

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute matrix multiplication using nc_matmul."""
            return nkigym.nc_matmul(a, b)

        tiled_func = ir_to_callable(generate_tiled_ir(matmul, {"a": a_shape, "b": b_shape}, np.float32))

        reuse_pairs = _analyze_data_reuse(tiled_func)

        if not reuse_pairs:
            return

        tensor_a, tensor_b = reuse_pairs[0]

        merged_func = _merge_reusable_tensors(tiled_func, tensor_a, tensor_b)

        np.random.seed(42)
        a = np.random.randn(*a_shape).astype(np.float32)
        np.random.seed(43)
        b = np.random.randn(*b_shape).astype(np.float32)

        expected = tiled_func(a, b)
        actual = merged_func(a, b)

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Merge operation changed output for shapes {a_shape} @ {b_shape}",
        )


class TestIterativeMerge:
    """Tests for iterative/chained merge operations.

    This class verifies that multiple merge operations can be chained together
    to merge all reusable tensors in a tiled function. Each merge operation
    produces a new function that can be further merged.

    Attributes:
        None

    Example:
        Run iterative merge tests::

            pytest test/test_data_reuse.py::TestIterativeMerge -v
    """

    def test_iterative_merge_4x1(self) -> None:
        """Verify iterative merging of 4 B tensors in 4x1 matmul.

        Tests that chaining three merge operations to merge b_sg0, b_sg1,
        b_sg2, and b_sg3 produces the expected source code and maintains
        numerical correctness.

        **Validates: Requirements 2.5**
        """
        merged_func = _merge_reusable_tensors(tiled_matmul_4x1, "b_sg0", "b_sg1")
        merged_func2 = _merge_reusable_tensors(merged_func, "b_sg0", "b_sg2")
        merged_func3 = _merge_reusable_tensors(merged_func2, "b_sg0", "b_sg3")

        assert normalize_source(get_source(merged_func3)) == normalize_source(MERGED_MATMUL_4X1_ALL_B)

        a = make_random_array((512, 128), seed=42)
        b = make_random_array((128, 128), seed=43)

        expected = tiled_matmul_4x1(a, b)
        actual = merged_func3(a, b)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    def test_iterative_merge_all_groups_2x2(self) -> None:
        """Verify merging all independent reuse groups in 2x2 matmul.

        Tests that merging all four reuse groups (a_sg0+a_sg1, a_sg2+a_sg3,
        b_sg0+b_sg2, b_sg1+b_sg3) produces the expected source code and
        maintains numerical correctness.

        **Validates: Requirements 2.5**
        """
        merged_func = _merge_reusable_tensors(tiled_matmul_2x2, "a_sg0", "a_sg1")
        merged_func2 = _merge_reusable_tensors(merged_func, "a_sg2", "a_sg3")
        merged_func3 = _merge_reusable_tensors(merged_func2, "b_sg0", "b_sg2")
        merged_func4 = _merge_reusable_tensors(merged_func3, "b_sg1", "b_sg3")

        assert normalize_source(get_source(merged_func4)) == normalize_source(MERGED_MATMUL_2X2_ALL_GROUPS)

        a = make_random_array((256, 128), seed=42)
        b = make_random_array((128, 256), seed=43)

        expected = tiled_matmul_2x2(a, b)
        actual = merged_func4(a, b)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


class TestDataReuseIRDirect:
    """Tests for DataReuseTransform.analyze_ir/transform_ir on program tuples directly."""

    def test_analyze_ir_finds_reuse_pairs(self) -> None:
        """Verify analyze_ir returns reuse pairs from a program tuple."""

        program = callable_to_ir(tiled_matmul_2x1)
        transform = DataReuseTransform()
        pairs = transform.analyze_ir(program)
        assert len(pairs) > 0
        for pair in pairs:
            assert len(pair) == 2

    def test_transform_ir_returns_valid_program(self) -> None:
        """Verify transform_ir returns a valid program with fewer statements."""

        program = callable_to_ir(tiled_matmul_2x1)
        transform = DataReuseTransform()
        pairs = transform.analyze_ir(program)
        assert len(pairs) > 0

        new_program = transform.transform_ir(program, pairs[0])
        assert new_program.name == program.name
        assert new_program.params == program.params
        assert new_program.return_var == program.return_var
        assert len(new_program.stmts) < len(program.stmts)

        func = ir_to_callable(new_program)
        a = make_random_array((256, 128), seed=42)
        b = make_random_array((128, 128), seed=43)
        expected = tiled_matmul_2x1(a, b)
        actual = func(a, b)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
