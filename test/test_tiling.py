"""Tests for dimension analysis and tiling pass.

This module contains tests organized by functionality:
- TestDimensionAnalysis: Tests for dimension classification and analysis
- TestTiledFunctionGeneration: Tests for code generation
- TestNumericalCorrectness: Tests for numerical equivalence
- TestNamingConventionProperties: Property tests for naming conventions
- TestErrorHandling: Tests for error conditions and validation
- TestOpSemantics: Tests for OP_SEMANTICS expression generation

Run with: pytest test/test_tiling.py -v
"""

import re

import numpy as np
import pytest
from conftest import assert_arrays_close, make_random_array, normalize_source, shape_id
from hypothesis import given, settings
from hypothesis import strategies as st
from tiling_golden import (
    EXPECTED_DOUBLE_MATMUL,
    EXPECTED_SINGLE_MATMUL,
    GOLDEN_DOUBLE_MATMUL_SOURCE,
    GOLDEN_SINGLE_MATMUL_SOURCE,
)

import nkigym
from nkigym.ops import OP_REGISTRY
from nkigym.tiling import analyze_dimension, generate_tiled_function, generate_tiled_source

SINGLE_MATMUL_NUMERICAL_SHAPES: list[tuple[tuple[int, int], tuple[int, int]]] = [
    ((128, 128), (128, 128)),
    ((128, 256), (128, 128)),
    ((128, 128), (128, 256)),
    ((128, 256), (128, 256)),
    ((128, 512), (128, 128)),
    ((256, 512), (256, 512)),
    ((512, 128), (512, 128)),
]

DOUBLE_MATMUL_NUMERICAL_SHAPES: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = [
    ((128, 128), (128, 128), (128, 128)),
    ((128, 256), (128, 128), (256, 128)),
    ((128, 256), (128, 128), (256, 256)),
    ((256, 256), (256, 256), (256, 256)),
]


class TestDimensionAnalysis:
    """Tests for dimension analysis on NumPy functions.

    This class verifies that analyze_dimension correctly classifies dimensions
    as parallel or reduction, computes tile counts, and determines slice
    parameters for various matmul configurations.

    Attributes:
        None

    Example:
        Run dimension analysis tests::

            pytest test/test_tiling.py::TestDimensionAnalysis -v
    """

    @pytest.mark.parametrize(
        "a_shape,b_shape", list(EXPECTED_SINGLE_MATMUL.keys()), ids=[shape_id(k) for k in EXPECTED_SINGLE_MATMUL.keys()]
    )
    def test_single_matmul_dimensions(self, a_shape, b_shape, matmul_func):
        """Verify dimension analysis for single matmul: C[m,n] = A[m,k] @ B[k,n].

        Args:
            a_shape: Shape of the first input matrix (m, k).
            b_shape: Shape of the second input matrix (k, n).
            matmul_func: Fixture providing the matmul function.
        """
        analysis = analyze_dimension(matmul_func, {"a": a_shape, "b": b_shape})
        expected = EXPECTED_SINGLE_MATMUL[(a_shape, b_shape)]
        analysis.assert_equal(expected)

    @pytest.mark.parametrize(
        "a_shape,b_shape,c_shape",
        list(EXPECTED_DOUBLE_MATMUL.keys()),
        ids=[shape_id(k) for k in EXPECTED_DOUBLE_MATMUL.keys()],
    )
    def test_double_matmul_dimensions(self, a_shape, b_shape, c_shape, double_matmul_func):
        """Verify dimension analysis for double matmul: D[m,n] = (A[m,k1] @ B[k1,k2]) @ C[k2,n].

        Args:
            a_shape: Shape of the first input matrix (m, k1).
            b_shape: Shape of the second input matrix (k1, k2).
            c_shape: Shape of the third input matrix (k2, n).
            double_matmul_func: Fixture providing the double matmul function.
        """
        analysis = analyze_dimension(double_matmul_func, {"a": a_shape, "b": b_shape, "c": c_shape})
        expected = EXPECTED_DOUBLE_MATMUL[(a_shape, b_shape, c_shape)]
        analysis.assert_equal(expected)


class TestTiledFunctionGeneration:
    """Tests for tiled function source code generation.

    This class verifies that generate_tiled_source produces syntactically
    valid Python code that matches expected golden values for various
    matmul configurations.

    Attributes:
        None

    Example:
        Run code generation tests::

            pytest test/test_tiling.py::TestTiledFunctionGeneration -v
    """

    @pytest.mark.parametrize(
        "a_shape,b_shape",
        list(GOLDEN_SINGLE_MATMUL_SOURCE.keys()),
        ids=[shape_id(k) for k in GOLDEN_SINGLE_MATMUL_SOURCE.keys()],
    )
    def test_single_matmul_tiled_source(self, a_shape, b_shape, matmul_func):
        """Verify generated tiled source matches golden string for single matmul.

        Args:
            a_shape: Shape of the first input matrix.
            b_shape: Shape of the second input matrix.
            matmul_func: Fixture providing the matmul function.
        """
        input_shapes = {"a": a_shape, "b": b_shape}
        actual_source = generate_tiled_source(matmul_func, input_shapes, output_dtype=np.float32)
        expected_source = GOLDEN_SINGLE_MATMUL_SOURCE[(a_shape, b_shape)]
        assert normalize_source(actual_source) == normalize_source(expected_source)

    @pytest.mark.parametrize(
        "a_shape,b_shape,c_shape",
        list(GOLDEN_DOUBLE_MATMUL_SOURCE.keys()),
        ids=[shape_id(k) for k in GOLDEN_DOUBLE_MATMUL_SOURCE.keys()],
    )
    def test_double_matmul_tiled_source(self, a_shape, b_shape, c_shape, double_matmul_func):
        """Verify generated tiled source matches golden string for double matmul.

        Args:
            a_shape: Shape of the first input matrix.
            b_shape: Shape of the second input matrix.
            c_shape: Shape of the third input matrix.
            double_matmul_func: Fixture providing the double matmul function.
        """
        input_shapes = {"a": a_shape, "b": b_shape, "c": c_shape}
        actual_source = generate_tiled_source(double_matmul_func, input_shapes, output_dtype=np.float32)
        expected_source = GOLDEN_DOUBLE_MATMUL_SOURCE[(a_shape, b_shape, c_shape)]
        assert normalize_source(actual_source) == normalize_source(expected_source)


class TestNumericalCorrectness:
    """Tests for numerical correctness of tiled functions.

    This class verifies that generate_tiled_function produces callables that
    compute the same results as the original NumPy functions within specified
    tolerances (rtol=1e-4, atol=1e-4).

    Attributes:
        None

    Example:
        Run numerical correctness tests::

            pytest test/test_tiling.py::TestNumericalCorrectness -v
    """

    @pytest.mark.parametrize(
        "a_shape,b_shape", SINGLE_MATMUL_NUMERICAL_SHAPES, ids=[shape_id(k) for k in SINGLE_MATMUL_NUMERICAL_SHAPES]
    )
    def test_single_matmul_numerical(self, a_shape, b_shape, matmul_func):
        """Verify numerical correctness for single matmul: C = A @ B.

        Args:
            a_shape: Shape of the first input matrix.
            b_shape: Shape of the second input matrix.
            matmul_func: Fixture providing the matmul function.
        """
        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)

        expected = matmul_func(a, b)

        input_shapes = {"a": a_shape, "b": b_shape}
        tiled_matmul = generate_tiled_function(matmul_func, input_shapes, output_dtype=a.dtype)
        actual = tiled_matmul(a, b)

        assert_arrays_close(actual, expected)

    @pytest.mark.parametrize(
        "a_shape,b_shape,c_shape",
        DOUBLE_MATMUL_NUMERICAL_SHAPES,
        ids=[shape_id(k) for k in DOUBLE_MATMUL_NUMERICAL_SHAPES],
    )
    def test_double_matmul_numerical(self, a_shape, b_shape, c_shape, double_matmul_func):
        """Verify numerical correctness for double matmul: D = (A @ B) @ C.

        Args:
            a_shape: Shape of the first input matrix.
            b_shape: Shape of the second input matrix.
            c_shape: Shape of the third input matrix.
            double_matmul_func: Fixture providing the double matmul function.
        """
        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        c = make_random_array(c_shape, seed=44)

        expected = double_matmul_func(a, b, c)

        input_shapes = {"a": a_shape, "b": b_shape, "c": c_shape}
        tiled_fn = generate_tiled_function(double_matmul_func, input_shapes, output_dtype=a.dtype)
        actual = tiled_fn(a, b, c)

        assert_arrays_close(actual, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtype_preservation(self, dtype, matmul_func):
        """Verify tiled function handles different input dtypes correctly.

        Args:
            dtype: NumPy dtype to test (float32 or float64).
            matmul_func: Fixture providing the matmul function.
        """
        a = make_random_array((128, 256), seed=42, dtype=dtype)
        b = make_random_array((128, 256), seed=43, dtype=dtype)

        input_shapes = {"a": (128, 256), "b": (128, 256)}
        tiled_fn = generate_tiled_function(matmul_func, input_shapes, output_dtype=dtype)

        expected = matmul_func(a, b)
        actual = tiled_fn(a, b)

        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    @settings(max_examples=100)
    @given(
        m=st.integers(min_value=1, max_value=4),
        k=st.integers(min_value=1, max_value=4),
        n=st.integers(min_value=1, max_value=4),
        seed_a=st.integers(min_value=0, max_value=10000),
        seed_b=st.integers(min_value=0, max_value=10000),
    )
    def test_tiled_output_matches_original_across_shapes(self, m: int, k: int, n: int, seed_a: int, seed_b: int):
        """Property: Tiled function output matches original across random shapes.

        **Validates: Requirements 5.1, 5.5**

        For any valid input shapes, the generated tiled function SHALL produce
        numerically equivalent results to the original untiled function within
        tolerance (rtol=1e-4, atol=1e-4).

        Args:
            m: Multiplier for M dimension (1-4, scaled by 128).
            k: Multiplier for K dimension (1-4, scaled by 128).
            n: Multiplier for N dimension (1-4, scaled by 128).
            seed_a: Random seed for first input array.
            seed_b: Random seed for second input array.
        """
        a_shape = (k * 128, m * 128)
        b_shape = (k * 128, n * 128)

        a = make_random_array(a_shape, seed=seed_a)
        b = make_random_array(b_shape, seed=seed_b)

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute matrix multiplication using nkigym.nc_matmul."""
            return nkigym.nc_matmul(a, b)

        expected = matmul(a, b)

        input_shapes = {"a": a_shape, "b": b_shape}
        tiled_matmul = generate_tiled_function(matmul, input_shapes, output_dtype=a.dtype)
        actual = tiled_matmul(a, b)

        assert actual.shape == expected.shape, f"Shape mismatch: actual {actual.shape} vs expected {expected.shape}"
        assert actual.dtype == expected.dtype, f"Dtype mismatch: actual {actual.dtype} vs expected {expected.dtype}"
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-4,
            atol=1e-4,
            err_msg=(
                f"Tiled function output differs from original.\n"
                f"Input shapes: a={a_shape}, b={b_shape}\n"
                f"Seeds: seed_a={seed_a}, seed_b={seed_b}"
            ),
        )


def _extract_function_signature_params(source: str) -> list[str]:
    """Extract parameter names from a generated function signature.

    Parses the function definition line to extract parameter names.
    For example, 'def tiled_matmul(a, b):' returns ['a', 'b'].

    Args:
        source: Generated Python source code.

    Returns:
        List of parameter names in order.

    Raises:
        ValueError: If no function definition is found in the source.
    """
    match = re.search(r"def\s+\w+\(([^)]*)\):", source)
    if not match:
        raise ValueError(f"Could not find function definition in source:\n{source}")
    params_str = match.group(1)
    if not params_str.strip():
        return []
    return [p.strip() for p in params_str.split(",")]


@st.composite
def valid_input_name(draw: st.DrawFn) -> str:
    """Generate a valid input parameter name that is not reserved.

    Valid names:
    - Start with a letter or underscore
    - Contain only letters, digits, and underscores
    - Are not "output"
    - Do not match the pattern "tensor_N" where N is an integer
    - Are not Python reserved keywords

    Args:
        draw: Hypothesis draw function for generating values.

    Returns:
        A valid input parameter name string.
    """
    python_keywords = {
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
    }

    first_char = draw(st.sampled_from("abcdefghijklmnopqrstuvwxyz"))
    rest_length = draw(st.integers(min_value=0, max_value=10))
    rest_chars = draw(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=rest_length, max_size=rest_length)
    )
    name = first_char + rest_chars

    if name == "output":
        name = "inp_" + name
    if re.match(r"^tensor_\d+$", name):
        name = "inp_" + name
    if name in python_keywords:
        name = "var_" + name
    if name == "np":
        name = "inp_" + name

    return name


@st.composite
def matmul_input_shapes_with_names(draw: st.DrawFn) -> dict[str, tuple[int, int]]:
    """Generate valid input shapes for nc_matmul with custom names.

    Generates two input names and compatible shapes for nc_matmul:
    - First input: (k, m) where k is the reduction dimension
    - Second input: (k, n) where k is the reduction dimension
    where m, k, n are multiples of 128.

    Args:
        draw: Hypothesis draw function for generating values.

    Returns:
        Dictionary mapping input names to their shapes.
    """
    name1 = draw(valid_input_name())
    name2 = draw(valid_input_name().filter(lambda n: n != name1))

    m = draw(st.integers(min_value=1, max_value=4)) * 128
    k = draw(st.integers(min_value=1, max_value=4)) * 128
    n = draw(st.integers(min_value=1, max_value=4)) * 128

    return {name1: (k, m), name2: (k, n)}


class TestCodeGenerationProperties:
    """Property-based tests for code generation correctness.

    This class verifies code generation properties using hypothesis for
    property-based testing. Tests ensure that:
    - Dimension classification is correct for matmul operations (Property 1)
    - Input parameter names are preserved in generated code
    - Computed variables follow the tensor_N naming pattern

    These property tests complement the golden value unit tests by verifying
    correctness across many randomly generated input configurations.

    Attributes:
        None

    Example:
        Run code generation property tests::

            pytest test/test_tiling.py::TestCodeGenerationProperties -v
    """

    @settings(max_examples=100)
    @given(
        m=st.integers(min_value=1, max_value=4),
        k=st.integers(min_value=1, max_value=4),
        n=st.integers(min_value=1, max_value=4),
    )
    def test_dimension_classification_parallel_vs_reduction(self, m: int, k: int, n: int):
        """Property: Dimension classification is correct for nc_matmul operations.

        **Validates: Requirements 1.1**

        For any valid nc_matmul input shapes (k, m) and (k, n), the dimension
        analysis SHALL classify the output dimensions (m, n) as "parallel"
        and the contracted dimension (k) as "reduction".

        For nc_matmul C[m,n] = A[k,m].T @ B[k,n]:
        - d0 (K dimension): reduction - contracted over
        - d1 (M dimension): parallel - appears in output
        - d2 (N dimension): parallel - appears in output

        Args:
            m: Multiplier for M dimension (1-4, scaled by 128).
            k: Multiplier for K dimension (1-4, scaled by 128).
            n: Multiplier for N dimension (1-4, scaled by 128).
        """
        a_shape = (k * 128, m * 128)
        b_shape = (k * 128, n * 128)

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute matrix multiplication using nc_matmul."""
            return nkigym.nc_matmul(a, b)

        analysis = analyze_dimension(matmul, {"a": a_shape, "b": b_shape})

        assert analysis.dim_order == ["d0", "d1", "d2"], (
            f"Expected dim_order ['d0', 'd1', 'd2'], got {analysis.dim_order}\n"
            f"Input shapes: a={a_shape}, b={b_shape}"
        )

        assert analysis.dim_info["d0"].iter_type == "reduction", (
            f"Expected d0 (K dimension) to be 'reduction', got '{analysis.dim_info['d0'].iter_type}'\n"
            f"Input shapes: a={a_shape}, b={b_shape}"
        )

        assert analysis.dim_info["d1"].iter_type == "parallel", (
            f"Expected d1 (M dimension) to be 'parallel', got '{analysis.dim_info['d1'].iter_type}'\n"
            f"Input shapes: a={a_shape}, b={b_shape}"
        )

        assert analysis.dim_info["d2"].iter_type == "parallel", (
            f"Expected d2 (N dimension) to be 'parallel', got '{analysis.dim_info['d2'].iter_type}'\n"
            f"Input shapes: a={a_shape}, b={b_shape}"
        )

        assert analysis.dim_info["d0"].size == k * 128, (
            f"Expected d0 size {k * 128}, got {analysis.dim_info['d0'].size}\n"
            f"Input shapes: a={a_shape}, b={b_shape}"
        )

        assert analysis.dim_info["d1"].size == m * 128, (
            f"Expected d1 size {m * 128}, got {analysis.dim_info['d1'].size}\n"
            f"Input shapes: a={a_shape}, b={b_shape}"
        )

        assert analysis.dim_info["d2"].size == n * 128, (
            f"Expected d2 size {n * 128}, got {analysis.dim_info['d2'].size}\n"
            f"Input shapes: a={a_shape}, b={b_shape}"
        )

    @settings(max_examples=100)
    @given(input_shapes=matmul_input_shapes_with_names())
    def test_input_names_preserved_in_signature(self, input_shapes: dict[str, tuple[int, int]]):
        """Property: Input parameter names are preserved in generated function signature.

        **Validates: Requirements 1.1, 1.3**

        For any user-defined function with named parameters, the generated tiled
        function signature SHALL contain those exact parameter names in the same
        order as the original function.

        Args:
            input_shapes: Dictionary mapping input names to their shapes.
        """
        input_names = list(input_shapes.keys())

        func_code = f"""
def matmul({', '.join(input_names)}):
    return nkigym.nc_matmul({input_names[0]}, {input_names[1]})
"""
        local_ns: dict = {"nkigym": nkigym}
        exec(func_code, local_ns)
        matmul_func = local_ns["matmul"]

        source = generate_tiled_source(matmul_func, input_shapes, output_dtype=np.float32)

        generated_params = _extract_function_signature_params(source)

        assert generated_params == input_names, (
            f"Parameter names not preserved.\n"
            f"Expected: {input_names}\n"
            f"Got: {generated_params}\n"
            f"Generated source:\n{source}"
        )

    @settings(max_examples=100)
    @given(input_shapes=matmul_input_shapes_with_names())
    def test_computed_vars_follow_tensor_n_pattern(self, input_shapes: dict[str, tuple[int, int]]):
        """Property: Computed variables use tensor_N naming pattern without suffixes.

        **Validates: Requirements 1.2, 2.1, 2.2, 2.3, 5.1**

        For any generated tiled code, all computed variable names (input slices,
        intermediate results, accumulators) SHALL match the pattern `tensor_\\d+`
        with no subgraph (`_sg`) or reduction (`_r`) suffixes.

        Args:
            input_shapes: Dictionary mapping input names to their shapes.
        """
        input_names = list(input_shapes.keys())

        func_code = f"""
def matmul({', '.join(input_names)}):
    return nkigym.nc_matmul({input_names[0]}, {input_names[1]})
"""
        local_ns: dict = {"nkigym": nkigym}
        exec(func_code, local_ns)
        matmul_func = local_ns["matmul"]

        source = generate_tiled_source(matmul_func, input_shapes, output_dtype=np.float32)

        assignment_pattern = re.compile(r"^\s+(\w+)\s*=", re.MULTILINE)
        all_assignments = assignment_pattern.findall(source)

        excluded_names = set(input_names) | {"output", "np"}
        computed_vars = [var for var in all_assignments if var not in excluded_names]

        tensor_n_pattern = re.compile(r"^tensor_\d+$")

        forbidden_suffix_pattern = re.compile(r"_sg\d*|_r\d*")

        for var in computed_vars:
            assert tensor_n_pattern.match(var), (
                f"Computed variable '{var}' does not match tensor_N pattern.\n"
                f"Expected pattern: tensor_\\d+ (e.g., tensor_0, tensor_1, ...)\n"
                f"Generated source:\n{source}"
            )

            assert not forbidden_suffix_pattern.search(var), (
                f"Computed variable '{var}' contains forbidden suffix (_sg or _r).\n"
                f"Variables should use simple tensor_N pattern without subgraph/reduction suffixes.\n"
                f"Generated source:\n{source}"
            )


class TestErrorHandling:
    """Tests for error handling and input validation.

    This class verifies that appropriate errors are raised for invalid inputs,
    including reserved names, unsupported operators, and invalid configurations.

    Attributes:
        None

    Example:
        Run error handling tests::

            pytest test/test_tiling.py::TestErrorHandling -v
    """

    def test_non_traced_tensor_return_raises_type_error(self):
        """Verify TypeError is raised when function returns non-TracedTensor.

        **Validates: Requirements 1.3**

        When a function returns a value that is not a TracedTensor (e.g., a raw
        numpy array or scalar), the system SHALL raise a TypeError with a
        descriptive message indicating the expected and actual return types.
        """

        def returns_raw_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Function that returns a raw numpy array instead of TracedTensor."""
            return np.zeros((128, 128))

        input_shapes = {"a": (128, 128), "b": (128, 128)}

        with pytest.raises(TypeError) as exc_info:
            analyze_dimension(returns_raw_array, input_shapes)

        error_message = str(exc_info.value)
        assert "TracedTensor" in error_message, f"Error message should mention TracedTensor.\nGot: {error_message}"

    def test_scalar_return_raises_type_error(self):
        """Verify TypeError is raised when function returns a scalar.

        **Validates: Requirements 1.3**

        When a function returns a scalar value instead of a TracedTensor,
        the system SHALL raise a TypeError.
        """

        def returns_scalar(a: np.ndarray, b: np.ndarray) -> float:
            """Function that returns a scalar instead of TracedTensor."""
            return 42.0

        input_shapes = {"a": (128, 128), "b": (128, 128)}

        with pytest.raises(TypeError) as exc_info:
            analyze_dimension(returns_scalar, input_shapes)

        error_message = str(exc_info.value)
        assert "TracedTensor" in error_message, f"Error message should mention TracedTensor.\nGot: {error_message}"
        assert "float" in error_message, f"Error message should mention the actual type 'float'.\nGot: {error_message}"

    def test_none_return_raises_type_error(self):
        """Verify TypeError is raised when function returns None.

        **Validates: Requirements 1.3**

        When a function returns None instead of a TracedTensor,
        the system SHALL raise a TypeError.
        """

        def returns_none(a: np.ndarray, b: np.ndarray) -> None:
            """Function that returns None instead of TracedTensor."""

        input_shapes = {"a": (128, 128), "b": (128, 128)}

        with pytest.raises(TypeError) as exc_info:
            analyze_dimension(returns_none, input_shapes)

        error_message = str(exc_info.value)
        assert "TracedTensor" in error_message, f"Error message should mention TracedTensor.\nGot: {error_message}"

    @settings(max_examples=100)
    @given(n=st.integers(min_value=0, max_value=10000))
    def test_tensor_n_names_rejected_as_reserved(self, n: int):
        """Property: Reserved tensor_N names are rejected with ValueError.

        **Validates: Requirements 1.3, 1.4**

        For any input name matching `tensor_\\d+`, the system SHALL raise a
        ValueError before tracing begins, since these names are reserved for
        generated intermediate variables.

        Args:
            n: Integer suffix for the tensor_N pattern.
        """
        reserved_name = f"tensor_{n}"

        input_shapes = {reserved_name: (128, 128), "b": (128, 128)}

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.matmul(a, b)

        with pytest.raises(ValueError) as exc_info:
            generate_tiled_source(matmul, input_shapes, output_dtype=np.float32)

        error_message = str(exc_info.value)
        assert reserved_name in error_message, (
            f"Error message should contain the reserved name '{reserved_name}'.\n" f"Got: {error_message}"
        )
        assert "reserved" in error_message.lower(), (
            f"Error message should indicate the name is reserved.\n" f"Got: {error_message}"
        )

    def test_reserved_output_name_rejected(self):
        """Verify reserved 'output' name is rejected with ValueError.

        **Validates: Requirements 1.3, 1.4**

        For any input name equal to "output", the system SHALL raise a
        ValueError before tracing begins.
        """
        input_shapes = {"output": (128, 128), "b": (128, 128)}

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.matmul(a, b)

        with pytest.raises(ValueError) as exc_info:
            generate_tiled_source(matmul, input_shapes, output_dtype=np.float32)

        error_message = str(exc_info.value)
        assert "output" in error_message, (
            f"Error message should contain the reserved name 'output'.\n" f"Got: {error_message}"
        )
        assert "reserved" in error_message.lower(), (
            f"Error message should indicate the name is reserved.\n" f"Got: {error_message}"
        )


class TestOpRegistry:
    """Tests for OP_REGISTRY expression generation functions.

    This class verifies that all operations in OP_REGISTRY correctly generate
    NumPy expressions and accumulation expressions for reduction tiling.

    The OP_REGISTRY dictionary maps operation names to NKIOp instances:
    - generate_expr: Creates the NumPy expression for the operation
    - reduce: Creates the in-place accumulation expression (for reduction ops)

    Attributes:
        None

    Example:
        Run operation registry tests::

            pytest test/test_tiling.py::TestOpRegistry -v
    """

    def test_nc_matmul_generate_expr(self):
        """Verify nc_matmul generates correct nkigym.nc_matmul expression.

        The nc_matmul operation SHALL generate an expression of the form
        'nkigym.nc_matmul(input1, input2)' for KM x KN layout inputs.
        """
        nki_op = OP_REGISTRY["nc_matmul"]
        inputs = ["a", "b"]
        expr = nki_op.generate_expr(inputs)

        assert expr == "nkigym.nc_matmul(a, b)", f"Expected 'nkigym.nc_matmul(a, b)', got '{expr}'"

    def test_nc_matmul_generate_expr_with_different_names(self):
        """Verify nc_matmul generates correct expression with arbitrary input names.

        The nc_matmul operation SHALL correctly substitute any valid input names
        into the generated expression.
        """
        nki_op = OP_REGISTRY["nc_matmul"]
        inputs = ["tensor_0", "tensor_1"]
        expr = nki_op.generate_expr(inputs)

        assert (
            expr == "nkigym.nc_matmul(tensor_0, tensor_1)"
        ), f"Expected 'nkigym.nc_matmul(tensor_0, tensor_1)', got '{expr}'"

    def test_nc_matmul_reduce(self):
        """Verify nc_matmul generates correct in-place accumulation expression.

        The nc_matmul reduce function SHALL generate an expression of the
        form 'result += nkigym.nc_matmul(input1, input2)' for additive accumulation.
        """
        nki_op = OP_REGISTRY["nc_matmul"]
        result_var = "output"
        inputs = ["a", "b"]
        expr = nki_op.reduce(result_var, inputs)

        assert expr == "output += nkigym.nc_matmul(a, b)", f"Expected 'output += nkigym.nc_matmul(a, b)', got '{expr}'"

    def test_nc_matmul_reduce_with_different_names(self):
        """Verify nc_matmul reduce with arbitrary variable names.

        The nc_matmul reduce function SHALL correctly substitute any valid
        variable names into the generated accumulation expression.
        """
        nki_op = OP_REGISTRY["nc_matmul"]
        result_var = "tensor_10"
        inputs = ["tensor_8", "tensor_9"]
        expr = nki_op.reduce(result_var, inputs)

        expected = "tensor_10 += nkigym.nc_matmul(tensor_8, tensor_9)"
        assert expr == expected, f"Expected '{expected}', got '{expr}'"

    def test_all_ops_have_op_name(self):
        """Verify all operations in OP_REGISTRY have correct op_name attribute.

        Each operation in OP_REGISTRY SHALL have an op_name attribute that
        matches its dictionary key.
        """
        for op_key, nki_op in OP_REGISTRY.items():
            assert (
                nki_op.op_name == op_key
            ), f"op_name mismatch for '{op_key}': expected '{op_key}', got '{nki_op.op_name}'"

    def test_all_ops_have_generate_expr(self):
        """Verify all operations in OP_REGISTRY have generate_expr method.

        Each operation in OP_REGISTRY SHALL have a generate_expr method
        that is callable.
        """
        for op_key, nki_op in OP_REGISTRY.items():
            assert hasattr(nki_op, "generate_expr"), f"'{op_key}' should have generate_expr method"
            assert callable(nki_op.generate_expr), f"'{op_key}' generate_expr should be callable"

    def test_expected_ops_present(self):
        """Verify nc_matmul is present in OP_REGISTRY.

        OP_REGISTRY SHALL contain an entry for nc_matmul.
        """
        assert "nc_matmul" in OP_REGISTRY, "nc_matmul should be in OP_REGISTRY"
