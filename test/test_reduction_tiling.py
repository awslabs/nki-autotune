"""Property-based tests for reduction dimension tiling.

This module contains property-based tests for reduction dimension tiling,
which handles the K dimension in matmul operations when K > 128. Tests verify:
- Reduction tile count computation
- Slice parameter correctness for reduction dimensions
- Generated code validity
- Parallel structure preservation
- Numerical equivalence of tiled functions

Run with: pytest test/test_reduction_tiling.py -v
"""

import numpy as np
import pytest
from conftest import matmul_shapes, matmul_shapes_with_reduction_tile_index
from hypothesis import given, settings

import nkigym
from nkigym.tiling import TILE_SIZE, analyze_dimension


class TestReductionTileCountComputation:
    """Property tests for reduction tile count computation.

    This class verifies that reduction tile counts are correctly computed
    based on the reduction dimension size. For matmul C[m,n] = A[m,k] @ B[k,n],
    the K dimension is the reduction dimension and its tile count determines
    how many partial products need to be accumulated.

    Attributes:
        None

    Example:
        Run reduction tile count tests::

            pytest test/test_reduction_tiling.py::TestReductionTileCountComputation -v

    **Feature: test-coverage-improvement, Property 3: Tile Count Computation Correctness**
    """

    @settings(max_examples=100)
    @given(shapes=matmul_shapes())
    def test_reduction_tile_count_equals_size_div_tile_size(self, shapes):
        """Property 3: Tile Count Computation Correctness.

        For any reduction dimension with size S, the computed reduction tile count
        SHALL equal S / TILE_SIZE (where TILE_SIZE = 128).

        Args:
            shapes: Tuple of (a_shape, b_shape) for matmul A @ B.

        **Validates: Requirements 1.5, 3.1**
        """
        a_shape, b_shape = shapes

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return nkigym.nc_matmul(a, b)

        analysis = analyze_dimension(matmul, {"a": a_shape, "b": b_shape})

        reduction_dims = analysis.get_reduction_dims()

        for dim_id in reduction_dims:
            dim_size = analysis.dim_info[dim_id].size
            expected_tile_count = dim_size // TILE_SIZE
            actual_tile_count = analysis.reduction_tile_counts[dim_id]

            assert actual_tile_count == expected_tile_count, (
                f"Reduction tile count mismatch for {dim_id}: "
                f"expected {expected_tile_count}, got {actual_tile_count}"
            )

    def test_minimum_reduction_tile_count(self, matmul_func):
        """Test minimum reduction tile count (K=128 yields 1 reduction tile).

        Edge case test for the smallest valid reduction dimension size.
        With K=128 (minimum valid size), there should be exactly 1 reduction tile.

        Args:
            matmul_func: Fixture providing the matmul function.

        **Validates: Requirements 1.5, 3.1**
        """
        a_shape = (128, 128)
        b_shape = (128, 128)

        analysis = analyze_dimension(matmul_func, {"a": a_shape, "b": b_shape})

        reduction_dims = analysis.get_reduction_dims()
        assert len(reduction_dims) == 1, "Matmul should have exactly one reduction dimension"

        reduction_dim_id = reduction_dims[0]
        actual_tile_count = analysis.reduction_tile_counts[reduction_dim_id]

        assert actual_tile_count == 1, f"Minimum reduction tile count should be 1 for K=128, got {actual_tile_count}"

    def test_maximum_reduction_tile_count_in_test_range(self, matmul_func):
        """Test maximum reduction tile count in test range (K=1024 yields 8 reduction tiles).

        Edge case test for the largest reduction dimension size in the test range.
        With K=1024, there should be exactly 8 reduction tiles (1024 / 128 = 8).

        Args:
            matmul_func: Fixture providing the matmul function.

        **Validates: Requirements 1.5, 3.1**
        """
        a_shape = (1024, 128)
        b_shape = (1024, 128)

        analysis = analyze_dimension(matmul_func, {"a": a_shape, "b": b_shape})

        reduction_dims = analysis.get_reduction_dims()
        assert len(reduction_dims) == 1, "Matmul should have exactly one reduction dimension"

        reduction_dim_id = reduction_dims[0]
        actual_tile_count = analysis.reduction_tile_counts[reduction_dim_id]

        assert actual_tile_count == 8, f"Maximum reduction tile count should be 8 for K=1024, got {actual_tile_count}"

    def test_tile_count_computation_for_parallel_and_reduction_dims(self, matmul_func):
        """Test tile count computation is correct for both parallel and reduction dimensions.

        Verifies that:
        - Parallel dimensions (M, N) have correct tile counts in tile_counts dict
        - Reduction dimension (K) has correct tile count in reduction_tile_counts dict

        Uses nc_matmul shapes (512, 256) @ (512, 384) which has:
        - K=512: 4 reduction tiles (first dim of both inputs)
        - M=256: 2 parallel tiles (second dim of a)
        - N=384: 3 parallel tiles (second dim of b)

        Args:
            matmul_func: Fixture providing the matmul function.

        **Validates: Requirements 1.5, 3.1**
        """
        a_shape = (512, 256)
        b_shape = (512, 384)

        analysis = analyze_dimension(matmul_func, {"a": a_shape, "b": b_shape})

        parallel_dims = analysis.get_parallel_dims()
        reduction_dims = analysis.get_reduction_dims()

        assert len(parallel_dims) == 2, "Matmul should have exactly two parallel dimensions"
        assert len(reduction_dims) == 1, "Matmul should have exactly one reduction dimension"

        expected_parallel_tiles = {256 // TILE_SIZE, 384 // TILE_SIZE}
        actual_parallel_tiles = {analysis.tile_counts[dim_id] for dim_id in parallel_dims}
        assert actual_parallel_tiles == expected_parallel_tiles, (
            f"Parallel tile counts mismatch: expected {expected_parallel_tiles}, " f"got {actual_parallel_tiles}"
        )

        reduction_dim_id = reduction_dims[0]
        expected_reduction_tiles = 512 // TILE_SIZE
        actual_reduction_tiles = analysis.reduction_tile_counts[reduction_dim_id]
        assert actual_reduction_tiles == expected_reduction_tiles, (
            f"Reduction tile count mismatch: expected {expected_reduction_tiles}, " f"got {actual_reduction_tiles}"
        )


class TestSliceParameterCorrectness:
    """Property tests for reduction slice parameter computation.

    This class verifies that slice parameters (offset and size) are correctly
    computed for each reduction tile position. For reduction dimension K with
    tile index i, the slice offset should be i * TILE_SIZE and size should be
    TILE_SIZE.

    Attributes:
        None

    Example:
        Run slice parameter tests::

            pytest test/test_reduction_tiling.py::TestSliceParameterCorrectness -v
    """

    @settings(max_examples=100)
    @given(shapes_and_index=matmul_shapes_with_reduction_tile_index())
    def test_slice_offset_equals_tile_index_times_tile_size(self, shapes_and_index):
        """Property 3: Slice Parameter Correctness.

        For any tensor with reduction dimension and any tile index i:
        - Slice offset along reduction dim equals i * TILE_SIZE
        - Slice size along reduction dim equals TILE_SIZE

        Args:
            shapes_and_index: Tuple of (a_shape, b_shape, reduction_tile_index).

        **Validates: Requirements 3.1, 3.2, 3.3**
        """
        a_shape, b_shape, reduction_tile_index = shapes_and_index

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return nkigym.nc_matmul(a, b)

        analysis = analyze_dimension(matmul, {"a": a_shape, "b": b_shape})

        reduction_dims = analysis.get_reduction_dims()
        assert len(reduction_dims) == 1, "Matmul should have exactly one reduction dimension"
        reduction_dim_id = reduction_dims[0]

        parallel_positions = list(analysis.iter_tile_positions())
        assert len(parallel_positions) > 0, "Should have at least one parallel position"
        _, parallel_pos = parallel_positions[0]

        reduction_pos = {reduction_dim_id: reduction_tile_index}

        slice_a = analysis.compute_reduction_slice_params("a", parallel_pos, reduction_pos)
        a_dims = analysis.tensor_dims["a"]

        for i, dim_id in enumerate(a_dims):
            if dim_id == reduction_dim_id:
                expected_offset = reduction_tile_index * TILE_SIZE
                expected_size = TILE_SIZE
                assert slice_a.offsets[i] == expected_offset, (
                    f"Tensor 'a' reduction dim offset mismatch: "
                    f"expected {expected_offset}, got {slice_a.offsets[i]}"
                )
                assert slice_a.sizes[i] == expected_size, (
                    f"Tensor 'a' reduction dim size mismatch: " f"expected {expected_size}, got {slice_a.sizes[i]}"
                )

        slice_b = analysis.compute_reduction_slice_params("b", parallel_pos, reduction_pos)
        b_dims = analysis.tensor_dims["b"]

        for i, dim_id in enumerate(b_dims):
            if dim_id == reduction_dim_id:
                expected_offset = reduction_tile_index * TILE_SIZE
                expected_size = TILE_SIZE
                assert slice_b.offsets[i] == expected_offset, (
                    f"Tensor 'b' reduction dim offset mismatch: "
                    f"expected {expected_offset}, got {slice_b.offsets[i]}"
                )
                assert slice_b.sizes[i] == expected_size, (
                    f"Tensor 'b' reduction dim size mismatch: " f"expected {expected_size}, got {slice_b.sizes[i]}"
                )

    @settings(max_examples=100)
    @given(shapes=matmul_shapes())
    def test_non_reduction_dims_unchanged(self, shapes):
        """Property 3 (continued): Non-reduction dimensions are unchanged.

        For tensors without a particular reduction dimension, slicing should be
        unchanged. For output tensor (which has no reduction dims), slicing
        should use parallel positions only.

        Args:
            shapes: Tuple of (a_shape, b_shape) for matmul A @ B.

        **Validates: Requirements 3.3**
        """
        a_shape, b_shape = shapes

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return nkigym.nc_matmul(a, b)

        analysis = analyze_dimension(matmul, {"a": a_shape, "b": b_shape})

        parallel_positions = list(analysis.iter_tile_positions())
        reduction_positions = list(analysis.iter_reduction_tile_positions())

        _, parallel_pos = parallel_positions[0]
        reduction_pos = reduction_positions[0] if reduction_positions else {}

        output_slice = analysis.compute_reduction_slice_params("output", parallel_pos, reduction_pos)
        output_dims = analysis.tensor_dims["output"]

        for i, dim_id in enumerate(output_dims):
            if dim_id in parallel_pos:
                tile_idx = parallel_pos[dim_id]
                expected_offset = tile_idx * TILE_SIZE
                expected_size = TILE_SIZE
                assert output_slice.offsets[i] == expected_offset, (
                    f"Output parallel dim offset mismatch: "
                    f"expected {expected_offset}, got {output_slice.offsets[i]}"
                )
                assert output_slice.sizes[i] == expected_size, (
                    f"Output parallel dim size mismatch: " f"expected {expected_size}, got {output_slice.sizes[i]}"
                )

    @pytest.mark.parametrize("k_size,expected_tiles", [(512, 4), (1024, 8)], ids=["K=512_4tiles", "K=1024_8tiles"])
    def test_slice_parameters_for_n_reduction_tiles(self, k_size, expected_tiles, matmul_func):
        """Test slice parameters for N reduction tiles.

        Verifies that for nc_matmul with K=k_size (expected_tiles reduction tiles),
        each reduction tile position has correct slice parameters:
        - Slice offset equals tile_index * TILE_SIZE (128)
        - Slice size equals TILE_SIZE (128)

        Args:
            k_size: Size of the K (reduction) dimension.
            expected_tiles: Expected number of reduction tiles.
            matmul_func: Fixture providing the matmul function.

        **Validates: Requirements 3.3**
        """
        a_shape = (k_size, 128)
        b_shape = (k_size, 128)

        analysis = analyze_dimension(matmul_func, {"a": a_shape, "b": b_shape})

        reduction_dims = analysis.get_reduction_dims()
        assert len(reduction_dims) == 1, "Matmul should have exactly one reduction dimension"
        reduction_dim_id = reduction_dims[0]

        num_reduction_tiles = analysis.reduction_tile_counts[reduction_dim_id]
        assert num_reduction_tiles == expected_tiles, (
            f"Expected {expected_tiles} reduction tiles for K={k_size}, " f"got {num_reduction_tiles}"
        )

        parallel_positions = list(analysis.iter_tile_positions())
        _, parallel_pos = parallel_positions[0]

        for tile_index in range(expected_tiles):
            reduction_pos = {reduction_dim_id: tile_index}

            slice_a = analysis.compute_reduction_slice_params("a", parallel_pos, reduction_pos)
            a_dims = analysis.tensor_dims["a"]

            for i, dim_id in enumerate(a_dims):
                if dim_id == reduction_dim_id:
                    expected_offset = tile_index * TILE_SIZE
                    expected_size = TILE_SIZE
                    assert slice_a.offsets[i] == expected_offset, (
                        f"Tensor 'a' reduction tile {tile_index} offset mismatch: "
                        f"expected {expected_offset}, got {slice_a.offsets[i]}"
                    )
                    assert slice_a.sizes[i] == expected_size, (
                        f"Tensor 'a' reduction tile {tile_index} size mismatch: "
                        f"expected {expected_size}, got {slice_a.sizes[i]}"
                    )

            slice_b = analysis.compute_reduction_slice_params("b", parallel_pos, reduction_pos)
            b_dims = analysis.tensor_dims["b"]

            for i, dim_id in enumerate(b_dims):
                if dim_id == reduction_dim_id:
                    expected_offset = tile_index * TILE_SIZE
                    expected_size = TILE_SIZE
                    assert slice_b.offsets[i] == expected_offset, (
                        f"Tensor 'b' reduction tile {tile_index} offset mismatch: "
                        f"expected {expected_offset}, got {slice_b.offsets[i]}"
                    )
                    assert slice_b.sizes[i] == expected_size, (
                        f"Tensor 'b' reduction tile {tile_index} size mismatch: "
                        f"expected {expected_size}, got {slice_b.sizes[i]}"
                    )


class TestGeneratedCodeValidity:
    """Property tests for generated code validity.

    This class verifies that generated tiled code is syntactically valid Python
    that executes without errors and produces correctly-shaped output arrays.

    Attributes:
        None

    Example:
        Run generated code validity tests::

            pytest test/test_reduction_tiling.py::TestGeneratedCodeValidity -v
    """

    @settings(max_examples=100)
    @given(shapes=matmul_shapes())
    def test_generated_code_is_valid_python(self, shapes):
        """Property 4: Generated Code Validity.

        For any valid matmul shapes, the generated reduction-tiled source code SHALL:
        - Be syntactically valid Python
        - Execute without runtime errors when called with appropriately-shaped arrays
        - Produce an output array of the expected shape

        Args:
            shapes: Tuple of (a_shape, b_shape) for matmul A @ B.

        **Validates: Requirements 6.3**
        """
        a_shape, b_shape = shapes

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return nkigym.nc_matmul(a, b)

        from nkigym.tiling import generate_tiled_function, generate_tiled_source

        source = generate_tiled_source(matmul, {"a": a_shape, "b": b_shape})
        try:
            compile(source, "<string>", "exec")
        except SyntaxError as e:
            raise AssertionError(f"Generated code has syntax error: {e}\n\nSource:\n{source}")

        tiled_func = generate_tiled_function(matmul, {"a": a_shape, "b": b_shape})

        np.random.seed(42)
        a = np.random.randn(*a_shape).astype(np.float32)
        np.random.seed(43)
        b = np.random.randn(*b_shape).astype(np.float32)

        try:
            result = tiled_func(a, b)
        except Exception as e:
            raise AssertionError(f"Generated function raised error: {e}\n\nSource:\n{source}")

        expected_shape = (a_shape[1], b_shape[1])
        assert result.shape == expected_shape, f"Output shape mismatch: expected {expected_shape}, got {result.shape}"

    @pytest.mark.skip(reason="NKIMatmul always implements reduce")
    def test_notimplementederror_when_reduce_missing(self, matmul_func):
        """Verify NotImplementedError is raised when reduce is missing.

        This test is skipped because NKIMatmul always implements reduce.
        The test was designed for the old OpSemantics system where reduce
        could be None.

        Args:
            matmul_func: Fixture providing the matmul function.
        """


class TestParallelStructurePreservation:
    """Property tests for parallel structure preservation.

    This class verifies that reduction tiling preserves the parallel structure
    of the tiled function, including the number of subgraphs, parallel tile
    counts, and parallel slice parameters.

    Attributes:
        None

    Example:
        Run parallel structure preservation tests::

            pytest test/test_reduction_tiling.py::TestParallelStructurePreservation -v
    """

    @settings(max_examples=100)
    @given(shapes=matmul_shapes())
    def test_reduction_tiling_preserves_parallel_structure(self, shapes):
        """Property 5: Parallel Structure Preservation.

        For any function, the reduction tiling pass SHALL preserve:
        - The number of subgraphs (determined by parallel dimensions)
        - The parallel dimension tile counts
        - The parallel dimension slice parameters

        Args:
            shapes: Tuple of (a_shape, b_shape) for matmul A @ B.

        **Validates: Requirements 2.4**
        """
        a_shape, b_shape = shapes

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return nkigym.nc_matmul(a, b)

        analysis = analyze_dimension(matmul, {"a": a_shape, "b": b_shape})

        expected_num_subgraphs = 1
        for dim_id in analysis.get_parallel_dims():
            expected_num_subgraphs *= analysis.tile_counts[dim_id]

        assert (
            analysis.num_subgraphs == expected_num_subgraphs
        ), f"num_subgraphs mismatch: expected {expected_num_subgraphs}, got {analysis.num_subgraphs}"

        for dim_id in analysis.get_parallel_dims():
            dim_size = analysis.dim_info[dim_id].size
            expected_tile_count = (dim_size + TILE_SIZE - 1) // TILE_SIZE
            actual_tile_count = analysis.tile_counts[dim_id]

            assert actual_tile_count == expected_tile_count, (
                f"Parallel tile count mismatch for {dim_id}: "
                f"expected {expected_tile_count}, got {actual_tile_count}"
            )

        for subgraph_idx, parallel_pos in analysis.iter_tile_positions():
            for tensor_id in ["a", "b", "output"]:
                if tensor_id not in analysis.tensor_dims:
                    continue

                slice_info = analysis.slice_params[tensor_id][subgraph_idx]
                tensor_dims = analysis.tensor_dims[tensor_id]

                for i, dim_id in enumerate(tensor_dims):
                    if dim_id in parallel_pos:
                        tile_idx = parallel_pos[dim_id]
                        expected_offset = tile_idx * TILE_SIZE
                        expected_size = TILE_SIZE

                        assert slice_info.offsets[i] == expected_offset, (
                            f"Parallel slice offset mismatch for {tensor_id}[{dim_id}] "
                            f"at subgraph {subgraph_idx}: "
                            f"expected {expected_offset}, got {slice_info.offsets[i]}"
                        )
                        assert slice_info.sizes[i] == expected_size, (
                            f"Parallel slice size mismatch for {tensor_id}[{dim_id}] "
                            f"at subgraph {subgraph_idx}: "
                            f"expected {expected_size}, got {slice_info.sizes[i]}"
                        )


class TestNumericalEquivalence:
    """Property tests for numerical equivalence of reduction-tiled functions.

    This class verifies that tiled functions produce numerically equivalent
    results to the original NumPy functions within floating-point tolerance.

    Attributes:
        None

    Example:
        Run numerical equivalence tests::

            pytest test/test_reduction_tiling.py::TestNumericalEquivalence -v
    """

    @settings(max_examples=100)
    @given(shapes=matmul_shapes())
    def test_tiled_matmul_equals_nc_matmul(self, shapes):
        """Property 1: Numerical Equivalence.

        For any valid nc_matmul shapes, the tiled function produces output numerically
        equivalent (within floating-point tolerance) to nkigym.nc_matmul.

        Uses relaxed tolerances (rtol=1e-4, atol=1e-4) to account for floating-point
        differences introduced by reduction tiling.

        Args:
            shapes: Tuple of (a_shape, b_shape) for matmul A @ B.

        **Validates: Requirements 5.1, 5.2**
        """
        a_shape, b_shape = shapes

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return nkigym.nc_matmul(a, b)

        from nkigym.tiling import generate_tiled_function

        np.random.seed(42)
        a = np.random.randn(*a_shape).astype(np.float32)
        np.random.seed(43)
        b = np.random.randn(*b_shape).astype(np.float32)

        expected = nkigym.nc_matmul(a, b)

        tiled_func = generate_tiled_function(matmul, {"a": a_shape, "b": b_shape})
        actual = tiled_func(a, b)

        assert actual.shape == expected.shape, f"Shape mismatch: expected {expected.shape}, got {actual.shape}"

        np.testing.assert_allclose(
            actual, expected, rtol=1e-4, atol=1e-4, err_msg=f"Numerical mismatch for shapes {a_shape} @ {b_shape}"
        )
