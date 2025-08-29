import math
from typing import Tuple

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
import tabulate

from autotune.core.tensor import SBUFTensor
from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSORS_DTYPE


class GEMMCompatibility:
    """
    Validates compatibility of input shapes and parameters for GEMM operations.

    This class checks whether the given matrix dimensions can be properly
    divided into blocks and tiles according to hardware constraints.
    """

    def __init__(self, transposed_lhs: bool) -> None:
        """
        Initialize GEMM compatibility checker.

        Args:
            transposed_lhs: Whether the LHS matrix should be interpreted as transposed.
        """
        self.transposed_lhs = transposed_lhs
        # Single tile sizes (hardware constants)
        self.TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
        self.TILE_N = nl.tile_size.gemm_moving_fmax  # 512
        self.TILE_K = nl.tile_size.pmax  # 128

    def __call__(self, input_tensors: INPUT_TENSORS_DTYPE, kernel_kwargs: KERNEL_KWARGS_DTYPE):
        """
        Check compatibility of input tensors for GEMM operation with specified parameters.

        Args:
            input_tensors: Tuple containing (lhs, rhs) matrices for GEMM operation.
                lhs_shape: Shape of left-hand side matrix, either (M,K) or (batch,M,K)
                           If transposed_lhs is True, dimensions are interpreted as (K,M) or (batch,K,M)
                rhs_shape: Shape of right-hand side matrix, expected to be (K,N)
            kernel_kwargs: Dictionary containing kernel parameters:
                NUM_BLOCK_M: Number of blocks in M dimension
                NUM_BLOCK_N: Number of blocks in N dimension
                NUM_BLOCK_K: Number of blocks in K dimension, or None to skip K blocking

        Raises:
            Exception: If matrix shapes are incompatible or cannot be properly divided
                       into the specified number of blocks and tiles.
        """
        lhs, rhs = input_tensors
        lhs_shape = lhs.shape
        rhs_shape = rhs.shape
        NUM_BLOCK_M: int = kernel_kwargs.get("NUM_BLOCK_M", 1)
        NUM_BLOCK_N: int = kernel_kwargs.get("NUM_BLOCK_N", 1)
        NUM_BLOCK_K: int = kernel_kwargs.get("NUM_BLOCK_K", 1)
        if len(rhs_shape) != 2:
            raise Exception(f"Expecting (K, N) in RHS. Received {rhs_shape}.")

        self.batch = None
        if len(lhs_shape) == 2:
            if self.transposed_lhs:
                self.K, self.M = lhs_shape
            else:
                self.M, self.K = lhs_shape
        elif len(lhs_shape) == 3:
            if self.transposed_lhs:
                self.batch, self.K, self.M = lhs_shape
            else:
                self.batch, self.M, self.K = lhs_shape
        else:
            raise Exception(f"lhs_shape must be either 2D or (batch, 2D). Received {lhs_shape}.")

        K_, self.N = rhs_shape

        # Validate dimensions > 0
        for dim_name, dim_value in [("M", self.M), ("K", self.K), ("N", self.N)]:
            if dim_value <= 0:
                raise Exception(f"Dimension {dim_name} must be positive, got {dim_value}")

        # Number of blocks (None means no blocking in that dimension)
        self.NUM_BLOCK_M = NUM_BLOCK_M
        self.NUM_BLOCK_N = NUM_BLOCK_N
        self.NUM_BLOCK_K = NUM_BLOCK_K

        # Calculate derived sizes
        self._calculate_sizes()

        # Validate contraction dimension matches
        if self.K != K_:
            raise Exception(
                f"Contraction dimension mismatch: LHS {lhs_shape} has K={self.K}, RHS {rhs_shape} has K={K_}"
            )

        # Validate dimensions
        # self._check()

    def _calculate_sizes(self) -> None:
        """
        Calculate derived sizes for each dimension (M, N, K).

        NUM_BLOCK_X * TILES_IN_BLOCK_X * TILE_X = X
        BLOCK_X = TILES_IN_BLOCK_X * TILE_X
        TILES_IN_X = NUM_BLOCK_X * TILES_IN_BLOCK_X

        For each dimension, calculates:
        - Block size (dimension divided by number of blocks)
        - Number of tiles in each block (block size divided by tile size)
        - Total number of tiles in the dimension (dimension size divided by tile size)

        Results are stored as attributes on the class instance.
        """
        # Calculate sizes for M dimension
        num_block_m = 1 if self.NUM_BLOCK_M is None else self.NUM_BLOCK_M
        self.TILES_IN_BLOCK_M = math.ceil(self.M / num_block_m / self.TILE_M)
        self.BLOCK_M = int(self.TILES_IN_BLOCK_M * self.TILE_M)
        self.TILES_IN_M = int(num_block_m * self.TILES_IN_BLOCK_M)

        # Calculate sizes for N dimension
        num_block_n = 1 if self.NUM_BLOCK_N is None else self.NUM_BLOCK_N
        self.TILES_IN_BLOCK_N = math.ceil(self.N / num_block_n / self.TILE_N)
        self.BLOCK_N = int(self.TILES_IN_BLOCK_N * self.TILE_N)
        self.TILES_IN_N = int(num_block_n * self.TILES_IN_BLOCK_N)

        # Calculate sizes for K dimension
        num_block_k = 1 if self.NUM_BLOCK_K is None else self.NUM_BLOCK_K
        self.TILES_IN_BLOCK_K = math.ceil(self.K / num_block_k / self.TILE_K)
        self.BLOCK_K = int(self.TILES_IN_BLOCK_K * self.TILE_K)
        self.TILES_IN_K = int(num_block_k * self.TILES_IN_BLOCK_K)

    def _check(self) -> None:
        """
        Validate that dimensions can be evenly divided into blocks and tiles.

        Verifies that for each dimension (M, N, K):
        1. The dimension size can be evenly divided into the specified number of blocks
        2. Each block can be evenly divided into tiles

        Raises:
            Exception: If any dimension cannot be evenly divided as required.
        """
        for dimension in ["M", "N", "K"]:
            size = getattr(self, f"{dimension}")
            num_block = getattr(self, f"NUM_BLOCK_{dimension}")

            # Handle dimensions with no blocking
            if num_block is None:
                num_block = 1

            tiles_in_block = getattr(self, f"TILES_IN_BLOCK_{dimension}")
            tile_size = getattr(self, f"TILE_{dimension}")

            # Check even division
            if num_block * tiles_in_block * tile_size != size:
                raise Exception(
                    f"{dimension} size {size} cannot be divided evenly into "
                    f"{num_block} blocks * {tiles_in_block} tiles * {tile_size}"
                )

    def __repr__(self) -> str:
        """
        Return a string representation of the GEMM compatibility checker as a table.

        Returns:
            A formatted table showing the configuration parameters.
        """
        header = f"GEMMCompatibility(transposed_lhs={self.transposed_lhs})"

        # Check if dimensions have been set (after __call__ has been invoked)
        if not hasattr(self, "M"):
            return f"{header}\n(Dimensions not yet set - call the object with matrices to validate)"

        # Format batch info if available
        batch_info = f" (batch={self.batch})" if self.batch is not None else ""

        # Create table data
        table_data = [
            ["Dimension", self.M, self.N, self.K],
            ["Num blocks", self.NUM_BLOCK_M, self.NUM_BLOCK_N, self.NUM_BLOCK_K],
            ["Block size", self.BLOCK_M, self.BLOCK_N, self.BLOCK_K],
            ["Tiles/block", self.TILES_IN_BLOCK_M, self.TILES_IN_BLOCK_N, self.TILES_IN_BLOCK_K],
            ["Tile size", self.TILE_M, self.TILE_N, self.TILE_K],
        ]

        # Generate the table with headers
        table = tabulate.tabulate(table_data, headers=["Parameter", "M", "N", "K"], tablefmt="grid")

        return f"{header}{batch_info}\n{table}"


def matmul_tensors(lhs_tiles: SBUFTensor, rhs_tiles: SBUFTensor, result_tiles: SBUFTensor, tile_transposed_lhs: bool):
    """
    Perform tiled matrix multiplication between SBUF tiles.

    Computes result_tiles += matmul(lhs_tiles, rhs_tiles) for the overlapping regions within each block.

    Args:
        lhs_tiles: Left-hand side matrix tiles stored in SBUF memory
        rhs_tiles: Right-hand side matrix tiles stored in SBUF memory
        result_tiles: Output matrix tiles stored in SBUF memory where results
            will be accumulated
        tile_transposed_lhs: (bool) - Whether lhs_tiles is transposed at the tile level.
        Note that this is not the same as lhsT_tiles.
    """
    if tile_transposed_lhs:
        TILE_M, _, _, TILE_K = lhs_tiles.tensor.shape
    else:
        TILE_K, _, _, TILE_M = lhs_tiles.tensor.shape
    _TILE_K, _, _, TILE_N = rhs_tiles.tensor.shape
    _TILE_M, _, _, _TILE_N = result_tiles.tensor.shape
    assert (
        TILE_K == _TILE_K
    ), f"lhs_tiles {lhs_tiles.tensor.shape} TILE_K mismatch with rhs_tiles {rhs_tiles.tensor.shape}"
    assert (
        TILE_M == _TILE_M and TILE_N == _TILE_N
    ), f"result_tiles {result_tiles.tensor.shape} shape mismatch with lhs_tiles {lhs_tiles.tensor.shape} @ rhs_tiles {rhs_tiles.tensor.shape}"
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]

    for tile_id_M in nl.affine_range(num_M_tiles):
        lhs_M_tile_index = lhs_M_tile_start + tile_id_M
        for tile_id_N in nl.affine_range(num_N_tiles):
            rhs_N_tile_index = rhs_N_tile_start + tile_id_N
            """
            Use PSUM buffer to accumulate into a single hardware tile
            """
            result_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for tile_id_K in nl.affine_range(num_K_tiles):
                lhs_K_tile_index = lhs_K_tile_start + tile_id_K
                rhs_K_tile_index = rhs_K_tile_start + tile_id_K
                lhs_tile = lhs.read_tile(tile_indices={"M": lhs_M_tile_index, "K": lhs_K_tile_index})
                rhs_tile = rhs.read_tile(tile_indices={"K": rhs_K_tile_index, "N": rhs_N_tile_index})
                result_tile += nisa.nc_matmul(lhs_tile, rhs_tile)
            result.tensor[
                idx_res.p, result_M_tile_start + tile_id_M, result_N_tile_start + tile_id_N, idx_res.x
            ] += result_tile[idx_res.p, idx_res.x]


def matmul_blocks_lhsT(
    lhsT_block,
    local_lhsT_tile_ofs: Tuple[int, int],
    rhs_block,
    local_rhs_tile_ofs: Tuple[int, int],
    result_block,
    local_result_tile_ofs: Tuple[int, int],
    num_tiles: Tuple[int, int, int],
    gemm_shape: Tuple[int, int, int],
    lhsT_global_tile_ofs: Tuple[int, int],
    rhs_global_tile_ofs: Tuple[int, int],
):
    """
    Perform tiled matrix multiplication between transposed left-hand side and right-hand side matrices.

    This function computes the matrix multiplication lhsT @ rhs, where lhsT is already in transposed format.
    Results are accumulated into the result_block at the specified offset position. The multiplication
    is performed by iterating over tiles in each dimension and accumulating results along the K dimension.
    This implementation optimizes performance by operating on blocked matrices for hardware efficiency.

    Args:
        lhsT_block: Left-hand side matrix block in transposed format. (K, M)
        rhs_block: Right-hand side matrix block. (K, N)
        result_block: Output matrix block where results are accumulated. (M, N)
        local_lhsT_tile_ofs: lhsT_K_tile_start, lhsT_M_tile_start
        local_rhs_tile_ofs: rhs_K_tile_start, rhs_N_tile_start
        local_result_tile_ofs: result_M_tile_start, result_N_tile_start
        num_tiles: number of M, N, K tiles to compute
    """
    TILE_K, _, _, TILE_M = lhsT_block.shape
    _TILE_K, _, _, TILE_N = rhs_block.shape
    _TILE_M, _, _, _TILE_N = result_block.shape
    assert TILE_K == _TILE_K, f"lhsT_block {lhsT_block.shape} tile K mismatch with rhs_block {rhs_block.shape}"
    assert (
        TILE_M == _TILE_M and TILE_N == _TILE_N
    ), f"result_block {result_block.shape} shape mismatch with lhsT_block {lhsT_block.shape} @ rhs_block {rhs_block.shape}"
    idx_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
    idx_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    lhsT_K_tile_start, lhsT_M_tile_start = local_lhsT_tile_ofs
    rhs_K_tile_start, rhs_N_tile_start = local_rhs_tile_ofs
    result_M_tile_start, result_N_tile_start = local_result_tile_ofs
    num_M_tiles, num_N_tiles, num_K_tiles = num_tiles

    for tile_id_M in nl.affine_range(num_M_tiles):
        for tile_id_N in nl.affine_range(num_N_tiles):
            result_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            """
            Use PSUM buffer to accumulate into a single hardware tile
            """
            for tile_id_K in nl.affine_range(num_K_tiles):
                result_tile += nisa.nc_matmul(
                    lhsT_block[idx_lhsT.p, lhsT_K_tile_start + tile_id_K, lhsT_M_tile_start + tile_id_M, idx_lhsT.x],
                    rhs_block[idx_rhs.p, rhs_K_tile_start + tile_id_K, rhs_N_tile_start + tile_id_N, idx_rhs.x],
                )
            result_block[
                idx_res.p, result_M_tile_start + tile_id_M, result_N_tile_start + tile_id_N, idx_res.x
            ] += result_tile[idx_res.p, idx_res.x]


def matmul_blocks_tile_transposed_lhs(
    tileT_lhs_block,
    lhs_tile_ofs: Tuple[int, int],
    rhs_block,
    rhs_tile_ofs: Tuple[int, int],
    result_block,
    result_tile_ofs: Tuple[int, int],
    num_tiles: Tuple[int, int, int],
):
    """
    Accumulate matmul result tiles between lhs and rhs into result_block
    LHS is transposed at the tile level.
    Note that this is not the same as lhsT.
    'matmul_block' module computes lhsT @ rhs.

    Args:
    tileT_lhs_block: TILE_M, TILES_IN_M, TILES_IN_K, TILE_K. Tile transposed LHS block.
    rhs_block: TILE_K, TILES_IN_K, TILES_IN_N, TILE_N
    result_block : TILE_M, TILE_N, TILES_IN_M, TILES_IN_N
    """
    TILE_M, _, _, TILE_K = tileT_lhs_block.shape
    _TILE_K, _, _, TILE_N = rhs_block.shape
    _TILE_M, _, _, _TILE_N = result_block.shape
    assert (
        TILE_K == _TILE_K
    ), f"tileT_lhs_block {tileT_lhs_block.shape} tile K mismatch with rhs_block {rhs_block.shape}"
    assert (
        TILE_M == _TILE_M and TILE_N == _TILE_N
    ), f"result_block {result_block.shape} shape mismatch with tileT_lhs_block {tileT_lhs_block.shape} @ rhs_block {rhs_block.shape}"
    idx_lhs = nl.mgrid[0:TILE_M, 0:TILE_K]
    idx_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    lhs_M_tile_start, lhs_K_tile_start = lhs_tile_ofs
    rhs_K_tile_start, rhs_N_tile_start = rhs_tile_ofs
    result_M_tile_start, result_N_tile_start = result_tile_ofs
    num_M_tiles, num_N_tiles, num_K_tiles = num_tiles

    for tile_id_M in nl.affine_range(num_M_tiles):
        for tile_id_N in nl.affine_range(num_N_tiles):
            result_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            """
            Use PSUM buffer to accumulate into a single hardware tile
            """
            for tile_id_K in nl.affine_range(num_K_tiles):
                result_tile += nisa.nc_matmul(
                    tileT_lhs_block[idx_lhs.p, lhs_M_tile_start + tile_id_M, lhs_K_tile_start + tile_id_K, idx_lhs.x],
                    rhs_block[idx_rhs.p, rhs_K_tile_start + tile_id_K, rhs_N_tile_start + tile_id_N, idx_rhs.x],
                )
            result_block[
                idx_res.p, result_M_tile_start + tile_id_M, result_N_tile_start + tile_id_N, idx_res.x
            ] += result_tile[idx_res.p, idx_res.x]


class GEMMCorrectness:
    def __init__(self, transposed_lhs: bool) -> None:
        self.transposed_lhs = transposed_lhs

    def __call__(
        self,
        input_tensors: INPUT_TENSORS_DTYPE,
        kernel_kwargs: KERNEL_KWARGS_DTYPE,
        nki_out_tensors: OUTPUT_TENSORS_DTYPE,
    ):
        data_type = np.float32
        atol, rtol = 1e-5, 1e-2
        lhs, rhs = input_tensors
        if self.transposed_lhs:
            golden = nl.static_cast(lhsT_rhs_gemm_np(lhs, rhs), data_type)
        else:
            golden = nl.static_cast(lhs_rhs_gemm_np(lhs, rhs), data_type)
        nki_out_tensor = nl.static_cast(nki_out_tensors[0], data_type)

        # Calculate absolute and relative differences
        abs_diff = np.abs(nki_out_tensor - golden)
        # Avoid division by zero in relative difference calculation
        rel_diff = np.divide(abs_diff, np.abs(golden), out=np.zeros_like(abs_diff), where=np.abs(golden) != 0)

        # Calculate tolerance threshold using numpy's allclose formula
        tolerance = atol + rtol * np.abs(golden)
        mismatches = abs_diff > tolerance
        total_mismatches = np.sum(mismatches)
        total_elements = golden.size

        if total_mismatches > 0:
            # Calculate statistics
            mismatch_percentage = (total_mismatches / total_elements) * 100
            max_abs_diff = np.max(abs_diff)
            max_rel_diff = np.max(rel_diff)

            # Generate error message with statistics and mismatch regions
            regions_summary = self._generate_mismatch_summary(mismatches)

            err_msg = (
                f"Mismatched elements: {total_mismatches} / {total_elements} ({mismatch_percentage:.6f}%)\n"
                f"Max absolute difference: {max_abs_diff}\n"
                f"Max relative difference: {max_rel_diff}\n"
                f"{regions_summary}"
            )

            # Raise custom assertion error
            raise AssertionError(err_msg)

    def _generate_mismatch_summary(self, mismatches):
        """Generate a summary of contiguous regions with mismatches."""
        if len(mismatches.shape) == 2:  # For 2D arrays
            return self._summarize_2d_mismatches(mismatches)
        else:
            # For other dimensions
            return self._summarize_nd_mismatches(mismatches)

    def _summarize_2d_mismatches(self, mismatches):
        """Summarize mismatches in 2D arrays as contiguous regions, sorted by size."""
        total_mismatches = np.sum(mismatches)

        if total_mismatches == 0:
            return "No mismatches found."

        if total_mismatches == 1:
            row, col = np.where(mismatches)
            return f"Only element [{row[0]}, {col[0]}] is wrong."

        # Find contiguous regions
        region_info = []  # Will store (size, r_start, c_start, r_end, c_end) tuples
        rows, cols = mismatches.shape

        # Process the array to find rectangular regions
        visited = np.zeros_like(mismatches, dtype=bool)

        for r in range(rows):
            for c in range(cols):
                if mismatches[r, c] and not visited[r, c]:
                    # Find the largest rectangle starting at (r,c)
                    max_r, max_c = r, c

                    # Extend rows
                    while max_r + 1 < rows and mismatches[max_r + 1, c]:
                        max_r += 1

                    # Find the maximum width for this range of rows
                    width = 1
                    while c + width < cols:
                        can_extend = True
                        for row_idx in range(r, max_r + 1):
                            if not mismatches[row_idx, c + width]:
                                can_extend = False
                                break
                        if can_extend:
                            width += 1
                        else:
                            break

                    # Mark this region as visited
                    visited[r : max_r + 1, c : c + width] = True

                    # Calculate region size
                    region_size = (max_r - r + 1) * width

                    # Add region info: (size, r_start, c_start, r_end, c_end)
                    region_info.append((region_size, r, c, max_r, c + width - 1))

        # Sort regions by size (descending) and then by coordinates (ascending)
        # For ties in size, sort by row_start, then col_start
        region_info.sort(key=lambda x: (-x[0], x[1], x[2]))

        # Format region strings
        region_strings = []
        for i, (size, r_start, c_start, r_end, c_end) in enumerate(region_info):
            # Only display top 10 regions if there are more than 10
            if i >= 10 and len(region_info) > 10:
                remaining = len(region_info) - 10
                region_strings.append(f"... {remaining} more regions not shown")
                break

            if r_start == r_end and c_start == c_end:
                region_strings.append(f"[{r_start}, {c_start}] (size: {size})")
            else:
                region_strings.append(f"[{r_start}:{r_end+1}, {c_start}:{c_end+1}] (size: {size})")

        if len(region_strings) == 1:
            return f"Elements {region_strings[0]} are wrong."
        else:
            total_regions = len(region_info)
            header = f"Found {total_regions} mismatched regions, sorted by size (largest first):"
            return f"{header}\n" + "\n".join(region_strings)

    def _summarize_nd_mismatches(self, mismatches):
        """Handle mismatches in arrays with dimensions other than 2."""
        total_mismatches = np.sum(mismatches)
        if total_mismatches == 1:
            coords = np.where(mismatches)
            coord_str = ", ".join(str(dim[0]) for dim in coords)
            return f"Only element [{coord_str}] is wrong."

        # For higher dimensions, just report the total and some examples
        coords = np.where(mismatches)
        # Get up to 5 examples
        examples = []
        for i in range(min(5, total_mismatches)):
            example = "[" + ", ".join(str(dim[i]) for dim in coords) + "]"
            examples.append(example)

        example_str = ", ".join(examples)
        if total_mismatches > 5:
            example_str += ", ..."

        return f"Found {total_mismatches} mismatches. Examples: {example_str}"


def lhs_rhs_gemm_np(lhs, rhs):
    """
    Calculate the general matrix multiplication (GEMM) between lhs and rhs.

    Parameters:
    -----------
    lhs : numpy.ndarray
        Left-hand side matrix or tensor. Can have an extra batch dimension.
    rhs : numpy.ndarray
        Right-hand side matrix.

    Returns:
    --------
    numpy.ndarray
        Result of the matrix multiplication.
    """
    return np.matmul(lhs, rhs)


def lhsT_rhs_gemm_np(lhsT, rhs):
    """
    Calculate the general matrix multiplication (GEMM) between lhsT and rhs.

    Parameters:
    -----------
    lhs : numpy.ndarray
        Left-hand side matrix or tensor. Can have an extra batch dimension.
    rhs : numpy.ndarray
        Right-hand side matrix.

    Returns:
    --------
    numpy.ndarray
        Result of the matrix multiplication.
    """
    if len(lhsT.shape) == 2:
        lhs = np.transpose(lhsT, (1, 0))
    elif len(lhsT.shape) == 3:  # Batch dimension exists
        lhs = np.transpose(lhsT, (0, 2, 1))
    else:
        raise NotImplementedError(f"lhsT shape {lhsT.shape} is not supported in GEMM.")
    return np.matmul(lhs, rhs)


def blocked_gemm_np_mkn(lhs, rhs, NUM_BLOCK_M: int, NUM_BLOCK_N: int, NUM_BLOCK_K: int):
    """
    GEMM algorithm with vectorized block processing using MKN loop ordering.
    """
    batch, M, K = lhs.shape
    K_check, N = rhs.shape

    if K != K_check:
        raise ValueError(f"Incompatible dimensions: lhs K dimension is {K} and rhs K dimension is {K_check}")

    # Calculate block sizes in each dimension
    BLOCK_M = M // NUM_BLOCK_M
    BLOCK_N = N // NUM_BLOCK_N
    BLOCK_K = K // NUM_BLOCK_K

    # Initialize output matrix with the input data type
    output = np.zeros((batch, M, N), dtype=lhs.dtype)

    # Process each batch
    for b in range(batch):
        # Process blocks of M dimension
        for block_m in range(NUM_BLOCK_M):
            m_start = block_m * BLOCK_M
            m_end = min(M, (block_m + 1) * BLOCK_M)
            m_size = m_end - m_start

            # Initialize accumulator for the entire M block with higher precision (float64)
            outputs = np.zeros((m_size, N), dtype=np.float64)

            # Process K dimension in blocks
            for block_k in range(NUM_BLOCK_K):
                k_start = block_k * BLOCK_K
                k_end = min(K, (block_k + 1) * BLOCK_K)

                # Extract current K block for all M rows in this block
                lhs_block = lhs[b, m_start:m_end, k_start:k_end]

                # Process each N block
                for block_n in range(NUM_BLOCK_N):
                    n_start = block_n * BLOCK_N
                    n_end = min(N, (block_n + 1) * BLOCK_N)

                    # Extract current N block for this K block
                    rhs_block = rhs[k_start:k_end, n_start:n_end]

                    # Calculate contribution and add to the higher precision accumulator
                    contribution = np.matmul(lhs_block, rhs_block)
                    outputs[:, n_start:n_end] += contribution

            # Store final results for this M block, casting back to the original data type
            output[b, m_start:m_end, :] = outputs.astype(lhs.dtype)

    return output
