import math
from typing import Tuple

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
import tabulate

from autotune.core.metrics import check_correctness
from autotune.core.tensor import SBUFTensor
from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSORS_DTYPE


class GEMMConfig:
    """
    Configuration and validation for GEMM (General Matrix Multiplication) operations.

    Manages matrix blocking and tiling parameters for efficient GEMM computation on
    specialized hardware. Validates that input matrix dimensions are compatible with
    the specified blocking strategy and hardware tile constraints.

    The class calculates block and tile arrangements using the formula:
    Dimension = NUM_BLOCKS x TILES_PER_BLOCK x TILE_SIZE

    Hardware tile sizes:
    - TILE_M: 128 (stationary dimension)
    - TILE_N: 512 (moving dimension)
    - TILE_K: 128 (contraction dimension)

    Args:
        transposed_lhs: If True, treats LHS matrix as transposed (KxM instead of MxK)
    """

    def __init__(self, transposed_lhs: bool) -> None:
        """
        Initialize GEMM config.

        Args:
            transposed_lhs: Whether the LHS matrix should be interpreted as transposed.
        """
        self.transposed_lhs = transposed_lhs
        # Single tile sizes (hardware constants)
        self.TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
        self.TILE_N = nl.tile_size.gemm_moving_fmax  # 512
        self.TILE_K = nl.tile_size.pmax  # 128

    def __call__(
        self,
        lhs_shape: Tuple[int, int],
        rhs_shape: Tuple[int, int],
        NUM_BLOCK_M: int,
        NUM_BLOCK_N: int,
        NUM_BLOCK_K: int,
    ) -> None:
        """
        Calculate sizes for each dimension (M, N, K).

        NUM_BLOCK_X * TILES_IN_BLOCK_X * TILE_X = X
        BLOCK_X = TILES_IN_BLOCK_X * TILE_X
        TILES_IN_X = NUM_BLOCK_X * TILES_IN_BLOCK_X

        For each dimension, calculates:
        - Block size (dimension divided by number of blocks)
        - Number of tiles in each block (block size divided by tile size)
        - Total number of tiles in the dimension (dimension size divided by tile size)
        """
        if self.transposed_lhs:
            self.K, self.M = lhs_shape
        else:
            self.M, self.K = lhs_shape
        K_, self.N = rhs_shape
        assert (
            self.K == K_
        ), f"Contraction dimension mismatch: LHS {lhs_shape} has K={self.K}, RHS {rhs_shape} has K={K_}"

        # Single tile sizes (hardware constants)
        self.TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
        self.TILE_N = nl.tile_size.gemm_moving_fmax  # 512
        self.TILE_K = nl.tile_size.pmax  # 128

        self.NUM_BLOCK_M = NUM_BLOCK_M
        self.NUM_BLOCK_N = NUM_BLOCK_N
        self.NUM_BLOCK_K = NUM_BLOCK_K

        self.TILES_IN_M = math.ceil(self.M / self.TILE_M)
        assert (
            self.TILES_IN_M >= self.NUM_BLOCK_M
        ), f"NUM_BLOCK_M={self.NUM_BLOCK_M} exceeds available M tiles={self.TILES_IN_M}"
        self.TILES_IN_BLOCK_M = math.ceil(self.TILES_IN_M / self.NUM_BLOCK_M)
        self.BLOCK_M = int(self.TILES_IN_BLOCK_M * self.TILE_M)

        self.TILES_IN_N = math.ceil(self.N / self.TILE_N)
        assert (
            self.TILES_IN_N >= self.NUM_BLOCK_N
        ), f"NUM_BLOCK_N={self.NUM_BLOCK_N} exceeds available N tiles={self.TILES_IN_N}"
        self.TILES_IN_BLOCK_N = math.ceil(self.TILES_IN_N / self.NUM_BLOCK_N)
        self.BLOCK_N = int(self.TILES_IN_BLOCK_N * self.TILE_N)

        self.TILES_IN_K = math.ceil(self.K / self.TILE_K)
        assert (
            self.TILES_IN_K >= self.NUM_BLOCK_K
        ), f"NUM_BLOCK_K={self.NUM_BLOCK_K} exceeds available K tiles={self.TILES_IN_K}"
        self.TILES_IN_BLOCK_K = math.ceil(self.TILES_IN_K / self.NUM_BLOCK_K)
        self.BLOCK_K = int(self.TILES_IN_BLOCK_K * self.TILE_K)

    def __repr__(self) -> str:
        """
        Return a comprehensive string representation of the GEMM configuration.

        Returns:
            Formatted string showing configuration parameters and computed values.
        """
        class_name = self.__class__.__name__
        header = f"{class_name}(transposed_lhs={self.transposed_lhs})"

        # Check if dimensions have been configured (after __call__ has been invoked)
        if not hasattr(self, "M"):
            return f"{header}\n  Status: Not configured - call with input matrices first"

        # Create comprehensive table data with better organization
        table_data = [
            ["Matrix dimensions", self.M, self.N, self.K],
            ["Hardware tile size", self.TILE_M, self.TILE_N, self.TILE_K],
            ["Total tiles", self.TILES_IN_M, self.TILES_IN_N, self.TILES_IN_K],
            ["Block count", self.NUM_BLOCK_M, self.NUM_BLOCK_N, self.NUM_BLOCK_K],
            ["Tiles per block", self.TILES_IN_BLOCK_M, self.TILES_IN_BLOCK_N, self.TILES_IN_BLOCK_K],
            ["Block size", self.BLOCK_M, self.BLOCK_N, self.BLOCK_K],
        ]

        # Generate formatted table
        table = tabulate.tabulate(
            table_data,
            headers=["Parameter", "M (rows)", "N (cols)", "K (inner)"],
            tablefmt="simple_outline",
            numalign="right",
        )
        return f"{header}\n{table}"


def calculate_tile_overlap(coords1: dict, coords2: dict) -> tuple:
    """
    Calculate the overlapping tile region between two tile coordinate ranges.

    Args:
        coords1: First tile coordinates dictionary with 'start_tile_index' and 'num_tiles'
        coords2: Second tile coordinates dictionary with 'start_tile_index' and 'num_tiles'

    Returns:
        Tuple of (overlap_start, overlap_end, num_overlap_tiles)
    """
    start1 = coords1["start_tile_index"]
    end1 = start1 + coords1["num_tiles"]
    start2 = coords2["start_tile_index"]
    end2 = start2 + coords2["num_tiles"]

    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    num_overlap_tiles = max(0, overlap_end - overlap_start)

    return overlap_start, overlap_end, num_overlap_tiles


def calculate_tile_overlap_ranges(lhs_tiles: SBUFTensor, rhs_tiles: SBUFTensor, result_tiles: SBUFTensor) -> dict:
    """
    Calculate the overlapping tile ranges for matrix multiplication.

    Args:
        lhs_tiles: Left-hand side SBUF tensor
        rhs_tiles: Right-hand side SBUF tensor
        result_tiles: Result SBUF tensor

    Returns:
        Dictionary containing:
        - num_tiles: (num_M_tiles, num_N_tiles, num_K_tiles)
        - global_ranges: Dictionary with global tile ranges for each dimension
            - M_range: (M_start, M_end)
            - N_range: (N_start, N_end)
            - K_range: (K_start, K_end)
        - result_offsets: (M_offset, N_offset) for result tensor local indexing
    """
    # Get tile coordinates from each tensor
    lhs_M_coords = lhs_tiles.tile_coordinates["M"]
    lhs_K_coords = lhs_tiles.tile_coordinates["K"]
    rhs_K_coords = rhs_tiles.tile_coordinates["K"]
    rhs_N_coords = rhs_tiles.tile_coordinates["N"]
    result_M_coords = result_tiles.tile_coordinates["M"]
    result_N_coords = result_tiles.tile_coordinates["N"]

    # Calculate overlapping regions for each dimension (in global coordinates)
    K_start, K_end, num_K_tiles = calculate_tile_overlap(lhs_K_coords, rhs_K_coords)
    M_start, M_end, num_M_tiles = calculate_tile_overlap(lhs_M_coords, result_M_coords)
    N_start, N_end, num_N_tiles = calculate_tile_overlap(rhs_N_coords, result_N_coords)

    # Calculate local offsets for result tensor (still needed for direct tensor access)
    result_M_offset = M_start - result_M_coords["start_tile_index"]
    result_N_offset = N_start - result_N_coords["start_tile_index"]

    return {
        "num_tiles": (num_M_tiles, num_N_tiles, num_K_tiles),
        "global_ranges": {"M_range": (M_start, M_end), "N_range": (N_start, N_end), "K_range": (K_start, K_end)},
        "result_offsets": (result_M_offset, result_N_offset),
    }


def matmul_tiles(lhs_tiles: SBUFTensor, rhs_tiles: SBUFTensor, result_tiles: SBUFTensor, tile_transposed_lhs: bool):
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

    # Calculate overlapping regions using the helper function
    overlap_info = calculate_tile_overlap_ranges(lhs_tiles, rhs_tiles, result_tiles)
    for key in overlap_info:
        print(key, overlap_info[key])
    num_M_tiles, num_N_tiles, num_K_tiles = overlap_info["num_tiles"]
    M_start, M_end = overlap_info["global_ranges"]["M_range"]
    N_start, N_end = overlap_info["global_ranges"]["N_range"]
    K_start, K_end = overlap_info["global_ranges"]["K_range"]
    result_M_offset, result_N_offset = overlap_info["result_offsets"]

    # Iterate over tiles using nl.affine_range for hardware optimization
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for tile_idx_M in nl.affine_range(num_M_tiles):
        global_M_tile = M_start + tile_idx_M
        for tile_idx_N in nl.affine_range(num_N_tiles):
            global_N_tile = N_start + tile_idx_N
            """
            Use PSUM buffer to accumulate into a single hardware tile
            """
            result_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for tile_idx_K in nl.affine_range(num_K_tiles):
                global_K_tile = K_start + tile_idx_K
                # Read tiles using global indices (the read_tile method now handles conversion)
                lhs_tile = lhs_tiles.read_tile(tile_indices={"M": global_M_tile, "K": global_K_tile})
                rhs_tile = rhs_tiles.read_tile(tile_indices={"K": global_K_tile, "N": global_N_tile})
                result_tile += nisa.nc_matmul(lhs_tile, rhs_tile)
            # Store result using local indices for direct tensor access
            result_tiles.tensor[
                idx_res.p, result_M_offset + tile_idx_M, result_N_offset + tile_idx_N, idx_res.x
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

        # Use the centralized check_correctness function from metrics module
        check_correctness(golden, nki_out_tensor, atol, rtol)


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
