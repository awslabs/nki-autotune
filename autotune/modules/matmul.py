import math
from typing import Dict, List

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
import tabulate

from autotune.core.metrics import check_correctness
from autotune.core.tensor import SBUFTensor
from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSORS_DTYPE


def _generate_valid_configs_for_dimension(dimension_size: int, tile_size: int) -> List[Dict[str, int]]:
    """
    Generate valid block configurations for tiling a dimension.

    Each configuration divides the dimension into blocks, where each block contains
    multiple tiles of size tile_size. All blocks except the last must be full,
    and the last block must have remaining data that fits within one block_size.

    Example: dimension_size=1279, tile_size=128
    - Valid: 3 blocks of 4 tiles each (block_size=512)
      → First 2 blocks: 1024, Last block covers remaining: 255
    - Invalid: 4 blocks of 4 tiles each (block_size=512)
      → First 3 blocks already cover 1536 > 1279

    Args:
        dimension_size: Size of the dimension to tile
        tile_size: Size of each hardware tile

    Returns:
        List of dicts with keys: num_blocks, tiles_per_block, block_size, total_tiles
    """
    valid_configs = []

    # Calculate the total tiles needed for this dimension
    max_tiles_needed = math.ceil(dimension_size / tile_size)

    # Try different numbers of blocks from 1 to max_tiles_needed
    for num_blocks in range(1, max_tiles_needed + 1):
        # Try different tiles_per_block values
        # Start from 1 and go up to a reasonable maximum
        max_tiles_per_block = max_tiles_needed

        for tiles_per_block in range(1, max_tiles_per_block + 1):
            block_size = tiles_per_block * tile_size

            # Calculate total coverage with this configuration
            # All blocks except the last are full
            full_blocks_coverage = (num_blocks - 1) * block_size

            # Check if we already exceed dimension_size with just full blocks
            if full_blocks_coverage >= dimension_size:
                # This means we don't need the last block, so it's invalid
                continue

            # Calculate what the last block needs to cover
            remaining_size = dimension_size - full_blocks_coverage

            # The last block must not be empty and should not exceed block_size
            if remaining_size <= 0 or remaining_size > block_size:
                continue

            # This is a valid configuration
            valid_configs.append(
                {
                    "num_blocks": num_blocks,
                    "tiles_per_block": tiles_per_block,
                    "block_size": block_size,
                    "total_tiles": num_blocks * tiles_per_block,
                }
            )
    return valid_configs


class GEMMConfig:
    def __init__(
        self,
        dimension_sizes: Dict[str, int],
        tile_sizes: Dict[str, int],
        m_config: Dict[str, int],
        n_config: Dict[str, int],
        k_config: Dict[str, int],
    ) -> None:
        # Set dimension sizes
        self.M = dimension_sizes["M"]
        self.N = dimension_sizes["N"]
        self.K = dimension_sizes["K"]

        # Set tile sizes
        self.TILE_M = tile_sizes["M"]
        self.TILE_N = tile_sizes["N"]
        self.TILE_K = tile_sizes["K"]

        # Set block configuration from configs
        self.NUM_BLOCK_M = m_config["num_blocks"]
        self.NUM_BLOCK_N = n_config["num_blocks"]
        self.NUM_BLOCK_K = k_config["num_blocks"]

        self.TILES_PER_BLOCK_M = m_config["tiles_per_block"]
        self.TILES_PER_BLOCK_N = n_config["tiles_per_block"]
        self.TILES_PER_BLOCK_K = k_config["tiles_per_block"]

        self.BLOCK_M = m_config["block_size"]
        self.BLOCK_N = n_config["block_size"]
        self.BLOCK_K = k_config["block_size"]

        # Set total tiles from configs
        self.TILES_IN_M = m_config["total_tiles"]
        self.TILES_IN_N = n_config["total_tiles"]
        self.TILES_IN_K = k_config["total_tiles"]

        self.PADDING_OVERHEAD_M = self.TILES_IN_M * self.TILE_M / self.M
        self.PADDING_OVERHEAD_N = self.TILES_IN_N * self.TILE_N / self.N
        self.PADDING_OVERHEAD_K = self.TILES_IN_K * self.TILE_K / self.K

    def __repr__(self) -> str:
        """
        Return a comprehensive string representation of the GEMM configuration.

        Returns:
            Formatted string showing configuration parameters and computed values.
        """
        class_name = self.__class__.__name__
        header = f"{class_name}"

        # Check if dimensions have been configured (after __call__ has been invoked)
        if not hasattr(self, "M"):
            return f"{header}\n  Status: Not configured - call with input matrices first"

        # Create comprehensive table data with better organization
        table_data = [
            ["Matrix dimensions", self.M, self.N, self.K],
            ["Hardware tile size", self.TILE_M, self.TILE_N, self.TILE_K],
            ["Total tiles", self.TILES_IN_M, self.TILES_IN_N, self.TILES_IN_K],
            ["Block count", self.NUM_BLOCK_M, self.NUM_BLOCK_N, self.NUM_BLOCK_K],
            ["Tiles per block", self.TILES_PER_BLOCK_M, self.TILES_PER_BLOCK_N, self.TILES_PER_BLOCK_K],
            ["Block size", self.BLOCK_M, self.BLOCK_N, self.BLOCK_K],
            ["Padding Overhead", self.PADDING_OVERHEAD_M, self.PADDING_OVERHEAD_N, self.PADDING_OVERHEAD_K],
        ]

        # Generate formatted table
        table = tabulate.tabulate(
            table_data,
            headers=["Parameter", "M (rows)", "N (cols)", "K (inner)"],
            tablefmt="simple_outline",
            numalign="right",
        )
        return f"{header}\n{table}"


class GEMMConfigGen:
    """
    Configuration and validation for GEMM (General Matrix Multiplication) operations.

    Manages matrix blocking and tiling parameters for efficient GEMM computation on
    specialized hardware. Validates that input matrix dimensions are compatible with
    the specified blocking strategy and hardware tile constraints.

    The class calculates block and tile arrangements using the formula:
    Dimension = NUM_BLOCKS x TILES_IN_BLOCK x TILE_SIZE
    """

    def __init__(self, transposed_lhs: bool, lhs_shape: tuple[int, int], rhs_shape: tuple[int, int]) -> None:
        """
        Initialize GEMM config.
        """
        self.transposed_lhs = transposed_lhs
        self.lhs_shape = lhs_shape
        self.rhs_shape = rhs_shape

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

    def generate_configs(self) -> List[GEMMConfig]:
        """
        Generate all possible valid GEMM configurations for the current matrix dimensions.

        Returns:
            List of dictionaries, each containing all values needed for table display:
            - Matrix dimensions: M, N, K
            - Hardware tile sizes: TILE_M, TILE_N, TILE_K
            - Total tiles: TILES_IN_M, TILES_IN_N, TILES_IN_K
            - Block counts: NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K
            - Tiles per block: TILES_PER_BLOCK_M, TILES_PER_BLOCK_N, TILES_PER_BLOCK_K
            - Block sizes: BLOCK_M, BLOCK_N, BLOCK_K
        """
        # Generate valid configurations for each dimension
        dimension_sizes = {"M": self.M, "N": self.N, "K": self.K}
        tile_sizes = {"M": self.TILE_M, "N": self.TILE_N, "K": self.TILE_K}
        m_configs = _generate_valid_configs_for_dimension(self.M, self.TILE_M)
        n_configs = _generate_valid_configs_for_dimension(self.N, self.TILE_N)
        k_configs = _generate_valid_configs_for_dimension(self.K, self.TILE_K)

        # Generate cartesian product of all combinations
        all_configs = []
        for m_config in m_configs:
            for n_config in n_configs:
                for k_config in k_configs:
                    config = GEMMConfig(dimension_sizes, tile_sizes, m_config, n_config, k_config)
                    all_configs.append(config)

        return all_configs


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


def max_nki_index(index1, index2):
    if index1 == 0:
        max_index = index2
    elif index2 == 0:
        max_index = index1
    else:
        max_index = max(index1, index2)
    return max_index


def calculate_tile_overlap(coords1: dict, coords2: dict) -> tuple[int, int]:
    """
    Calculate the overlapping tile region between two tile coordinate ranges.

    Args:
        coords1: First tile coordinates dictionary with 'start_tile_index' and 'num_tiles'
        coords2: Second tile coordinates dictionary with 'start_tile_index' and 'num_tiles'

    Returns:
        Tuple of (overlap_start, num_overlap_tiles)
    """
    start_1 = coords1["start_tile_index"]
    start_2 = coords2["start_tile_index"]
    overlap_start = max_nki_index(start_1, start_2)
    # print(f"start_1 {start_1} start_2 {start_2} --> overlap_start = {overlap_start}.")

    num_tiles_1 = coords1["num_tiles"]
    num_tiles_2 = coords2["num_tiles"]
    num_overlap_tiles = min(num_tiles_1, num_tiles_2)
    # print(f"num_tiles_1 {num_tiles_1} num_tiles_2 {num_tiles_2} --> num_overlap_tiles = {num_overlap_tiles}.")

    return overlap_start, num_overlap_tiles


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
        - global_starts: Dictionary with global tile ranges for each dimension
            - M: M_start
            - N: N_start
            - K: K_start
        - result_offsets: (M_offset, N_offset) for result tensor local indexing
    """
    # Calculate overlapping regions for each dimension (in global coordinates)
    K_start, num_K_tiles = calculate_tile_overlap(lhs_tiles.tile_coordinates["K"], rhs_tiles.tile_coordinates["K"])
    M_start, num_M_tiles = calculate_tile_overlap(lhs_tiles.tile_coordinates["M"], result_tiles.tile_coordinates["M"])
    N_start, num_N_tiles = calculate_tile_overlap(rhs_tiles.tile_coordinates["N"], result_tiles.tile_coordinates["N"])

    # Calculate local offsets for result tensor (still needed for direct tensor access)
    result_M_offset = M_start - result_tiles.tile_coordinates["M"]["start_tile_index"]
    result_N_offset = N_start - result_tiles.tile_coordinates["N"]["start_tile_index"]

    return {
        "num_tiles": (num_M_tiles, num_N_tiles, num_K_tiles),
        "global_starts": {"M": M_start, "N": N_start, "K": K_start},
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
    # for key in overlap_info:
    #     print(key, overlap_info[key])
    num_M_tiles, num_N_tiles, num_K_tiles = overlap_info["num_tiles"]
    M_start = overlap_info["global_starts"]["M"]
    N_start = overlap_info["global_starts"]["N"]
    K_start = overlap_info["global_starts"]["K"]
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
            # FIXME: if K=1, just copy not add
            result_tiles.tensor[
                idx_res.p, result_M_offset + tile_idx_M, result_N_offset + tile_idx_N, idx_res.x
            ] += result_tile[idx_res.p, idx_res.x]


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
