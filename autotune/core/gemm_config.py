import math
from itertools import permutations, product
from typing import Dict, List, Tuple

import neuronxcc.nki.language as nl
import tabulate


def str_to_dict(loop_order: str) -> Dict:
    """
    Convert a loop order string into a bidirectional mapping dictionary.

    Args:
        loop_order (str): String representing loop order (e.g., "MNK", "KNM").

    Returns:
        Dict: Bidirectional mapping where indices map to characters and vice versa.

    Example:
        >>> str_to_dict("MNK")
        {0: 'M', 1: 'N', 2: 'K', 'M': 0, 'N': 1, 'K': 2}
    """
    assert sorted(loop_order) == sorted("MNK"), f"Invalid loop_order: {loop_order}. Must contain exactly M, N, and K."
    loop_order_dict = {}
    for position, axis in enumerate(loop_order):
        loop_order_dict[position] = axis
        loop_order_dict[axis] = position
    return loop_order_dict


def generate_configs(**kwargs) -> List[Dict]:
    """
    Generate all combinations of configuration options.

    Args:
        **kwargs: Component names and their option lists.
                  e.g., NUM_BLOCK_M=[1,2,4], template=["MN","MKN"]

    Returns:
        List[Dict]: All possible configuration dictionaries from the
                    Cartesian product of input options.
    """
    # Get all component names and their option lists
    component_names = list(kwargs.keys())
    option_lists = list(kwargs.values())

    # Generate all possible combinations
    all_combinations = list(product(*option_lists))

    # Create a configuration dictionary for each combination
    configs = []
    for combo in all_combinations:
        # Start with base config if provided, otherwise empty dict
        config = {}

        # Add the specific component values to this config
        for i, name in enumerate(component_names):
            config[name] = combo[i]

        configs.append(config)

    return configs


def _generate_blocks_for_axis(axis: str, dimension_size: int, tile_size: int) -> List[Dict[str, int]]:
    """
    Generate valid block configurations for tiling a dimension.

    Each configuration divides the dimension into blocks, where each block contains
    multiple tiles. All blocks except the last must be full, and the last block
    must have remaining data that fits within one block_size.

    Example: dimension_size=1279, tile_size=128
    - Valid: 3 blocks of 4 tiles each (block_size=512)
      → First 2 blocks: 1024, Last block covers remaining: 255

    Args:
        axis: Axis identifier (unused, kept for compatibility)
        dimension_size: Size of the dimension to tile
        tile_size: Size of each hardware tile

    Returns:
        List[Dict[str, int]]: Configurations with keys: size, tile_size, num_blocks,
                              tiles_per_block, block_size, total_tiles
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
                    "size": dimension_size,
                    "tile_size": tile_size,
                    "num_blocks": num_blocks,
                    "tiles_per_block": tiles_per_block,
                    "block_size": block_size,
                    "total_tiles": num_blocks * tiles_per_block,
                }
            )
    return valid_configs


class GEMMConfig:
    """Configuration for GEMM (General Matrix Multiplication) operations.

    Manages matrix blocking, tiling, and loop ordering parameters for
    efficient computation on specialized hardware.
    """

    def __init__(
        self,
        m_config: Dict[str, int],
        n_config: Dict[str, int],
        k_config: Dict[str, int],
        lhs_position: int,
        rhs_position: int,
        loop_order: str,
    ) -> None:
        """
        Initialize GEMM configuration with dimension and operation parameters.

        Args:
            m_config: Configuration for M dimension (matrix rows)
            n_config: Configuration for N dimension (matrix columns)
            k_config: Configuration for K dimension (contraction)
            lhs_position: Relative position for left-hand side operations (0, 1, or 2)
            rhs_position: Relative position for right-hand side operations (0, 1, or 2)
            loop_order: String specifying loop nesting order (e.g., "MNK", "KNM")
        """
        # Set dimension sizes
        self.M = m_config["size"]
        self.N = n_config["size"]
        self.K = k_config["size"]

        # Set tile sizes
        self.TILE_M = m_config["tile_size"]
        self.TILE_N = n_config["tile_size"]
        self.TILE_K = k_config["tile_size"]

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

        self.loop_order_str = loop_order
        self.loop_order = str_to_dict(loop_order)

        self.op_positions: Dict[str, int] = {}
        self.op_positions["lhs"] = self._parse_absolute_position(lhs_position, ("M", "K"))
        self.op_positions["rhs"] = self._parse_absolute_position(rhs_position, ("K", "N"))
        self.op_positions["result"] = self.loop_order["K"]
        self.op_positions["save"] = self.loop_order["K"]

    def _parse_absolute_position(self, relative_position: int, axes: Tuple[str, ...]) -> int:
        """
        Convert relative position to absolute position in loop order.

        Relative positions (0, 1, 2) map to gaps between/around axes:
        |0| Axis0 |1| Axis1 |2|

        Args:
            relative_position: Position relative to axes (0, 1, or 2)
            axes: Tuple of axis names to position relative to

        Returns:
            int: Absolute position in the loop order

        Example:
            If axes=('M','K') with positions [1,3] and relative_position=1:
            Returns 2 (between the two axes)
        """
        axis_positions = sorted([self.loop_order[axis] for axis in axes])
        if relative_position == 0:
            absolute_position = 0
        elif relative_position == 1:
            absolute_position = axis_positions[0] + 1
        elif relative_position == 2:
            absolute_position = axis_positions[1] + 1
        else:
            raise Exception(f"relative_position {relative_position} is out of bound for axes {axes}.")
        return absolute_position

    def __repr__(self) -> str:
        """
        Return a formatted string representation of the GEMM configuration.

        Returns:
            str: Tabulated view of all configuration parameters and values.
        """
        class_name = self.__class__.__name__
        header = f"{class_name}(lhs_position {self.op_positions['lhs']} rhs_position {self.op_positions['rhs']} loop_order {self.loop_order_str})"

        # Check if dimensions have been configured (after __call__ has been invoked)
        if not hasattr(self, "M"):
            return f"{header}\n  Status: Not configured - call with input matrices first"

        # Create complete table data including merged rows
        table_data = [
            ["Matrix dimensions", self.M, self.N, self.K],
            ["Hardware tile size", self.TILE_M, self.TILE_N, self.TILE_K],
            ["Total tiles", self.TILES_IN_M, self.TILES_IN_N, self.TILES_IN_K],
            ["Block count", self.NUM_BLOCK_M, self.NUM_BLOCK_N, self.NUM_BLOCK_K],
            ["Tiles per block", self.TILES_PER_BLOCK_M, self.TILES_PER_BLOCK_N, self.TILES_PER_BLOCK_K],
            ["Block size", self.BLOCK_M, self.BLOCK_N, self.BLOCK_K],
            ["Padding Overhead", self.PADDING_OVERHEAD_M, self.PADDING_OVERHEAD_N, self.PADDING_OVERHEAD_K],
        ]

        # Generate the complete table with merged rows
        table = tabulate.tabulate(
            table_data,
            headers=["Parameter", "M (rows)", "N (cols)", "K (contraction)"],
            tablefmt="simple_outline",
            numalign="right",
        )

        return f"{header}\n{table}"


def generate_gemm_configs(M: int, N: int, K: int) -> List[Dict]:
    """
    Generate all valid GEMM configurations for given matrix dimensions.

    Creates configurations for matrix multiplication C = A @ B where:
    - A has shape (M, K)
    - B has shape (K, N)
    - C has shape (M, N)

    Uses hardware-specific tile sizes and generates all valid combinations
    of blocking strategies, loop orders, and operation positions.

    Args:
        M: Number of rows in output matrix
        N: Number of columns in output matrix
        K: Contraction dimension size

    Returns:
        List[Dict]: Configuration dictionaries containing:
            - m_config, n_config, k_config: Blocking parameters for each dimension
            - loop_order: Nesting order for loops (permutation of "MNK")
            - lhs_position, rhs_position: Operation placement in loop structure
    """
    # Single tile sizes (hardware constants)
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512
    TILE_K = nl.tile_size.pmax  # 128

    m_configs = _generate_blocks_for_axis("M", M, TILE_M)
    n_configs = _generate_blocks_for_axis("N", N, TILE_N)
    k_configs = _generate_blocks_for_axis("K", K, TILE_K)
    loop_orders = ["".join(loop_order) for loop_order in permutations("MNK")]
    lhs_positions = [0, 1, 2]
    rhs_positions = [0, 1, 2]

    configs = generate_configs(
        m_config=m_configs,
        n_config=n_configs,
        k_config=k_configs,
        loop_order=loop_orders,
        lhs_position=lhs_positions,
        rhs_position=rhs_positions,
    )

    return configs
