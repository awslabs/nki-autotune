"""GEMM configuration generation for template-based autotuning."""

import math
import random
from itertools import permutations, product

import nki.language as nl

"""
Hardware tile size constants for Trainium2.
nkipy's nl.tile_size returns -1 (compile-time placeholder),
so we use neuronxcc values for config generation.
"""
_TILE_M = 128
_TILE_N = 512
_TILE_K = 128


def str_to_dict(loop_order: str) -> dict:
    """Convert a loop order string into a bidirectional mapping.

    Args:
        loop_order: String like "MNK", "KNM", etc.

    Returns:
        Bidirectional dict mapping indices to chars and chars to indices.
    """
    c0 = loop_order[0]
    c1 = loop_order[1]
    c2 = loop_order[2]
    return chars_to_dict(c0, c1, c2)


def chars_to_dict(c0: str, c1: str, c2: str) -> dict:
    """Build a bidirectional mapping from three axis characters.

    Uses string keys "0", "1", "2" for position-to-char mapping
    since NKI dicts require string keys.

    Args:
        c0: Character at position 0.
        c1: Character at position 1.
        c2: Character at position 2.

    Returns:
        Bidirectional dict mapping string positions to chars and chars to ints.
    """
    loop_order_dict: dict = {}
    loop_order_dict["0"] = c0
    loop_order_dict["1"] = c1
    loop_order_dict["2"] = c2
    loop_order_dict[c0] = 0
    loop_order_dict[c1] = 1
    loop_order_dict[c2] = 2
    return loop_order_dict


def generate_configs(**kwargs: list) -> list[dict]:
    """Generate all combinations of configuration options.

    Args:
        **kwargs: Component names mapped to their option lists.

    Returns:
        All configuration dicts from the Cartesian product.
    """
    component_names = list(kwargs.keys())
    option_lists = list(kwargs.values())
    all_combinations = list(product(*option_lists))
    configs = [dict(zip(component_names, combo)) for combo in all_combinations]
    return configs


def generate_blocks_for_axis(size: int, tile_size: int) -> list[dict[str, int]]:
    """Generate valid block configurations for tiling an axis.

    Args:
        size: Size of the axis to tile.
        tile_size: Size of each hardware tile.

    Returns:
        List of config dicts with size, tile_size, num_blocks,
        tiles_per_block, block_size, total_tiles.
    """
    total_tiles = math.ceil(size / tile_size)
    configurations = []
    for num_blocks in range(1, total_tiles + 1):
        if total_tiles % num_blocks == 0:
            tiles_per_block = total_tiles // num_blocks
            block_size = tiles_per_block * tile_size
            configurations.append(
                {
                    "size": size,
                    "tile_size": tile_size,
                    "num_blocks": num_blocks,
                    "tiles_per_block": tiles_per_block,
                    "block_size": block_size,
                    "total_tiles": total_tiles,
                }
            )
    return configurations


class GEMMConfig(nl.NKIObject):
    """Configuration for GEMM blocking, tiling, and loop ordering.

    Attributes:
        M: Matrix rows.
        N: Matrix columns.
        K: Contraction dimension.
        loop_order: Bidirectional mapping of loop nesting order.
        op_positions: Dict mapping operation names to loop positions.
    """

    def __init__(
        self,
        m_config: dict[str, int],
        n_config: dict[str, int],
        k_config: dict[str, int],
        lhs_position: int,
        rhs_position: int,
        loop_order_0: str,
        loop_order_1: str,
        loop_order_2: str,
    ) -> None:
        """Initialize GEMM configuration.

        Args:
            m_config: Configuration for M dimension.
            n_config: Configuration for N dimension.
            k_config: Configuration for K dimension.
            lhs_position: Relative position for LHS operations (0, 1, or 2).
            rhs_position: Relative position for RHS operations (0, 1, or 2).
            loop_order_0: First axis in loop nesting (e.g. "M").
            loop_order_1: Second axis in loop nesting (e.g. "N").
            loop_order_2: Third axis in loop nesting (e.g. "K").
        """
        self._init_dimensions(m_config, n_config, k_config)
        self._init_loop_order(loop_order_0, loop_order_1, loop_order_2, lhs_position, rhs_position)

    def _init_dimensions(self, m_config: dict[str, int], n_config: dict[str, int], k_config: dict[str, int]) -> None:
        """Set dimension, tile, and block parameters from configs.

        Args:
            m_config: M dimension config.
            n_config: N dimension config.
            k_config: K dimension config.
        """
        self.M = m_config["size"]
        self.N = n_config["size"]
        self.K = k_config["size"]
        self.TILE_M = m_config["tile_size"]
        self.TILE_N = n_config["tile_size"]
        self.TILE_K = k_config["tile_size"]
        self.NUM_BLOCK_M = m_config["num_blocks"]
        self.NUM_BLOCK_N = n_config["num_blocks"]
        self.NUM_BLOCK_K = k_config["num_blocks"]
        self.TILES_PER_BLOCK_M = m_config["tiles_per_block"]
        self.TILES_PER_BLOCK_N = n_config["tiles_per_block"]
        self.TILES_PER_BLOCK_K = k_config["tiles_per_block"]
        self.BLOCK_M = m_config["block_size"]
        self.BLOCK_N = n_config["block_size"]
        self.BLOCK_K = k_config["block_size"]
        self.TILES_IN_M = m_config["total_tiles"]
        self.TILES_IN_N = n_config["total_tiles"]
        self.TILES_IN_K = k_config["total_tiles"]
        self.PADDED_M = self.TILES_IN_M * self.TILE_M
        self.PADDED_N = self.TILES_IN_N * self.TILE_N
        self.PADDED_K = self.TILES_IN_K * self.TILE_K

    def _init_loop_order(self, lo0: str, lo1: str, lo2: str, lhs_position: int, rhs_position: int) -> None:
        """Set loop order and operation position mappings.

        Args:
            lo0: First axis character in loop nesting.
            lo1: Second axis character in loop nesting.
            lo2: Third axis character in loop nesting.
            lhs_position: Relative position for LHS operations.
            rhs_position: Relative position for RHS operations.
        """
        self.loop_order_str = lo0 + lo1 + lo2
        self.loop_order = chars_to_dict(lo0, lo1, lo2)
        self.op_positions: dict[str, int] = {}
        self.op_positions["lhs"] = self._parse_absolute_position(lhs_position, "M", "K")
        self.op_positions["rhs"] = self._parse_absolute_position(rhs_position, "K", "N")
        self.op_positions["result"] = self.loop_order["K"]
        self.op_positions["save"] = self.loop_order["K"]
        self.op_positions["x_op"] = self.loop_order["K"]

    def _parse_absolute_position(self, relative_position: int, axis_a: str, axis_b: str) -> int:
        """Convert relative position to absolute position in loop order.

        Relative positions (0, 1, 2) map to gaps between/around axes:
        |0| Axis0 |1| Axis1 |2|

        Args:
            relative_position: Position relative to axes (0, 1, or 2).
            axis_a: First axis name.
            axis_b: Second axis name.

        Returns:
            Absolute position in the loop order.

        Raises:
            Exception: If relative_position is out of bounds.
        """
        pos_a = self.loop_order[axis_a]
        pos_b = self.loop_order[axis_b]
        lo = min(pos_a, pos_b)
        hi = max(pos_a, pos_b)
        if relative_position == 0:
            absolute_position = 0
        elif relative_position == 1:
            absolute_position = lo + 1
        elif relative_position == 2:
            absolute_position = hi + 1
        else:
            assert False, "relative_position must be 0, 1, or 2"
        return absolute_position

    def __repr__(self) -> str:
        """Return formatted string representation."""
        return f"GEMMConfig(M={self.M} N={self.N} K={self.K} loop={self.loop_order_str})"


def _expand_loop_order_tuples(configs: list[dict]) -> list[dict]:
    """Expand loop_order_tuple into loop_order_0/1/2 keys.

    Args:
        configs: List of config dicts with loop_order_tuple key.

    Returns:
        Updated configs with loop_order_0, loop_order_1, loop_order_2.
    """
    result = []
    for cfg in configs:
        lo_tuple = cfg["loop_order_tuple"]
        new_cfg = {k: v for k, v in cfg.items() if k != "loop_order_tuple"}
        new_cfg["loop_order_0"] = lo_tuple[0]
        new_cfg["loop_order_1"] = lo_tuple[1]
        new_cfg["loop_order_2"] = lo_tuple[2]
        result.append(new_cfg)
    return result


def sample_gemm_configs(M: int, N: int, K: int, max_configs: int) -> list[dict]:
    """Generate valid GEMM configurations for given matrix dimensions.

    Creates configs for C = A @ B where A is (M, K), B is (K, N), C is (M, N).
    Uses hardware tile sizes and generates all valid combinations.

    Args:
        M: Number of rows in output matrix.
        N: Number of columns in output matrix.
        K: Contraction dimension size.
        max_configs: Max configs to return. Use -1 for all configs.

    Returns:
        List of configuration dictionaries.
    """
    TILE_M = _TILE_M
    TILE_N = _TILE_N
    TILE_K = _TILE_K

    m_configs = generate_blocks_for_axis(M, TILE_M)
    n_configs = generate_blocks_for_axis(N, TILE_N)
    k_configs = generate_blocks_for_axis(K, TILE_K)
    loop_order_tuples = list(permutations("MNK"))
    lhs_positions = [0, 1, 2]
    rhs_positions = [0, 1, 2]

    raw_configs = generate_configs(
        m_config=m_configs,
        n_config=n_configs,
        k_config=k_configs,
        loop_order_tuple=loop_order_tuples,
        lhs_position=lhs_positions,
        rhs_position=rhs_positions,
    )
    configs = _expand_loop_order_tuples(raw_configs)
    if max_configs > 0 and max_configs < len(configs):
        configs = random.sample(configs, max_configs)
    return configs
