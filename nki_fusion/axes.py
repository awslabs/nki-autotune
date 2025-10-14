import math
from itertools import product
from typing import Dict, List, Tuple

import numpy as np


class Axis:
    def __init__(
        self,
        name: str,
        axis_index: int,
        size: int,
        tile_size: int,
        num_blocks: int,
        tiles_per_block: int,
        block_size: int,
        total_tiles: int,
    ) -> None:
        self.name = name
        self.index = axis_index
        self.size = size
        self.tile_size = tile_size
        self.num_blocks = num_blocks
        self.tiles_per_block = tiles_per_block
        self.block_size = block_size
        self.total_tiles = total_tiles

    def __repr__(self) -> str:
        return (
            f"Axis({self.name}[{self.index}], "
            f"size={self.size}, "
            f"num_blocks={self.num_blocks}, "
            f"tiles_per_block={self.tiles_per_block}, "
            f"tile_size={self.tile_size}, "
            f"block_size={self.block_size}, "
            f"total_tiles={self.total_tiles})"
        )


def generate_blocks_for_axis(size: int, tile_size: int) -> List[Dict]:
    """
    Generate valid block configurations for tiling an axis.

    Divides the axis into evenly-sized blocks, each containing multiple tiles.
    Returns all divisor-based configurations that fully cover the axis size.

    Args:
        size: Size of the axis to tile
        tile_size: Size of each hardware tile

    Returns:
        List[Dict[str, int]]: Configurations with keys: size, tile_size, num_blocks,
                            tiles_per_block, block_size, total_tiles
    """
    # Calculate total number of tiles needed to cover the axis
    total_tiles = math.ceil(size / tile_size)

    # Find all divisors of total_tiles
    configurations = []
    for num_blocks in range(1, total_tiles + 1):
        if total_tiles % num_blocks == 0:
            tiles_per_block = total_tiles // num_blocks
            block_size = tiles_per_block * tile_size

            config = {
                "num_blocks": num_blocks,
                "tiles_per_block": tiles_per_block,
                "block_size": block_size,
                "total_tiles": total_tiles,
            }
            configurations.append(config)

    return configurations


def generate_parallel_axes_configs(
    input_tensors: Dict[str, np.ndarray], parallel_axes: List[Tuple[str, int, int]]
) -> List[List[Axis]]:
    """
    Return a list of parallel axes configs.
    Each config is a list of Axis, one per parallel axis.
    """
    all_axis_configs = []
    for tensor_name, axis_index, tile_size in parallel_axes:
        size = input_tensors[tensor_name].shape[axis_index]
        block_configs = generate_blocks_for_axis(size=size, tile_size=tile_size)
        axis_configs = []
        for config in block_configs:
            axis = Axis(name=tensor_name, axis_index=axis_index, size=size, tile_size=tile_size, **config)
            axis_configs.append(axis)
        all_axis_configs.append(axis_configs)
    all_combinations = [list(combo) for combo in product(*all_axis_configs)]
    return all_combinations


def generate_sequential_axes_configs(
    input_tensors: Dict[str, np.ndarray], sequential_axes: List[Tuple[str, int]], tile_size: int
) -> List[Axis]:
    sequential_size = None
    for tensor_name, axis_index in sequential_axes:
        size = input_tensors[tensor_name].shape[axis_index]
        if sequential_size:
            assert sequential_size == size, f"Different sequential sizes are not supported."
        else:
            sequential_size = size
    assert sequential_size
    block_configs = generate_blocks_for_axis(size=sequential_size, tile_size=tile_size)
    axis_configs = []
    for config in block_configs:
        axis = Axis(name=tensor_name, axis_index=axis_index, size=sequential_size, tile_size=tile_size, **config)
        axis_configs.append(axis)
    return axis_configs
