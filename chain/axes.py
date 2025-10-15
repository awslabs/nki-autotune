import math
import random
from itertools import product
from typing import Dict, List, Tuple


class Axis:
    def __init__(
        self,
        tensor_axes: List[Tuple[str, int]],
        size: int,
        tile_size: int,
        num_blocks: int,
        tiles_per_block: int,
        block_size: int,
        total_tiles: int,
    ) -> None:
        self.tensor_axes = tensor_axes
        self.size = size
        self.tile_size = tile_size
        self.num_blocks = num_blocks
        self.tiles_per_block = tiles_per_block
        self.block_size = block_size
        self.total_tiles = total_tiles

    def __repr__(self) -> str:
        tensor_axes_str = ", ".join([f"{name}[{idx}]" for name, idx in self.tensor_axes])
        return (
            f"Axis({tensor_axes_str}, "
            f"size={self.size}, "
            f"num_blocks={self.num_blocks}, "
            f"tiles_per_block={self.tiles_per_block}, "
            f"tile_size={self.tile_size}, "
            f"block_size={self.block_size}, "
            f"total_tiles={self.total_tiles})"
        )


def sample_axes_configs(
    input_tensor_shapes: Dict[str, Tuple[int, ...]], parallel_axes: List[Tuple[str, int, int]]
) -> Tuple[Axis, ...]:
    """
    Sample a valid tiling configuration for parallel axes.

    For each parallel axis, generates all divisor-based block configurations that evenly
    divide the total tiles. Returns a random permutation of configurations across axes.

    Args:
        input_tensor_shapes: Mapping from tensor names to their shapes
        parallel_axes: List of (tensor_name, axis_index, tile_size) tuples

    Returns:
        Tuple[Axis, ...]: Randomly selected configuration with one Axis per parallel axis
    """

    all_configs: List[List[Axis]] = []
    for parallel_axis in parallel_axes:
        tensor_name, axis_index, tile_size = parallel_axis
        tensor_shape = input_tensor_shapes[tensor_name]
        size = tensor_shape[axis_index]

        total_tiles = math.ceil(size / tile_size)
        axis_configs: List[Axis] = []
        for num_blocks in range(1, total_tiles + 1):
            if total_tiles % num_blocks == 0:
                tiles_per_block = total_tiles // num_blocks
                block_size = tiles_per_block * tile_size

                config = Axis(
                    tensor_axes=[(tensor_name, axis_index)],
                    size=size,
                    tile_size=tile_size,
                    num_blocks=num_blocks,
                    tiles_per_block=tiles_per_block,
                    block_size=block_size,
                    total_tiles=total_tiles,
                )
                axis_configs.append(config)
        all_configs.append(axis_configs)

    config_permutations = list(product(*all_configs))
    axes_config = random.choice(config_permutations)

    return axes_config
