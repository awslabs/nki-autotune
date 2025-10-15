import math
from itertools import product
from typing import Dict, List, Tuple, TypedDict

import numpy as np


class AxesConfig(TypedDict):
    parallel_axes_config: List["ParallelAxis"]
    sequential_axis_config: "SequentialAxis"


class Axis:
    def __init__(
        self, size: int, tile_size: int, num_blocks: int, tiles_per_block: int, block_size: int, total_tiles: int
    ) -> None:
        self.size = size
        self.tile_size = tile_size
        self.num_blocks = num_blocks
        self.tiles_per_block = tiles_per_block
        self.block_size = block_size
        self.total_tiles = total_tiles

    def __repr__(self) -> str:
        return (
            f"Axis(size={self.size}, "
            f"num_blocks={self.num_blocks}, "
            f"tiles_per_block={self.tiles_per_block}, "
            f"tile_size={self.tile_size}, "
            f"block_size={self.block_size}, "
            f"total_tiles={self.total_tiles})"
        )


class ParallelAxis(Axis):
    def __init__(
        self,
        tensor_name: str,
        axis_index: int,
        size: int,
        tile_size: int,
        num_blocks: int,
        tiles_per_block: int,
        block_size: int,
        total_tiles: int,
    ) -> None:
        super().__init__(size, tile_size, num_blocks, tiles_per_block, block_size, total_tiles)
        self.tensor_name = tensor_name
        self.axis_index = axis_index

    def __repr__(self) -> str:
        return (
            f"ParallelAxis({self.tensor_name}[{self.axis_index}], "
            f"size={self.size}, "
            f"num_blocks={self.num_blocks}, "
            f"tiles_per_block={self.tiles_per_block}, "
            f"tile_size={self.tile_size}, "
            f"block_size={self.block_size}, "
            f"total_tiles={self.total_tiles})"
        )


class SequentialAxis(Axis):
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
        super().__init__(size, tile_size, num_blocks, tiles_per_block, block_size, total_tiles)
        self.tensor_axes = tensor_axes

    def __repr__(self) -> str:
        tensor_axes_str = ", ".join([f"{name}[{idx}]" for name, idx in self.tensor_axes])
        return (
            f"SequentialAxis({tensor_axes_str}, "
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
) -> List[List[ParallelAxis]]:
    """
    Return a list of parallel axes configs.
    Each config is a list of ParallelAxis, one per parallel axis.
    """
    parallel_axis_configs = []
    for tensor_name, axis_index, tile_size in parallel_axes:
        size = input_tensors[tensor_name].shape[axis_index]
        block_configs = generate_blocks_for_axis(size=size, tile_size=tile_size)
        axis_configs = []
        for config in block_configs:
            axis = ParallelAxis(
                tensor_name=tensor_name, axis_index=axis_index, size=size, tile_size=tile_size, **config
            )
            axis_configs.append(axis)
        parallel_axis_configs.append(axis_configs)
    return parallel_axis_configs


def generate_sequential_axes_configs(
    input_tensors: Dict[str, np.ndarray], sequential_axes: List[Tuple[str, int]], tile_size: int
) -> List[SequentialAxis]:
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
        axis = SequentialAxis(tensor_axes=sequential_axes, size=sequential_size, tile_size=tile_size, **config)
        axis_configs.append(axis)
    return axis_configs


def generate_axes_configs(
    input_tensors: Dict[str, np.ndarray],
    parallel_axes: List[Tuple[str, int, int]],
    sequential_axes: List[Tuple[str, int]],
    sequential_tile_size: int,
) -> List[AxesConfig]:
    """Generate all valid combinations of axes configurations for fusion.

    Args:
        input_tensors: Dictionary mapping tensor names to numpy arrays
        parallel_axes: List of (tensor_name, axis_index, tile_size) for parallel axes
        sequential_axes: List of (tensor_name, axis_index) for sequential axes
        sequential_tile_size: Tile size for the sequential axis

    Returns:
        List of configuration dictionaries with keys:
            - "parallel_axes_config": List of ParallelAxis objects for parallel axes
            - "sequential_axis_config": Single SequentialAxis object for sequential axis
    """
    parallel_axes_configs = generate_parallel_axes_configs(input_tensors=input_tensors, parallel_axes=parallel_axes)
    sequential_axis_configs = generate_sequential_axes_configs(
        input_tensors=input_tensors, sequential_axes=sequential_axes, tile_size=sequential_tile_size
    )

    axes_configs = []

    if parallel_axes_configs:
        parallel_combinations = list(product(*parallel_axes_configs))
    else:
        parallel_combinations = [[]]

    for parallel_combo in parallel_combinations:
        for sequential_config in sequential_axis_configs:
            axes_config = {"parallel_axes_config": list(parallel_combo), "sequential_axis_config": sequential_config}
            axes_configs.append(axes_config)

    return axes_configs
