import math
from typing import List, Tuple


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


def generate_axis_configs(tensor_axes: List[Tuple[str, int]], size: int, tile_size: int) -> List[Axis]:
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
    configurations: List[Axis] = []
    for num_blocks in range(1, total_tiles + 1):
        if total_tiles % num_blocks == 0:
            tiles_per_block = total_tiles // num_blocks
            block_size = tiles_per_block * tile_size

            config = Axis(
                tensor_axes=tensor_axes,
                size=size,
                tile_size=tile_size,
                num_blocks=num_blocks,
                tiles_per_block=tiles_per_block,
                block_size=block_size,
                total_tiles=total_tiles,
            )
            configurations.append(config)

    return configurations
