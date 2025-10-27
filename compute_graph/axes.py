import math


class Axis:
    def __init__(self, size: int, tile_size: int, depepdency: str) -> None:
        self.size = size
        self.tile_size = tile_size
        self.num_tiles = math.ceil(size / tile_size)
        self.dependency = depepdency

    def __repr__(self) -> str:
        return f"({self.dependency[:3]}){self.num_tiles}x{self.tile_size}={self.size}"


def linear_counter_to_indices(counter: int, axes: list[Axis]) -> dict[str, dict[int, int]]:
    """
    Convert linear counter to parallel axis indices.
    """
    print(counter, axes)
    indices = {}
    total_blocks = math.prod([axis.num_tiles for axis in axes])
    stride = total_blocks

    for axis in axes:
        stride = stride // axis.num_tiles
        tile_idx = (counter // stride) % axis.num_tiles

        if axis.tensor_name not in indices:
            indices[axis.tensor_name] = {}
        indices[axis.tensor_name][axis.axis_index] = tile_idx

    return indices
