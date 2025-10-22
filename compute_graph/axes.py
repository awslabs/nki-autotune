import math
from typing import Dict, List

from compute_graph.primitives import AXIS, INPUT_TENSOR_SHAPE


class Axis:
    def __init__(self, tensor_name: str, axis_index: int, tile_size: int, size: int, num_tiles: int) -> None:
        self.tensor_name = tensor_name
        self.axis_index = axis_index
        self.tile_size = tile_size
        self.size = size
        self.num_tiles = num_tiles

    def __repr__(self) -> str:
        return (
            f"Axis({self.tensor_name}[{self.axis_index}], "
            f"size={self.size}, "
            f"tile_size={self.tile_size}, "
            f"num_tiles={self.num_tiles})"
        )


def make_axes(input_tensors: Dict[str, INPUT_TENSOR_SHAPE], axes: List[AXIS]) -> List[Axis]:
    processed_axes = []
    for tensor_name, axis_idx, tile_size in axes:
        size = input_tensors[tensor_name][axis_idx]
        num_tiles = math.ceil(size / tile_size)
        axis = Axis(tensor_name, axis_idx, tile_size, size, num_tiles)
        processed_axes.append(axis)
    return processed_axes


def linear_counter_to_indices(counter: int, axes: List[Axis]) -> Dict[str, Dict[int, int]]:
    """
    Convert linear counter to parallel axis indices.
    """
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
