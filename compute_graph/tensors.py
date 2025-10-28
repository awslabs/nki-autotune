import math


class Axis:
    def __init__(self, size: int, tile_size: int, dependency: str) -> None:
        self.size = size
        self.tile_size = tile_size
        self.num_tiles = math.ceil(size / tile_size)
        self.dependency = dependency
        assert dependency in ["parallel", "sequential"]

    def __repr__(self) -> str:
        return f"({self.dependency[:3]}){self.num_tiles}x{self.tile_size}={self.size}"


class TensorCoordinate:
    def __init__(self, start_tile_index: int, num_tiles: int, tile_size: int) -> None:
        """
        Coordinate information for one axis of a tiled tensor.

        Args:
            start_tile_index: Starting tile index in the source tensor
            num_tiles: Number of tiles along this axis
            tile_size: Size of each tile along this axis
        """
        self.start_tile_index = start_tile_index
        self.num_tiles = num_tiles
        self.tile_size = tile_size
        self.size = self.tile_size * self.num_tiles

    def __repr__(self) -> str:
        end_idx = self.start_tile_index + self.num_tiles
        return f"[{self.start_tile_index}:{end_idx}]*{self.tile_size}={self.size}"


class HBMTensor:
    def __init__(self, name: str, axes: list[Axis]) -> None:
        """
        HBM tensor with shape and name.

        Args:
            name: Name of the HBM tensor
            axes: size, tile size of each axis
        """
        self.name = name
        self.axes = axes

    def __repr__(self) -> str:
        axes_repr = ", ".join([repr(axis) for axis in self.axes])
        return f"HBMTensor({self.name}, [{axes_repr}])"


class TensorBuffer:
    def __init__(self, name: str, shape: tuple[int, ...]) -> None:
        """
        SBUF tensor with shape and name.

        Args:
            name: Name of the HBM tensor
            axes: size, tile size of each axis
        """
        self.name = name
        self.shape = shape

    def __repr__(self) -> str:
        return f"TensorBuffer({self.name}, {self.shape})"


def compute_num_parallel_tiles(
    hbm_tensors: dict[str, tuple[int, ...]], parallel_axes: list[tuple[str, int, int]]
) -> int:
    """
    Compute total number of parallel tiles across all HBM tensors.

    Args:
        hbm_tensors: List of HBM tensors

    Returns:
        Product of num_tiles for all parallel axes
    """
    num_parallel_tiles = 1
    for parallel_axis in parallel_axes:
        tensor_name, axis_idx, tile_size = parallel_axis
        tensor_shape = hbm_tensors[tensor_name]
        size = tensor_shape[axis_idx]
        num_tiles = math.ceil(size / tile_size)
        num_parallel_tiles *= num_tiles
    return num_parallel_tiles
