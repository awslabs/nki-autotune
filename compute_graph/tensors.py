from compute_graph.axes import Axis


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
    def __init__(self, name: str, hbm_coordinates: list[TensorCoordinate]) -> None:
        """
        SBUF tensor with shape and name.

        Args:
            name: Name of the SBUF tensor
            shape: Shape of the SBUF tensor
        """
        self.name = name
        self.hbm_coordinates = hbm_coordinates

    def __repr__(self) -> str:
        coords_repr = ", ".join([repr(coord) for coord in self.hbm_coordinates])
        return f"TensorBuffer({self.name}, [{coords_repr}])"


def compute_num_parallel_tiles(hbm_tensors: list[HBMTensor]) -> int:
    """
    Compute total number of parallel tiles across all HBM tensors.

    Args:
        hbm_tensors: List of HBM tensors

    Returns:
        Product of num_tiles for all parallel axes
    """
    num_parallel_tiles = 1
    for tensor in hbm_tensors:
        for axis in tensor.axes:
            if axis.dependency == "parallel":
                num_parallel_tiles *= axis.num_tiles
    return num_parallel_tiles
