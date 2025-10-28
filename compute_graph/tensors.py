import math


class Axis:
    """Represents a tensor axis with tiling and dependency information."""

    def __init__(self, size: int, tile_size: int, dependency: str) -> None:
        """
        Args:
            size: Total size of the axis
            tile_size: Size of each tile along this axis
            dependency: Either "parallel" or "sequential"
        """
        self.size = size
        self.tile_size = tile_size
        self.num_tiles = math.ceil(size / tile_size)
        self.dependency = dependency
        assert dependency in ["parallel", "sequential"]

    def __repr__(self) -> str:
        return f"({self.dependency[:3]}){self.num_tiles}x{self.tile_size}={self.size}"


class HBMTensor:
    """Represents a tensor stored in HBM with tiling configuration."""

    def __init__(self, name: str, axes: list[Axis]) -> None:
        """
        Args:
            name: Name of the HBM tensor
            axes: List of Axis objects defining tensor shape and tiling
        """
        self.name = name
        self.axes = axes

    def __repr__(self) -> str:
        axes_repr = ", ".join([repr(axis) for axis in self.axes])
        return f"HBMTensor({self.name}, [{axes_repr}])"


class TensorBuffer:
    """Represents an on-chip SBUF tensor buffer."""

    def __init__(self, name: str, shape: tuple[int, ...]) -> None:
        """
        Args:
            name: Name of the tensor buffer
            shape: Shape of the tensor buffer
        """
        self.name = name
        self.shape = shape

    def __repr__(self) -> str:
        return f"TensorBuffer({self.name}, {self.shape})"
