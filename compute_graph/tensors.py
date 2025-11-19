class Axis:
    """Represents a tensor axis with tiling information."""

    def __init__(self, start_tile: int, end_tile: int, stride: int, tile_size: int) -> None:
        self.start_tile = start_tile
        self.end_tile = end_tile
        self.stride = stride
        self.tile_size = tile_size
        self.num_tiles = (end_tile - start_tile) // stride
        self.size = self.num_tiles * self.tile_size

    def __repr__(self) -> str:
        return f"{self.start_tile}:{self.end_tile}:{self.stride}x{self.tile_size}={self.size}"


class Tensor:
    """Represents a tensor stored in HBM with tiling configuration."""

    def __init__(self, name: str, axes: tuple[Axis, ...]) -> None:
        """
        Args:
            name: Name of the HBM tensor
        """
        self.name = name
        self.axes = axes
        self.shape = tuple([axis.size for axis in axes])

    def __repr__(self) -> str:
        axes_str = ", ".join(str(axis) for axis in self.axes)
        return f"Tensor({self.name}, axes=[{axes_str}])"


class TensorBuffer:
    """Represents an on-chip SBUF/PSUM tensor buffer."""

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
