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
        return f"Tensor({self.name}[{axes_str}])"


def shape_to_axes(shape: tuple[int, ...]) -> tuple[Axis, ...]:
    axes: list[Axis] = []
    for size in shape:
        axis = Axis(start_tile=0, end_tile=1, stride=1, tile_size=size)
        axes.append(axis)
    return tuple(axes)


def create_tensor(name: str, shape: tuple[int, ...]) -> Tensor:
    axes = shape_to_axes(shape)
    tensor = Tensor(name=name, axes=axes)
    return tensor
