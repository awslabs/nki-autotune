class Axis:
    """Represents a tensor axis with tiling information."""

    def __init__(self, name: str, start_tile: int, end_tile: int, stride: int, tile_size: int) -> None:
        self.name = name
        self.start_tile = start_tile
        self.end_tile = end_tile
        self.stride = stride
        self.tile_size = tile_size
        self.num_tiles = (end_tile - start_tile) // stride
        self.size = self.num_tiles * self.tile_size

    def __repr__(self) -> str:
        return f"{self.name}:({self.start_tile},{self.end_tile},{self.stride})x{self.tile_size}={self.size}"


class HBMTensor:
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
        return f"HBMTensor({self.name}[{axes_str}])"


def create_hbm_tensor(name: str, shape: tuple[int, ...]) -> HBMTensor:
    axes: list[Axis] = []
    for i, size in enumerate(shape):
        axis = Axis(name=f"{name}_axis_{i}", start_tile=0, end_tile=1, stride=1, tile_size=size)
        axes.append(axis)
    tensor = HBMTensor(name=name, axes=tuple(axes))
    return tensor
