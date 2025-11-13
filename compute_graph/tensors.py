import math


class Axis:
    """Represents a tensor axis with tiling and dependency information."""

    def __init__(self, start_tile: int, end_tile: int, stride: int, tile_size: int, dependency: str) -> None:
        assert dependency in ["parallel", "reduction"]
        self.start_tile = start_tile
        self.end_tile = end_tile
        self.stride = stride
        self.tile_size = tile_size
        self.dependency = dependency
        self.num_tiles = (end_tile - start_tile) // stride
        self.size = self.num_tiles * self.tile_size

    def __repr__(self) -> str:
        return f"({self.dependency[:5]}){self.start_tile}:{self.end_tile}:{self.stride}x{self.tile_size}={self.size}"


class HBMTensor:
    """Represents a tensor stored in HBM with tiling configuration."""

    def __init__(
        self, name: str, shape: tuple[int, ...], tile_sizes: tuple[int, ...], axis_dependencies: tuple[str, ...]
    ) -> None:
        """
        Args:
            name: Name of the HBM tensor
        """
        self.name = name
        self.axes: list[Axis] = []
        for size, tile_size, dependency in zip(shape, tile_sizes, axis_dependencies):
            assert size % tile_size == 0, f"HBM tensor size {size} not divisible by tile_size {tile_size}"
            num_tiles = math.ceil(size / tile_size)
            axis = Axis(start_tile=0, end_tile=num_tiles, stride=1, tile_size=tile_size, dependency=dependency)
            self.axes.append(axis)
        self.shape = tuple([axis.size for axis in self.axes])

    @classmethod
    def from_axes(cls, name: str, axes: list[Axis]) -> "HBMTensor":
        """Create HBMTensor directly from pre-constructed Axis objects."""
        result = cls.__new__(cls)
        result.name = name
        result.axes = axes
        result.shape = tuple([axis.size for axis in axes])
        return result

    def access(self, indices: list[tuple[int, int, int]]) -> "HBMTensor":
        """Return new HBMTensor with axes sliced according to indices."""
        if len(indices) != len(self.axes):
            raise ValueError(f"Expected {len(self.axes)} indices, got {len(indices)}")

        new_axes = []
        for axis_idx, (axis, (start_tile, end_tile, stride)) in enumerate(zip(self.axes, indices)):
            if stride <= 0:
                raise ValueError(f"Axis {axis_idx}: stride must be positive, got {stride}")
            if start_tile >= end_tile:
                raise ValueError(f"Axis {axis_idx}: start_tile {start_tile} >= end_tile {end_tile}")
            for tile_idx in range(start_tile, end_tile, stride):
                if tile_idx not in range(axis.start_tile, axis.end_tile, axis.stride):
                    raise IndexError(f"Axis {axis_idx}: tile index {tile_idx} out of bounds {self}")

            new_axis = Axis(
                start_tile=start_tile,
                end_tile=end_tile,
                stride=stride,
                tile_size=axis.tile_size,
                dependency=axis.dependency,
            )
            new_axes.append(new_axis)

        new_tensor = HBMTensor.from_axes(self.name, new_axes)
        return new_tensor

    def __repr__(self) -> str:
        axes_repr = ", ".join([repr(axis) for axis in self.axes])
        return f"HBMTensor({self.name}, shape=[{axes_repr}])"


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
