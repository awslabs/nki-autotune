class HBMAxis:
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

    def __init__(self, name: str, axes: tuple[HBMAxis, ...]) -> None:
        """
        Args:
            name: Name of the HBM tensor
        """
        self.name = name
        self.axes = axes
        self.shape = tuple(axis.size for axis in axes)

    def __repr__(self) -> str:
        axes_str = ", ".join(str(axis) for axis in self.axes)
        return f"HBMTensor({self.name}[{axes_str}])"


def get_parallel_axes(input_tensors: dict[str, HBMTensor], output_tensors: dict[str, HBMTensor]) -> list[str]:
    """Determine parallel axes from input and output HBM tensors.

    Parallel axes are those that appear in both input and output tensors,
    matched by exact axis name. These axes can be processed independently
    in parallel.

    Args:
        input_tensors: Dict of input HBM tensors (name -> tensor)
        output_tensors: Dict of output HBM tensors (name -> tensor)

    Returns:
        List of axis names that are parallel, sorted alphabetically
    """
    input_axes: set[str] = set()
    for tensor in input_tensors.values():
        for axis in tensor.axes:
            input_axes.add(axis.name)

    output_axes: set[str] = set()
    for tensor in output_tensors.values():
        for axis in tensor.axes:
            output_axes.add(axis.name)

    return sorted(input_axes & output_axes)


def _get_axis_sizes(tensors: dict[str, HBMTensor], axis_names: set[str]) -> dict[str, int]:
    """Extract axis sizes from tensors for given axis names."""
    axis_sizes: dict[str, int] = {}
    for tensor in tensors.values():
        for axis in tensor.axes:
            if axis.name in axis_names:
                if axis.name in axis_sizes:
                    if axis_sizes[axis.name] != axis.size:
                        raise ValueError(
                            f"Axis '{axis.name}' has inconsistent sizes: " f"{axis_sizes[axis.name]} vs {axis.size}"
                        )
                else:
                    axis_sizes[axis.name] = axis.size
    return axis_sizes


def get_num_shards(tensors: dict[str, HBMTensor], parallel_axes: list[str], tile_size: int = 128) -> int:
    """Return the total number of shards for the given tensors and parallel axes.

    Args:
        tensors: Dict of HBM tensors (name -> tensor)
        parallel_axes: Ordered list of parallel axis names
        tile_size: Tile size for partitioning (default 128)

    Returns:
        Total number of shards (product of tiles per axis)

    Raises:
        ValueError: If any parallel axis size is not divisible by tile_size
    """
    axis_sizes = _get_axis_sizes(tensors, set(parallel_axes))

    total_shards = 1
    for axis_name in parallel_axes:
        size = axis_sizes[axis_name]
        if size % tile_size != 0:
            raise ValueError(f"Axis '{axis_name}' size {size} is not divisible by tile_size {tile_size}")
        total_shards *= size // tile_size

    return total_shards


def shard_tensors(
    tensors: dict[str, HBMTensor], parallel_axes: list[str], parallel_index: int, tile_size: int = 128
) -> dict[str, HBMTensor]:
    """Shard tensors along parallel axes for a specific parallel index.

    Args:
        tensors: Dict of HBM tensors (name -> tensor)
        parallel_axes: Ordered list of parallel axis names
        parallel_index: Index of the shard to return (0 to get_num_shards()-1)
        tile_size: Tile size for partitioning (default 128)

    Returns:
        Dict of sharded tensors for the given parallel index
    """
    axis_sizes = _get_axis_sizes(tensors, set(parallel_axes))

    num_tiles_per_axis: dict[str, int] = {}
    for axis_name in parallel_axes:
        size = axis_sizes[axis_name]
        num_tiles_per_axis[axis_name] = size // tile_size

    tile_indices: dict[str, int] = {}
    remaining = parallel_index
    for axis_name in parallel_axes:
        num_tiles = num_tiles_per_axis[axis_name]
        tile_indices[axis_name] = remaining % num_tiles
        remaining //= num_tiles

    sharded_tensors: dict[str, HBMTensor] = {}
    for tensor_name, tensor in tensors.items():
        new_axes: list[HBMAxis] = []
        for axis in tensor.axes:
            if axis.name in tile_indices:
                tile_idx = tile_indices[axis.name]
                new_axis = HBMAxis(
                    name=axis.name, start_tile=tile_idx, end_tile=tile_idx + 1, stride=1, tile_size=tile_size
                )
            else:
                new_axis = HBMAxis(
                    name=axis.name,
                    start_tile=axis.start_tile,
                    end_tile=axis.end_tile,
                    stride=axis.stride,
                    tile_size=axis.tile_size,
                )
            new_axes.append(new_axis)
        sharded_tensors[tensor_name] = HBMTensor(name=tensor_name, axes=tuple(new_axes))

    return sharded_tensors


def create_hbm_tensor(name: str, shape: tuple[int, ...], axis_names: list[str] | None = None) -> HBMTensor:
    """Create an HBMTensor with default tiling (single tile per axis).

    Args:
        name: Tensor name
        shape: Tensor shape
        axis_names: Optional axis names. If None, auto-generates as "{name}_axis_{i}"
    """
    axes: list[HBMAxis] = []
    for i, size in enumerate(shape):
        ax_name = axis_names[i] if axis_names else f"{name}_axis_{i}"
        axis = HBMAxis(name=ax_name, start_tile=0, end_tile=1, stride=1, tile_size=size)
        axes.append(axis)
    tensor = HBMTensor(name=name, axes=tuple(axes))
    return tensor
