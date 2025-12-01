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


def get_parallel_axes(input_tensors: list[HBMTensor], output_tensors: list[HBMTensor]) -> set[str]:
    """Determine parallel axes from input and output HBM tensors.

    Parallel axes are those that appear in both input and output tensors,
    matched by exact axis name. These axes can be processed independently
    in parallel.

    Args:
        input_tensors: List of input HBM tensors
        output_tensors: List of output HBM tensors

    Returns:
        Set of axis names that are parallel (appear in both inputs and outputs)
    """
    input_axes: set[str] = set()
    for tensor in input_tensors:
        for axis in tensor.axes:
            input_axes.add(axis.name)

    output_axes: set[str] = set()
    for tensor in output_tensors:
        for axis in tensor.axes:
            output_axes.add(axis.name)

    return input_axes & output_axes


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
