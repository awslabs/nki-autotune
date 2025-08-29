import math
from typing import Dict, Tuple

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np


class HBMTensor:
    """High Bandwidth Memory tensor wrapper with named axes.

    Provides an interface for tensors stored in HBM by associating
    dimension names with tensor axes and maintaining size mappings for efficient
    tiled operations with SBUF tensors.

    Sizes can be arbitrary and do not require any alignment.

    Attributes:
        tensor: The wrapped tensor data
        axes: Tuple of axis names corresponding to tensor dimensions
        sizes: Dictionary mapping axis names to their corresponding dimension sizes,
               enabling efficient lookup of tensor dimensions by name
    """

    def __init__(self, tensor, axes: Tuple[str, ...]) -> None:
        """Initialize HBM tensor wrapper with named axes.

        Args:
            tensor: The tensor data to wrap (numpy array or NKI tensor)
            axes: Tuple of axis names corresponding to tensor dimensions
                  (e.g., ("M", "N") for a 2D matrix)

        Raises:
            AssertionError: If number of axes doesn't match tensor dimensions
        """
        self.tensor = tensor
        self.axes = axes
        self.sizes: Dict[str, int] = {}
        assert len(axes) == len(
            tensor.shape
        ), f"Tensor {tensor.shape} has {len(tensor.shape)} axes, but was given {axes} {len(axes)} axes."
        for axis, size in zip(axes, tensor.shape):
            self.sizes[axis] = size


class SBUFTensor:
    def __init__(self, par_axis: str, tile_sizes: Dict[str, int]) -> None:
        """Initialize SBUF tensor with specified tile sizes.

        Args:
            par_axis: Partition axis name
            tile_sizes: Dictionary mapping axis names to tile sizes for axes
        """
        self.par_axis = par_axis
        self.tile_sizes = tile_sizes
        assert par_axis in tile_sizes, f"par_axis {par_axis} is not in tile_sizes {tile_sizes}."

    def load(self, source: HBMTensor, coordinates: Dict[str, Dict[str, int]]) -> None:
        """Load data from HBM tensor into SBUF tiles with automatic padding.

        Loads a region of the source tensor defined by coordinates into tiled SBUF memory.
        Automatically pads with zeros if the requested region extends beyond tensor boundaries.

        Args:
            source: HBM tensor to load data from
            coordinates: Region specification for each axis, where each axis maps to:
                - "start": Starting index in the source tensor
                - "size": Number of elements to load along this axis
        """
        assert set(coordinates.keys()) == set(self.tile_sizes.keys()), (
            f"Axes mismatch:"
            f"coordinates {coordinates},"
            f"tile_sizes has {self.tile_sizes}."
            f"Do not have exactly the same axes."
        )
        self.global_coordinates = coordinates
        self.axes = source.axes
        assert len(self.axes) == 2, f"Expected 2 axes, got {len(self.axes)}: {self.axes}"
        par_axis, free_axis = self.axes[0], self.axes[1]
        num_par_tiles = math.ceil(self.global_coordinates[par_axis]["size"] / self.tile_sizes[par_axis])
        num_free_tiles = math.ceil(self.global_coordinates[free_axis]["size"] / self.tile_sizes[free_axis])

        # Set attributes needed by other methods
        self.max_par_size = source.sizes[par_axis]
        self.max_free_size = source.sizes[free_axis]

        self.init_as_zero(num_tiles={par_axis: num_par_tiles, free_axis: num_free_tiles}, dtype=source.tensor.dtype)
        par_indices = nl.arange(self.tile_sizes[par_axis])[:, None]
        free_indices = nl.arange(self.tile_sizes[free_axis])[None, :]

        for par_tile_id in nl.affine_range(num_par_tiles):
            for free_tile_id in nl.affine_range(num_free_tiles):
                par_start = self.global_coordinates[par_axis]["start"] + par_tile_id * self.tile_sizes[par_axis]
                free_start = self.global_coordinates[free_axis]["start"] + free_tile_id * self.tile_sizes[free_axis]
                par_mask = par_start + par_indices < self.max_par_size
                free_mask = free_start + free_indices < self.max_free_size
                self.tensor[par_indices, par_tile_id, free_tile_id, free_indices] = nl.load(
                    source.tensor[par_start + par_indices, free_start + free_indices], mask=par_mask & free_mask
                )

    def init_as_zero(self, num_tiles: Dict[str, int], dtype):
        """
        (tile_size_0, num_tiles_0, num_tiles_1, ..., num_tiles_N-1, tile_size_1, ..., tile_size_N-1)
            0               1           2                 N             N + 1               2N - 1

        Args:
            num_tiles (Dict[str, int]): _description_
            dtype (_type_): _description_
        """
        tensor_shape = (
            self.tile_sizes[self.axes[0]],
            *[num_tiles[axis] for axis in self.axes],
            *[self.tile_sizes[axis] for axis in self.axes[1:]],
        )
        self.tensor = nl.zeros(tensor_shape, dtype=dtype, buffer=nl.sbuf)

    def dump(self):
        """Dump SBUF tensor data back to HBM.

        Returns:
            HBM tensor containing the dumped data
        """
        row_tile_size, row_num_tiles, column_num_tiles, column_tile_size = self.tensor.shape
        row_size = int(row_num_tiles * row_tile_size)
        column_size = int(column_num_tiles * column_tile_size)
        idx_res = nl.mgrid[0:row_tile_size, 0:column_tile_size]
        result = nl.ndarray((row_size, column_size), dtype=self.tensor.dtype, buffer=nl.shared_hbm)
        for row_tile_id in nl.affine_range(row_num_tiles):
            row_indices = row_tile_id * row_tile_size + idx_res.p
            for column_tile_id in nl.affine_range(column_num_tiles):
                column_indices = column_tile_id * column_tile_size + idx_res.x
                nl.store(
                    result[row_indices, column_indices],
                    value=self.tensor[idx_res.p, row_tile_id, column_tile_id, idx_res.x],
                )
        return result

    def tile_transpose(self):
        """Transpose tensor tile-by-tile in place.

        Performs transpose operation on each tile,
        handling boundary conditions for padded regions.
        """
        pmax = nl.tile_size.pmax
        if nisa.get_nc_version() == nisa.nc_version.gen3:
            tileT_dtype = self.tensor.dtype
        else:
            tileT_dtype = np.float32

        idx_transp = nl.mgrid[0:pmax, 0:pmax]
        row_tile_size, row_num_tiles, column_num_tiles, column_tile_size = self.tensor.shape
        num_row_transp_tiles = math.ceil(row_tile_size / pmax)
        num_column_transp_tiles = math.ceil(column_tile_size / pmax)
        row_tile_offset, column_tile_offset = self.tile_offsets[self.axes[0]], self.tile_offsets[self.axes[1]]

        for row_tile_id in nl.affine_range(row_num_tiles):
            for column_tile_id in nl.affine_range(column_num_tiles):
                for row_transp_tile_id in nl.affine_range(num_row_transp_tiles):
                    row_indices = row_transp_tile_id * pmax + idx_transp.p
                    row_mask = (row_tile_offset + row_tile_id) * row_tile_size + row_indices < self.max_rows
                    for column_transp_tile_id in nl.affine_range(num_column_transp_tiles):
                        column_indices = column_transp_tile_id * pmax + idx_transp.x
                        column_mask = (
                            column_tile_offset + column_tile_id
                        ) * column_tile_size + column_indices < self.max_columns
                        mask = row_mask & column_mask

                        tileT = nl.ndarray((nl.par_dim(pmax), pmax), dtype=tileT_dtype, buffer=nl.psum)
                        tileT[idx_transp.p, idx_transp.x] = nisa.nc_transpose(
                            self.tensor[row_indices, row_tile_id, column_tile_id, column_indices], mask=mask
                        )
                        self.tensor[row_indices, row_tile_id, column_tile_id, column_indices] = nl.copy(
                            tileT, dtype=self.tensor.dtype
                        )

    def read_tile(self, tile_indices: Dict[str, int]):
        """Extract a specific tile from the tensor.

        Args:
            tile_indices: Dictionary mapping axis names to tile indices

        Returns:
            The requested tile as a tensor
        """
        row_tile_size, row_num_tiles, column_num_tiles, column_tile_size = self.tensor.shape
        row_tile_index, column_tile_index = tile_indices[self.axes[0]], tile_indices[self.axes[1]]
        row_tile_offset, column_tile_offset = self.tile_offsets[self.axes[0]], self.tile_offsets[self.axes[1]]
        idx_tile = nl.mgrid[0:row_tile_size, 0:column_tile_size]
        row_mask = (row_tile_offset + row_tile_index) * row_tile_size + idx_tile.p < self.max_rows
        column_mask = (column_tile_offset + column_tile_index) * column_tile_size + idx_tile.x < self.max_columns
        tile = self.tensor[idx_tile.p, row_tile_index, column_tile_index, idx_tile.x][row_mask][column_mask]
        return tile

    def _pad_num_tiles(self, hbm_tensor: HBMTensor, num_tiles: Dict[str, int]) -> Dict[str, int]:
        """Process and validate num_tiles parameter, expanding zero values to maximum possible tiles.

        Converts num_tiles entries of 0 to the maximum number of tiles that can fit
        for each axis given the tensor size, tile size, and tile offset. The maximum
        tile count is calculated using math.ceil to handle cases where the tensor
        dimension is not evenly divisible by the tile size (padding the last tile).
        Also validates that requested tile counts are within valid bounds.

        Args:
            hbm_tensor: HBM tensor to calculate maximum tiles from
            num_tiles: Dictionary mapping axis names to number of tiles per axis.
                      Value of 0 means "load all remaining tiles from offset"

        Returns:
            Processed num_tiles dictionary with 0 values replaced by maximum possible tiles

        Raises:
            AssertionError: If any axis in num_tiles is not present in tile_sizes or tile_offsets
            AssertionError: If any requested tile count exceeds maximum possible or is negative
        """
        # Validate all axes in num_tiles exist in tile_sizes and tile_offsets
        for axis in num_tiles:
            assert axis in self.tile_sizes, f"Axis '{axis}' not found in tile_sizes: {set(self.tile_sizes.keys())}"
            assert (
                axis in self.tile_offsets
            ), f"Axis '{axis}' not found in tile_offsets: {set(self.tile_offsets.keys())}"
            assert (
                axis in hbm_tensor.sizes
            ), f"Axis '{axis}' not found in HBM tensor sizes: {set(hbm_tensor.sizes.keys())}"

        for axis in num_tiles:
            axis_num_tiles = num_tiles[axis]
            axis_size = hbm_tensor.sizes[axis]
            tile_size = self.tile_sizes[axis]
            tile_offset = self.tile_offsets[axis]

            # Calculate maximum tiles available from the offset position
            max_axis_num_tiles = math.ceil(axis_size / tile_size) - tile_offset

            # Validate original input before modification
            assert axis_num_tiles >= 0, f"num_tiles for axis '{axis}' must be non-negative, got {axis_num_tiles}"
            assert axis_num_tiles <= max_axis_num_tiles, (
                f"num_tiles for axis '{axis}' ({axis_num_tiles}) exceeds maximum possible "
                f"({max_axis_num_tiles}) given tensor size {axis_size}, tile size {tile_size}, "
                f"and tile offset {tile_offset}"
            )

            # Replace 0 with maximum possible tiles
            if axis_num_tiles == 0:
                num_tiles[axis] = max_axis_num_tiles

        return num_tiles


def linear_to_coordinates(linear_index: int, dimensions: dict, axes: Tuple[str, ...]) -> Tuple[int, ...]:
    """
    Convert a linear index to multi-dimensional coordinates.

    Args:
        linear_index (int): Linear index identifier
        dimensions (dict): Dictionary mapping axis names to size along that axis
                         e.g., {'M': 2, 'K': 1}
        axes: Tuple of axis names in the desired output order

    Returns:
        Tuple[int, ...]: Tuple of coordinate values in the same order as axes
                        e.g., (1, 0) for axes=('M', 'K')

    Example:
        >>> linear_to_coordinates(0, {'M': 2, 'K': 1}, ('M', 'K'))
        (0, 0)
        >>> linear_to_coordinates(1, {'M': 2, 'K': 1}, ('M', 'K'))
        (1, 0)
        >>> linear_to_coordinates(3, {'M': 2, 'K': 2, 'N': 2}, ('M', 'K', 'N'))
        (1, 1, 0)
    """
    if not dimensions:
        return ()

    # Get axis names and sizes in a consistent order
    sizes = [dimensions[axis] for axis in axes]

    # Calculate strides for each dimension
    # Stride for dimension i is the product of all dimensions to the right of i
    strides = []
    stride = 1
    for i in range(len(sizes) - 1, -1, -1):
        strides.insert(0, stride)
        stride *= sizes[i]

    # Convert linear index to multi-dimensional coordinates
    coordinates = []
    remaining_id = linear_index

    for i, axis in enumerate(axes):
        coordinate_value = remaining_id // strides[i]
        coordinates.append(coordinate_value)
        remaining_id = remaining_id % strides[i]

    return tuple(coordinates)
