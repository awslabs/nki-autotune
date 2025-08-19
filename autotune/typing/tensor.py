import math
from typing import Dict, Tuple

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np


class HBMTensor:
    """High Bandwidth Memory tensor wrapper with named axes.

    Provides a convenient interface for tensors stored in HBM by associating
    dimension names with tensor axes and maintaining size mappings for efficient
    tiled operations with SBUF tensors.

    Attributes:
        tensor: The wrapped tensor data
        axes: Tuple of axis names corresponding to tensor dimensions
        sizes: Dictionary mapping axis names to their corresponding dimension sizes,
               enabling efficient lookup of tensor dimensions by name
    """

    def __init__(self, tensor, axes: Tuple[str, str]) -> None:
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
    def __init__(self, tile_sizes: Dict[str, int]) -> None:
        """Initialize SBUF tensor with specified tile sizes.

        Args:
            tile_sizes: Dictionary mapping axis names to tile sizes
        """
        self.tile_sizes = tile_sizes

    def load(self, hbm_tensor: HBMTensor, tile_offsets: Dict[str, int], num_tiles: Dict[str, int]) -> None:
        """Load data from HBM tensor into SBUF with tiling.

        Args:
            hbm_tensor: Source HBM tensor to load from
            tile_offsets: Starting tile offsets for each axis
            num_tiles: Number of tiles to load per axis (0 = load all)
        """
        # Ensure all dictionaries have the same axes
        tile_sizes_axes = set(self.tile_sizes.keys())
        tile_offsets_axes = set(tile_offsets.keys())
        num_tiles_axes = set(num_tiles.keys())
        assert tile_sizes_axes == tile_offsets_axes == num_tiles_axes, (
            f"Axes mismatch: tile_sizes has {tile_sizes_axes}, "
            f"tile_offsets has {tile_offsets_axes}, "
            f"num_tiles has {num_tiles_axes}. All must have exactly the same axes."
        )

        self.tile_offsets = tile_offsets
        self.axes = hbm_tensor.axes
        num_tiles = self._pad_num_tiles(hbm_tensor, num_tiles)

        row_tile_size, column_tile_size = self.tile_sizes[self.axes[0]], self.tile_sizes[self.axes[1]]
        row_num_tiles, column_num_tiles = num_tiles[self.axes[0]], num_tiles[self.axes[1]]
        row_tile_offset, column_tile_offset = tile_offsets[self.axes[0]], tile_offsets[self.axes[1]]
        self.max_rows, self.max_columns = hbm_tensor.sizes[self.axes[0]], hbm_tensor.sizes[self.axes[1]]

        tile_index = nl.mgrid[0:row_tile_size, 0:column_tile_size]
        self.init_as_zero(num_tiles, hbm_tensor.tensor.dtype)
        for row_tile_id in nl.affine_range(row_num_tiles):
            row_start = (row_tile_offset + row_tile_id) * row_tile_size
            row_indices = row_start + tile_index.p
            row_mask = row_indices < self.max_rows
            for column_tile_id in nl.affine_range(column_num_tiles):
                column_start = (column_tile_offset + column_tile_id) * column_tile_size
                column_indices = column_start + tile_index.x
                column_mask = column_indices < self.max_columns
                self.tensor[tile_index.p, row_tile_id, column_tile_id, tile_index.x] = nl.load(
                    hbm_tensor.tensor[row_indices, column_indices], mask=row_mask & column_mask
                )

    def copy(self, sbuf_tensor: "SBUFTensor"):
        pass

    def init_as_zero(self, num_tiles: Dict[str, int], dtype):
        row_tile_size, column_tile_size = self.tile_sizes[self.axes[0]], self.tile_sizes[self.axes[1]]
        row_num_tiles, column_num_tiles = num_tiles[self.axes[0]], num_tiles[self.axes[1]]
        self.tensor = nl.zeros(
            (nl.par_dim(row_tile_size), row_num_tiles, column_num_tiles, column_tile_size), dtype=dtype, buffer=nl.sbuf
        )

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
