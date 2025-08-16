import math
from typing import Dict, Tuple

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np


class HBMTensor:
    def __init__(self, tensor, axes: Tuple[str, str]) -> None:
        """Initialize HBM tensor wrapper with axis names.

        Args:
            tensor: Input tensor data
            axes: Tuple of axis names (e.g., ("M", "N"))
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
        print(f"SBUFTensor.tile_sizes = {self.tile_sizes}")

    def load(self, hbm_tensor: HBMTensor, tile_offsets: Dict[str, int], num_tiles: Dict[str, int]) -> None:
        """Load data from HBM tensor into SBUF with tiling.

        Args:
            hbm_tensor: Source HBM tensor to load from
            tile_offsets: Starting tile offsets for each axis
            num_tiles: Number of tiles to load per axis (0 = load all)
        """
        self.tile_offsets = tile_offsets
        self.axes = hbm_tensor.axes
        num_tiles = self._process_num_tiles(hbm_tensor, num_tiles)

        row_tile_size, column_tile_size = self.tile_sizes[self.axes[0]], self.tile_sizes[self.axes[1]]
        row_num_tiles, column_num_tiles = num_tiles[self.axes[0]], num_tiles[self.axes[1]]
        row_tile_offset, column_tile_offset = tile_offsets[self.axes[0]], tile_offsets[self.axes[1]]
        self.max_rows, self.max_columns = hbm_tensor.sizes[self.axes[0]], hbm_tensor.sizes[self.axes[1]]

        tile_index = nl.mgrid[0:row_tile_size, 0:column_tile_size]
        self.tensor = nl.zeros(
            (nl.par_dim(row_tile_size), row_num_tiles, column_num_tiles, column_tile_size),
            dtype=hbm_tensor.tensor.dtype,
            buffer=nl.sbuf,
        )
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
        assert (
            row_tile_index < row_num_tiles and column_tile_index < column_num_tiles
        ), f"Out of bound access of tile {tile_indices} in a {self.tensor.shape} tensor."
        idx_tile = nl.mgrid[0:row_tile_size, 0:column_tile_size]
        tile = self.tensor[idx_tile.p, row_tile_index, column_tile_index, idx_tile.x]
        return tile

    def _process_num_tiles(self, hbm_tensor: HBMTensor, num_tiles: Dict[str, int]) -> Dict[str, int]:
        """Process and validate num_tiles parameter.

        Args:
            hbm_tensor: HBM tensor to calculate max tiles from
            num_tiles: Dictionary of number of tiles per axis (0 = all tiles)

        Returns:
            Processed num_tiles with 0 values replaced by maximum possible tiles
        """
        for axis in num_tiles:
            axis_num_tiles = num_tiles[axis]
            axis_size = hbm_tensor.sizes[axis]
            max_axis_num_tiles = math.ceil(axis_size / self.tile_sizes[axis])
            assert (
                axis_num_tiles <= max_axis_num_tiles and axis_num_tiles >= 0
            ), f"axis_num_tiles {axis_num_tiles} is out of bound for [0, {max_axis_num_tiles}]."
            if axis_num_tiles == 0:
                num_tiles[axis] = max_axis_num_tiles
        return num_tiles
