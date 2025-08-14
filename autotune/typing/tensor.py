import math
from typing import Dict, Tuple

import neuronxcc.nki.language as nl


class HBMTensor:
    def __init__(self, tensor, axes: Tuple[str, str]) -> None:
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
        """
        tile_sizes[axis] = tile size along axis
        """
        self.tile_sizes = tile_sizes

    def load(self, hbm_tensor: HBMTensor, tile_offsets: Dict[str, int], num_tiles: Dict[str, int]) -> None:
        """
        tile_offsets[axis] = offsets in #tiles along axis
        num_tiles[axis] = number of tiles to load along axis
        0: load the entire input tensor
        """
        self.tile_offsets = tile_offsets
        self.axes = hbm_tensor.axes
        num_tiles = self._process_num_tiles(hbm_tensor, num_tiles)

        row_tile_size, column_tile_size = self.tile_sizes[self.axes[0]], self.tile_sizes[self.axes[1]]
        row_num_tiles, column_num_tiles = num_tiles[self.axes[0]], num_tiles[self.axes[1]]
        row_tile_offset, column_tile_offset = tile_offsets[self.axes[0]], tile_offsets[self.axes[1]]
        self.max_rows, self.max_columns = hbm_tensor.sizes[self.axes[0]], hbm_tensor.sizes[self.axes[1]]

        tile_index = nl.mgrid[0:row_tile_size, 0:column_tile_size]
        self.tensor = nl.ndarray(
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
                    mask=(row_indices < self.max_rows) & (column_indices < self.max_columns),
                )
        return result

    def _process_num_tiles(self, hbm_tensor: HBMTensor, num_tiles: Dict[str, int]) -> Dict[str, int]:
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
