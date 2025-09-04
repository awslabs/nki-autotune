import math
from typing import Dict, Tuple

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np


class TileCoordinates:
    def __init__(self) -> None:
        self.axes = set()
        self.data = {}

    def add_axis(self, axis: str, start_tile_index: int, num_tiles: int):
        self.data[axis] = {"start_tile_index": start_tile_index, "num_tiles": num_tiles}
        self.axes.add(axis)

    def __getitem__(self, axis: str) -> Dict[str, int]:
        """Access coordinate information for a specific axis.

        Args:
            axis: The axis name to retrieve coordinates for

        Returns:
            Dictionary with 'start' and 'size' keys mapping to start_tile_index
            and num_tiles respectively

        Raises:
            KeyError: If the requested axis doesn't exist
        """
        if axis not in self.data:
            raise KeyError(f"Axis '{axis}' not found in TileCoordinates. Available axes: {list(self.data.keys())}")

        return self.data[axis]


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
    def __init__(self, par_axis: str, tile_sizes: Dict[str, int], tile_coordinates: TileCoordinates) -> None:
        """Initialize SBUF tensor with specified tile sizes.

        Args:
            par_axis: Partition axis name
            tile_sizes: Dictionary mapping axis names to tile sizes for axes
            tile_coordinates (TileCoordinates): Region specification for each axis, where each axis maps to:
                - "start": Start tile index in the source tensor
                - "size": Number of tiles to load along this axis
        """
        assert par_axis in tile_sizes, f"par_axis {par_axis} is not in tile_sizes {tile_sizes}."
        self.par_axis = par_axis
        self.tile_sizes = tile_sizes
        free_axes = []
        for axis in tile_sizes:
            if axis != par_axis:
                free_axes.append(axis)
        assert len(free_axes) == 1, f"Expected 1 free axis, got {len(free_axes)} : {free_axes}"
        self.free_axis = free_axes[0]
        assert tile_coordinates.axes == set(self.tile_sizes.keys()), (
            f"Axes mismatch:"
            f"tile_coordinates {tile_coordinates},"
            f"tile_sizes {self.tile_sizes}."
            f"Do not have exactly the same axes."
        )
        self.tile_coordinates = tile_coordinates

    def load(self, source: HBMTensor) -> None:
        """Load data from HBM tensor into SBUF tiles with automatic padding.

        Loads a region of the source tensor defined by coordinates into tiled SBUF memory.
        Automatically pads with zeros if the requested region extends beyond tensor boundaries.

        Args:
            source: HBM tensor to load data from
        """
        assert len(source.axes) == 2, f"Expected 2 axes, got {len(source.axes)}: {source.axes}"

        # Set attributes needed by other methods
        self.max_par_size = source.sizes[self.par_axis]
        self.max_free_size = source.sizes[self.free_axis]

        self.init_as_zero(dtype=source.tensor.dtype)
        par_indices = nl.arange(self.tile_sizes[self.par_axis])[:, None]
        free_indices = nl.arange(self.tile_sizes[self.free_axis])[None, :]

        par_tile_offset = self.tile_coordinates[self.par_axis]["start_tile_index"]
        free_tile_offset = self.tile_coordinates[self.free_axis]["start_tile_index"]

        for par_tile_id in nl.affine_range(self.tile_coordinates[self.par_axis]["num_tiles"]):
            par_start = (par_tile_offset + par_tile_id) * self.tile_sizes[self.par_axis]
            par_mask = par_start + par_indices < self.max_par_size
            for free_tile_id in nl.affine_range(self.tile_coordinates[self.free_axis]["num_tiles"]):
                free_start = (free_tile_offset + free_tile_id) * self.tile_sizes[self.free_axis]
                free_mask = free_start + free_indices < self.max_free_size
                self.tensor[par_indices, par_tile_id, free_tile_id, free_indices] = nl.load(
                    source.tensor[par_start + par_indices, free_start + free_indices], mask=par_mask & free_mask
                )

    def init_as_zero(self, dtype):
        """
        (tile_size_0, num_tiles_0, num_tiles_1, ..., num_tiles_N-1, tile_size_1, ..., tile_size_N-1)
            0               1           2                 N             N + 1               2N - 1

        Args:
            num_tiles (Dict[str, int]): _description_
            dtype (_type_): _description_
        """
        tensor_shape = (
            self.tile_sizes[self.par_axis],
            self.tile_coordinates[self.par_axis]["num_tiles"],
            self.tile_coordinates[self.free_axis]["num_tiles"],
            self.tile_sizes[self.free_axis],
        )
        self.tensor = nl.zeros(tensor_shape, dtype=dtype, buffer=nl.sbuf)

    def dump(self):
        """Dump SBUF tensor data back to HBM.

        Returns:
            HBM tensor containing the dumped data
        """
        par_tile_size, num_par_tiles, num_free_tiles, free_tile_size = self.tensor.shape
        par_size = int(num_par_tiles * par_tile_size)
        free_size = int(num_free_tiles * free_tile_size)
        idx_res = nl.mgrid[0:par_tile_size, 0:free_tile_size]
        result = nl.ndarray((par_size, free_size), dtype=self.tensor.dtype, buffer=nl.shared_hbm)
        for par_tile_id in nl.affine_range(num_par_tiles):
            par_indices = par_tile_id * par_tile_size + idx_res.p
            for free_tile_id in nl.affine_range(num_free_tiles):
                free_indices = free_tile_id * free_tile_size + idx_res.x
                nl.store(
                    result[par_indices, free_indices],
                    value=self.tensor[idx_res.p, par_tile_id, free_tile_id, idx_res.x],
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
        par_tile_size, num_par_tiles, num_free_tiles, free_tile_size = self.tensor.shape
        num_par_transp_tiles = math.ceil(par_tile_size / pmax)
        num_free_transp_tiles = math.ceil(free_tile_size / pmax)
        par_tile_offset = self.tile_coordinates[self.par_axis]["start_tile_index"]
        free_tile_offset = self.tile_coordinates[self.free_axis]["start_tile_index"]

        for par_tile_id in nl.affine_range(num_par_tiles):
            for free_tile_id in nl.affine_range(num_free_tiles):
                for par_transp_tile_id in nl.affine_range(num_par_transp_tiles):
                    par_indices = par_transp_tile_id * pmax + idx_transp.p
                    par_mask = (par_tile_offset + par_tile_id) * par_tile_size + par_indices < self.max_par_size
                    for free_transp_tile_id in nl.affine_range(num_free_transp_tiles):
                        free_indices = free_transp_tile_id * pmax + idx_transp.x
                        free_mask = (
                            free_tile_offset + free_tile_id
                        ) * free_tile_size + free_indices < self.max_free_size
                        mask = par_mask & free_mask

                        tileT = nl.ndarray((nl.par_dim(pmax), pmax), dtype=tileT_dtype, buffer=nl.psum)
                        tileT[idx_transp.p, idx_transp.x] = nisa.nc_transpose(
                            self.tensor[par_indices, par_tile_id, free_tile_id, free_indices], mask=mask
                        )
                        self.tensor[par_indices, par_tile_id, free_tile_id, free_indices] = nl.copy(
                            tileT, dtype=self.tensor.dtype
                        )

    def read_tile(self, tile_indices: Dict[str, int]):
        """Extract a specific tile from the tensor using global tile indices.

        Args:
            tile_indices: Dictionary mapping axis names to global tile indices

        Returns:
            The requested tile as a tensor
        """
        par_tile_size, par_num_tiles, free_num_tiles, free_tile_size = self.tensor.shape

        # Convert global indices to local indices
        par_tile_index = tile_indices[self.par_axis] - self.tile_coordinates[self.par_axis]["start_tile_index"]
        free_tile_index = tile_indices[self.free_axis] - self.tile_coordinates[self.free_axis]["start_tile_index"]

        # Validate that the indices are within bounds
        assert 0 <= par_tile_index < par_num_tiles, (
            f"Global {self.par_axis} tile index {tile_indices[self.par_axis]} "
            f"(local: {par_tile_index}) out of range [0, {par_num_tiles})"
        )
        assert 0 <= free_tile_index < free_num_tiles, (
            f"Global {self.free_axis} tile index {tile_indices[self.free_axis]} "
            f"(local: {free_tile_index}) out of range [0, {free_num_tiles})"
        )

        idx_tile = nl.mgrid[0:par_tile_size, 0:free_tile_size]
        tile = self.tensor[idx_tile.p, par_tile_index, free_tile_index, idx_tile.x]
        return tile
