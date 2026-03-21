"""Tile-aware tensor wrappers for HBM and SBUF memory spaces."""

from typing import Any

import nki.isa as nisa
import nki.language as nl


class TileCoordinates(nl.NKIObject):
    """Region specification for tiled tensor operations.

    Tracks per-axis tile start indices and tile counts.
    """

    def __init__(self) -> None:
        """Initialize empty tile coordinates."""
        self.data: dict[str, dict[str, int]] = {}

    def add_axis(self, axis: str, start_tile_index: int, num_tiles: int) -> None:
        """Register a tile range for the given axis.

        Args:
            axis: Axis name (e.g. "M", "N", "K").
            start_tile_index: First tile index in the source tensor.
            num_tiles: Number of tiles along this axis.
        """
        self.data[axis] = {"start_tile_index": start_tile_index, "num_tiles": num_tiles}

    def get_start(self, axis: str) -> int:
        """Get the start tile index for an axis.

        Args:
            axis: Axis name.

        Returns:
            Start tile index.
        """
        return self.data[axis]["start_tile_index"]

    def get_num_tiles(self, axis: str) -> int:
        """Get the number of tiles for an axis.

        Args:
            axis: Axis name.

        Returns:
            Number of tiles.
        """
        return self.data[axis]["num_tiles"]

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TileCoordinates({self.data})"


class HBMTensor(nl.NKIObject):
    """HBM tensor wrapper with named axes.

    Associates dimension names with tensor axes for tiled operations.

    Attributes:
        tensor: The wrapped tensor data.
        axis0: Name of the first dimension.
        axis1: Name of the second dimension.
        sizes: Mapping from axis name to dimension size.
    """

    def __init__(self, tensor: Any, axis0: str, axis1: str) -> None:
        """Initialize HBM tensor with named axes.

        Args:
            tensor: The tensor data to wrap.
            axis0: Name for first dimension.
            axis1: Name for second dimension.
        """
        self.tensor = tensor
        self.axis0 = axis0
        self.axis1 = axis1
        self.sizes: dict[str, int] = {}
        self.sizes[axis0] = tensor.shape[0]
        self.sizes[axis1] = tensor.shape[1]
        self.dtype = tensor.dtype


def _transpose_tile(sbuf_tensor: Any, par_tid: int, free_tid: int, pmax: int, tileT_dtype: Any) -> None:
    """Transpose a single tile at (par_tid, free_tid) in place.

    Args:
        sbuf_tensor: The 4D SBUF tensor to transpose in-place.
        par_tid: Partition tile index.
        free_tid: Free tile index.
        pmax: Partition axis maximum (hardware constant).
        tileT_dtype: Data type for intermediate transpose buffer.
    """
    tileT = nl.ndarray((pmax, pmax), dtype=tileT_dtype, buffer=nl.psum)
    nisa.nc_transpose(
        dst=tileT[0:pmax, 0:pmax], data=sbuf_tensor[0:pmax, par_tid : par_tid + 1, free_tid : free_tid + 1, 0:pmax]
    )
    nisa.tensor_copy(
        dst=sbuf_tensor[0:pmax, par_tid : par_tid + 1, free_tid : free_tid + 1, 0:pmax], src=tileT[0:pmax, 0:pmax]
    )


def _transpose_all_tiles(sbuf_tensor: Any, num_par_tiles: int, num_free_tiles: int, tileT_dtype: Any) -> None:
    """Execute tile-by-tile transpose over all tiles.

    Args:
        sbuf_tensor: The 4D SBUF tensor to transpose.
        num_par_tiles: Number of partition tiles.
        num_free_tiles: Number of free tiles.
        tileT_dtype: Data type for intermediate transpose buffers.
    """
    pmax = nl.tile_size.pmax
    for par_tid in nl.affine_range(num_par_tiles):
        for free_tid in nl.affine_range(num_free_tiles):
            _transpose_tile(sbuf_tensor, par_tid, free_tid, pmax, tileT_dtype)


class SBUFTensor(nl.NKIObject):
    """SBUF tile buffer with named axes.

    Manages tiled data in SBUF with load/store/transpose operations.

    Attributes:
        par_axis: Partition axis name.
        free_axis: Free axis name.
        tile_sizes: Per-axis tile sizes.
        tile_coordinates: Region specification for each axis.
        tensor: The underlying 4D SBUF tensor.
    """

    def __init__(
        self, par_axis: str, free_axis: str, tile_sizes: dict[str, int], tile_coordinates: TileCoordinates, name: str
    ) -> None:
        """Initialize SBUF tensor.

        Args:
            par_axis: Partition axis name.
            free_axis: Free axis name.
            tile_sizes: Mapping of axis names to tile sizes.
            tile_coordinates: Region spec for each axis.
            name: Unique tensor name for NKI allocation.
        """
        self.par_axis = par_axis
        self.free_axis = free_axis
        self.tile_sizes = tile_sizes
        self.tile_coordinates = tile_coordinates
        self.name = name

    def load(self, source: HBMTensor) -> None:
        """Load data from HBM into SBUF tiles via DMA copy.

        Args:
            source: HBM tensor to load from.
        """
        self.init_as_zero(dtype=source.tensor.dtype)
        self._load_tiles_from_hbm(source)

    def _load_tiles_from_hbm(self, source: HBMTensor) -> None:
        """Copy tile data from HBM using DMA copy with slice notation.

        Args:
            source: HBM tensor to copy from.
        """
        ts_par = self.tile_sizes[self.par_axis]
        ts_free = self.tile_sizes[self.free_axis]
        par_offset = self.tile_coordinates.get_start(self.par_axis)
        free_offset = self.tile_coordinates.get_start(self.free_axis)
        num_par = self.tile_coordinates.get_num_tiles(self.par_axis)
        num_free = self.tile_coordinates.get_num_tiles(self.free_axis)

        for par_tid in nl.affine_range(num_par):
            par_start = (par_offset + par_tid) * ts_par
            for free_tid in nl.affine_range(num_free):
                free_start = (free_offset + free_tid) * ts_free
                nisa.dma_copy(
                    dst=self.tensor[0:ts_par, par_tid : par_tid + 1, free_tid : free_tid + 1, 0:ts_free],
                    src=source.tensor[par_start : par_start + ts_par, free_start : free_start + ts_free],
                )

    def init_as_zero(self, dtype: Any) -> None:
        """Allocate zero-initialized 4D SBUF tensor.

        Shape: (par_tile_size, num_par_tiles, num_free_tiles, free_tile_size).

        Args:
            dtype: Data type for the tensor.
        """
        tensor_shape = (
            self.tile_sizes[self.par_axis],
            self.tile_coordinates.get_num_tiles(self.par_axis),
            self.tile_coordinates.get_num_tiles(self.free_axis),
            self.tile_sizes[self.free_axis],
        )
        self.tensor = nl.zeros(tensor_shape, dtype=dtype, buffer=nl.sbuf, name=self.name)

    def tile_transpose(self) -> None:
        """Transpose tensor tile-by-tile in place."""
        num_par_tiles = self.tile_coordinates.get_num_tiles(self.par_axis)
        num_free_tiles = self.tile_coordinates.get_num_tiles(self.free_axis)
        _transpose_all_tiles(self.tensor, num_par_tiles, num_free_tiles, self.tensor.dtype)

    def read_tile(self, tile_indices: dict[str, int]) -> Any:
        """Extract a tile using global tile indices.

        Args:
            tile_indices: Mapping of axis names to global tile indices.

        Returns:
            The requested tile tensor slice (4D).
        """
        par_tile_size = self.tile_sizes[self.par_axis]
        free_tile_size = self.tile_sizes[self.free_axis]
        par_local = tile_indices[self.par_axis] - self.tile_coordinates.get_start(self.par_axis)
        free_local = tile_indices[self.free_axis] - self.tile_coordinates.get_start(self.free_axis)
        return self.tensor[0:par_tile_size, par_local : par_local + 1, free_local : free_local + 1, 0:free_tile_size]

    def save_to_hbm(self, result: Any) -> None:
        """Store SBUF tiles into a 2D HBM tensor via DMA copy.

        Args:
            result: Destination tensor with shape (M, N).
        """
        ts_par = self.tile_sizes[self.par_axis]
        ts_free = self.tile_sizes[self.free_axis]
        par_start_idx = self.tile_coordinates.get_start(self.par_axis)
        free_start_idx = self.tile_coordinates.get_start(self.free_axis)
        num_par = self.tile_coordinates.get_num_tiles(self.par_axis)
        num_free = self.tile_coordinates.get_num_tiles(self.free_axis)

        for row_off in nl.affine_range(num_par):
            g_row = par_start_idx + row_off
            row_start = g_row * ts_par
            for col_off in nl.affine_range(num_free):
                g_col = free_start_idx + col_off
                col_start = g_col * ts_free
                nisa.dma_copy(
                    dst=result[row_start : row_start + ts_par, col_start : col_start + ts_free],
                    src=self.tensor[0:ts_par, row_off : row_off + 1, col_off : col_off + 1, 0:ts_free],
                )
