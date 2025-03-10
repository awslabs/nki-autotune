import neuronxcc.nki.language as nl


def load_tensor_by_par_tiles(
    input_tensor, num_par_tiles: int, par_tile_size: int, free_dim_size: int, par_ofs: int, free_ofs: int
):
    """
    Load a rectangular 2D tile of shape (num_par_tiles * par_tile_size, free_dim_size) from the input tensor.
    The location of the 2D tile from the input is offset by (par_ofs, free_ofs).
    Load the input tile by tile in parallel in the par dimension.

    Input tensor (par_size, free_size):
    +------------------+
    |                  |
    |    +--------+    |  ← Starting at (par_ofs, free_ofs)
    |    |Tile 0  |    |
    |    |Tile 1  |    |  Each tile is (par_tile_size * free_dim_size)
    |    |  ...   |    |
    |    |Tile N-1|    |  N = num_par_tiles
    |    +--------+    |
    |                  |
    +------------------+

    Args:
        input_tensor: the input tensor to load from, arranged in (par_size, free_size)
        num_par_tiles: number of partition tiles to load
        par_tile_size: the size of each partition tile
        free_dim_size: the size of free dimension to load
        par_ofs: offset in the partition dimension
        free_ofs: offset in the free dimension

    Returns:
        Loaded tiles in SBUF in the shape of (num_par_tiles, nl.par_dim(par_tile_size), free_dim_size)
    TODO: adapt this into a more general loading function that handles both hoist vs no hoist.
    This version can be viewed as no hoist.
    """

    tiles = nl.ndarray(
        (num_par_tiles, nl.par_dim(par_tile_size), free_dim_size), dtype=input_tensor.dtype, buffer=nl.sbuf
    )

    idx = nl.mgrid[0:par_tile_size, 0:free_dim_size]
    for par_tile_id in nl.affine_range(num_par_tiles):
        tiles[par_tile_id, idx.p, idx.x] = nl.load(
            input_tensor[par_ofs + par_tile_id * par_tile_size + idx.p, free_ofs + idx.x]
        )
    return tiles
