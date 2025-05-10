import neuronxcc.nki.language as nl


def scale_tile(input_tile, scale_tile):
    """
    Scale rows of a 2D tile by corresponding values in a scale tile.

    This function performs in-place scaling of each row in the input tile by
    multiplying it with the corresponding scalar value from the scale tile.

    Parameters:
    -----------
    input_tile : ndarray
        A 2D array with shape (par_size, free_size) that will be modified in-place.

    scale_tile : ndarray
        A 2D array with shape (par_size, 1) containing scaling factors for each row
        of the input tile.

    Returns:
    --------
    None
        The function modifies input_tile in-place.
    """
    par_size, free_size = input_tile.shape
    assert scale_tile.shape == (
        par_size,
        1,
    ), f"Scale tile shape mismatch: expected ({par_size}, 1), got {scale_tile.shape}. The scale_tile must have free size = 1 and the same number of partition dimension as input_tile."
    i_grid = nl.mgrid[0:par_size, 0:free_size]
    scaled_tile = nl.ndarray((nl.par_dim(par_size), free_size), dtype=input_tile.dtype, buffer=nl.sbuf)
    scaled_tile[...] = nl.multiply(input_tile[i_grid.p, i_grid.x], scale_tile[...], dtype=input_tile.dtype)
    input_tile[i_grid.p, i_grid.x] = nl.copy(scaled_tile, dtype=input_tile.dtype)
