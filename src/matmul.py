# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki as nki
import numpy as np


def common_head(lhsT, rhs, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N):
    """NKI kernel to compute a large matrix multiplication efficiently by
       blocking all dimensions and doing layout optimization.

    Args:
        lhsT: an input tensor of shape [K,M], where K is a multiple of 128 *
          TILES_IN_BLOCK_K and M is a multiple of 128 * TILES_IN_BLOCK_M.  It is the
          left-hand-side argument of the matrix multiplication, delivered transposed
          for optimal performance.
        rhs: an input tensor of shape [K,N],  where K is a multiple of 128 *
          TILES_IN_BLOCK_K and N is a multiple of 512 * TILES_IN_BLOCK_N.  It is
          the right-hand-side argument of the matrix multiplication.
        TILES_IN_BLOCK_*: meta parameters to control blocking dimensions

        result: the resulting output tensor of shape [M, N]
    """
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert (
        K == K_
    ), f"lhsT and rhs contraction dimension mismatch, got {lhsT.shape} and {rhs.shape}"

    # Tile sizes
    TILE_K = nl.tile_size.pmax  # 128
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    # Block sizes
    BLOCK_K = TILE_K * TILES_IN_BLOCK_K
    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N

    # Number of blocks
    NUM_BLOCK_K = K // BLOCK_K
    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N

    assert NUM_BLOCK_K * TILES_IN_BLOCK_K * TILE_K == K
    assert NUM_BLOCK_M * TILES_IN_BLOCK_M * TILE_M == M
    assert NUM_BLOCK_N * TILES_IN_BLOCK_N * TILE_N == N

    return (
        M,
        N,
        K,
        TILE_M,
        TILE_N,
        TILE_K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_BLOCK_M,
        NUM_BLOCK_N,
        NUM_BLOCK_K,
    )


def load_tensor_by_par_tiles(
    input_tensor, num_par_tiles, par_tile_size, free_dim_size, par_offset, free_offset
):
    """
    Load a rectangular area of shape (num_par_tiles * par_tile_size, free_dim_size) from the input tensor.
    The location of the rectangle from the input is offset by (par_offset, free_offset).
    Load the input tile by tile in parallel in the par dimension.

    Args:
        input_tensor: the input tensor to load from
        num_par_tiles: number of partition tiles to load
        par_tile_size: the size of each partition tile
        free_dim_size: the size of free dimension to load
        par_offset: offset in the partition dimension
        free_offset: offset in the free dimension

    Returns:
        Loaded tiles in SBUF in the shape of (num_par_tiles, nl.par_dim(par_tile_size), free_dim_size)
    TODO: adapt this into a more general loading function that handles both hoist vs no hoist.
    This version can be viewed as no hoist.
    """
    tiles = nl.ndarray(
        (num_par_tiles, nl.par_dim(par_tile_size), free_dim_size),
        dtype=input_tensor.dtype,
        buffer=nl.sbuf,
    )
    for par_tile_id in nl.affine_range(num_par_tiles):
        par_start = par_offset + par_tile_id * par_tile_size
        par_end = par_start + par_tile_size
        free_start = free_offset
        free_end = free_start + free_dim_size
        tiles[par_tile_id] = nl.load(
            input_tensor[par_start:par_end, free_start:free_end]
        )
    return tiles


def matmul_tiles(lhsT_tiles, rhs_tiles, result_tiles, result_dtype, block_id_K):
    num_k_tiles, TILE_K, m = lhsT_tiles.shape
    _num_k_tiles, _TILE_K, n = rhs_tiles.shape
    num_m_tiles, TILE_M, _n = result_tiles.shape
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512
    num_n_tiles = n // TILE_N
    assert (
        num_k_tiles == _num_k_tiles and TILE_K == _TILE_K
    ), f"lhsT_tiles {lhsT_tiles.shape} does not match with rhs_tiles {rhs_tiles.shape}"
    assert (
        m == num_m_tiles * TILE_M
    ), f"lhsT_tiles {lhsT_tiles.shape} does not match with result_tiles {result_tiles.shape}"
    assert (
        n == _n
    ), f"rhs_tiles {rhs_tiles.shape} does not match with result_tiles {result_tiles.shape}"
    assert (
        n == num_n_tiles * TILE_N
    ), f"N tile size {n} is not divisible by hardware tile size {TILE_N}"

    for tile_id_M in nl.affine_range(num_m_tiles):
        for tile_id_N in nl.affine_range(num_n_tiles):
            res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

            m_start = tile_id_M * TILE_M
            m_end = m_start + TILE_M

            n_start = tile_id_N * TILE_N
            n_end = n_start + TILE_N

            # Use PSUM buffer to accumulate into a single hardware tile
            # to minimize the number of calls to nl.loop_reduce
            for tile_id_K in nl.affine_range(num_k_tiles):
                res_tile += nisa.nc_matmul(
                    lhsT_tiles[tile_id_K, :, m_start:m_end],
                    rhs_tiles[tile_id_K, :, n_start:n_end],
                )

            # Block-level accumulate into the full result
            result_tiles[tile_id_M, :, n_start:n_end] = nl.loop_reduce(
                res_tile,
                op=np.add,
                loop_indices=[block_id_K],
                dtype=result_dtype,
            )


def save_result(result, result_tiles, m_offset, n_offset):
    num_m_tiles, TILE_M, n_size = result_tiles.shape
    for tile_id_M in nl.affine_range(num_m_tiles):
        m_start = m_offset + tile_id_M * TILE_M
        m_end = m_start + TILE_M

        n_start = n_offset
        n_end = n_start + n_size

        nl.store(result[m_start:m_end, n_start:n_end], value=result_tiles[tile_id_M])


@nki.jit
def matmul_NMK(lhsT, rhs, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N):
    (
        M,
        N,
        K,
        TILE_M,
        TILE_N,
        TILE_K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_BLOCK_M,
        NUM_BLOCK_N,
        NUM_BLOCK_K,
    ) = common_head(lhsT, rhs, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N)

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    for block_id_N in nl.affine_range(NUM_BLOCK_N):
        for block_id_M in nl.affine_range(NUM_BLOCK_M):
            result_tiles = nl.zeros(
                (TILES_IN_BLOCK_M, nl.par_dim(TILE_M), BLOCK_N),
                dtype=result.dtype,
                buffer=nl.sbuf,
            )

            for block_id_K in nl.affine_range(NUM_BLOCK_K):
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_M
                lhsT_tiles = load_tensor_by_par_tiles(
                    input_tensor=lhsT,
                    num_par_tiles=TILES_IN_BLOCK_K,
                    par_tile_size=TILE_K,
                    free_dim_size=BLOCK_M,
                    par_offset=block_id_K * BLOCK_K,
                    free_offset=block_id_M * BLOCK_M,
                )
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_N
                rhs_tiles = load_tensor_by_par_tiles(
                    input_tensor=rhs,
                    num_par_tiles=TILES_IN_BLOCK_K,
                    par_tile_size=TILE_K,
                    free_dim_size=BLOCK_N,
                    par_offset=block_id_K * BLOCK_K,
                    free_offset=block_id_N * BLOCK_N,
                )
                matmul_tiles(
                    lhsT_tiles, rhs_tiles, result_tiles, result.dtype, block_id_K
                )

            save_result(
                result,
                result_tiles,
                m_offset=block_id_M * BLOCK_M,
                n_offset=block_id_N * BLOCK_N,
            )
    return result


@nki.jit
def matmul_NKM(lhsT, rhs, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N):
    (
        M,
        N,
        K,
        TILE_M,
        TILE_N,
        TILE_K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_BLOCK_M,
        NUM_BLOCK_N,
        NUM_BLOCK_K,
    ) = common_head(lhsT, rhs, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N)

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    # Blocking N dimension (the RHS free dimension)
    for block_id_N in nl.affine_range(NUM_BLOCK_N):
        # M * BLOCK_N
        result_tiles = nl.zeros(
            (
                NUM_BLOCK_M,
                TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_N,
                nl.par_dim(TILE_M),
                TILE_N,
            ),
            dtype=lhsT.dtype,
            buffer=nl.sbuf,
        )

        # Blocking K dimension (the contraction dimension)
        # Use `sequential_range` because we do not want the compiler to change this loop by,
        # for example, vectorizing it
        for block_id_K in nl.sequential_range(NUM_BLOCK_K):
            # Loading tiles from rhs
            # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
            i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
            rhs_tiles = nl.ndarray(
                (TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                dtype=rhs.dtype,
                buffer=nl.sbuf,
            )

            for tile_id_K in nl.affine_range(TILES_IN_BLOCK_K):
                rhs_tiles[tile_id_K, i_rhs.p, i_rhs.x] = nl.load(
                    rhs[
                        (TILES_IN_BLOCK_K * block_id_K + tile_id_K) * TILE_K + i_rhs.p,
                        BLOCK_N * block_id_N + i_rhs.x,
                    ]
                )

            # Blocking M dimension (the LHS free dimension)
            for block_id_M in nl.affine_range(NUM_BLOCK_M):
                # Loading tiles from lhsT
                i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
                # lhsT = (BLOCK_K, BLOCK_M)
                lhsT_tiles = nl.ndarray(
                    (TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                    dtype=lhsT.dtype,
                    buffer=nl.sbuf,
                )
                for tile_id_K in nl.affine_range(TILES_IN_BLOCK_K):
                    lhsT_tiles[tile_id_K, i_lhsT.p, i_lhsT.x] = nl.load(
                        lhsT[
                            (TILES_IN_BLOCK_K * block_id_K + tile_id_K) * TILE_K
                            + i_lhsT.p,
                            BLOCK_M * block_id_M + i_lhsT.x,
                        ]
                    )

                    """
                    (TILES_IN_BLOCK_K * block_id_K + tile_id_K) * TILE_K
                    k_start = block_id_K * BLOCK_K + tile_id_K * TILE_K
                    BLOCK_M * block_id_M
                    m_start = block_id_M * BLOCK_M
                    n_start = block_id_N * BLOCK_N

                    lhsT_tiles[tile_id_K] = nl.load(
                        lhsT[k_start + i_lhsT.p, m_start + i_lhsT.x]
                    )
                    """

                # Do matmul with all tiles in the blocks
                i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
                i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
                i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
                for tile_id_N in nl.affine_range(TILES_IN_BLOCK_N):
                    for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
                        res_tile = nl.zeros(
                            (TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum
                        )

                        for tile_id_K in nl.affine_range(TILES_IN_BLOCK_K):
                            res_tile += nisa.nc_matmul(
                                lhsT_tiles[
                                    tile_id_K,
                                    i_lhsT_mm.p,
                                    tile_id_M * TILE_M + i_lhsT_mm.x,
                                ],
                                rhs_tiles[
                                    tile_id_K,
                                    i_rhs_mm.p,
                                    tile_id_N * TILE_N + i_rhs_mm.x,
                                ],
                            )

                        # Accumulate on corresponding SBUF tile
                        result_tiles[
                            block_id_M, tile_id_M, tile_id_N, i_res_mm.p, i_res_mm.x
                        ] += res_tile[i_res_mm.p, i_res_mm.x]

        # Copying the result from SBUF to HBM
        for block_id_M in nl.affine_range(NUM_BLOCK_M):
            for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
                i_res = nl.mgrid[0:TILE_K, 0:TILE_N]
                i_res_packed = nl.mgrid[0:TILE_K, 0:BLOCK_N]
                result_packed = nl.ndarray(
                    (TILE_K, BLOCK_N), dtype=result_tiles.dtype, buffer=nl.sbuf
                )

                # coalesce result tiles for better DMA performance
                for tile_id_N in nl.affine_range(TILES_IN_BLOCK_N):
                    result_packed[i_res.p, tile_id_N * TILE_N + i_res.x] = nl.copy(
                        result_tiles[block_id_M, tile_id_M, tile_id_N, i_res.p, i_res.x]
                    )
                nl.store(
                    result[
                        (TILES_IN_BLOCK_M * block_id_M + tile_id_M) * TILE_K
                        + i_res_packed.p,
                        BLOCK_N * block_id_N + i_res_packed.x,
                    ],
                    value=result_packed[i_res_packed.p, i_res_packed.x],
                )
    return result
