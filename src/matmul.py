# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki as nki
import numpy as np


class MatMul:
    def __init__(
        self,
        lhsT_shape,
        rhs_shape,
        TILES_IN_BLOCK_K,
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
    ):
        # Input sizes
        self.K, self.M = lhsT_shape
        self.K_, self.N = rhs_shape

        # Tile sizes
        self.TILE_K = nl.tile_size.pmax  # 128
        self.TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
        self.TILE_N = nl.tile_size.gemm_moving_fmax  # 512

        # Block sizes
        self.BLOCK_K = self.TILE_K * TILES_IN_BLOCK_K
        self.BLOCK_M = self.TILE_M * TILES_IN_BLOCK_M
        self.BLOCK_N = self.TILE_N * TILES_IN_BLOCK_N

        # Number of blocks
        self.NUM_BLOCK_K = self.K // self.BLOCK_K
        self.NUM_BLOCK_M = self.M // self.BLOCK_M
        self.NUM_BLOCK_N = self.N // self.BLOCK_N

        # Data checks
        assert (
            self.K == self.K_
        ), f"lhsT and rhs contraction dimension mismatch, got {lhsT_shape} and {rhs_shape}"
        assert self.NUM_BLOCK_K * TILES_IN_BLOCK_K * self.TILE_K == self.K
        assert self.NUM_BLOCK_M * TILES_IN_BLOCK_M * self.TILE_M == self.M
        assert self.NUM_BLOCK_N * TILES_IN_BLOCK_N * self.TILE_N == self.N


def load_tensor_by_par_tiles(
    input_tensor, num_par_tiles, par_tile_size, free_dim_size, par_ofs, free_ofs
):
    """
    Load a rectangular area of shape (num_par_tiles * par_tile_size, free_dim_size) from the input tensor.
    The location of the rectangle from the input is offset by (par_ofs, free_ofs).
    Load the input tile by tile in parallel in the par dimension.

    Args:
        input_tensor: the input tensor to load from
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
        (num_par_tiles, nl.par_dim(par_tile_size), free_dim_size),
        dtype=input_tensor.dtype,
        buffer=nl.sbuf,
    )

    idx = nl.mgrid[0:par_tile_size, 0:free_dim_size]
    for par_tile_id in nl.affine_range(num_par_tiles):
        tiles[par_tile_id, idx.p, idx.x] = nl.load(
            input_tensor[
                par_ofs + par_tile_id * par_tile_size + idx.p, free_ofs + idx.x
            ]
        )
    return tiles


def matmul_tiles(lhsT_tiles, rhs_tiles, result_tiles, result_dtype):
    num_k_tiles, TILE_K, m = lhsT_tiles.shape
    _num_k_tiles, _TILE_K, n = rhs_tiles.shape
    num_m_tiles, num_n_tiles, TILE_M, TILE_N = result_tiles.shape
    _n = num_n_tiles * TILE_N

    # Data checks
    assert (
        num_k_tiles == _num_k_tiles and TILE_K == _TILE_K
    ), f"lhsT_tiles {lhsT_tiles.shape} does not match with rhs_tiles {rhs_tiles.shape}"
    assert (
        m == num_m_tiles * TILE_M
    ), f"lhsT_tiles {lhsT_tiles.shape} does not match with result_tiles {result_tiles.shape}"
    assert (
        n == _n
    ), f"rhs_tiles {rhs_tiles.shape} does not match with result_tiles {result_tiles.shape}"

    idx_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
    idx_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for tile_id_M in nl.affine_range(num_m_tiles):
        for tile_id_N in nl.affine_range(num_n_tiles):
            res_tile = nl.zeros((TILE_M, TILE_N), dtype=result_dtype, buffer=nl.psum)

            # Use PSUM buffer to accumulate into a single hardware tile
            # to minimize the number of calls to nl.loop_reduce
            for tile_id_K in nl.affine_range(num_k_tiles):
                res_tile += nisa.nc_matmul(
                    lhsT_tiles[tile_id_K, idx_lhsT.p, tile_id_M * TILE_M + idx_lhsT.x],
                    rhs_tiles[tile_id_K, idx_rhs.p, tile_id_N * TILE_N + idx_rhs.x],
                )

            result_tiles[tile_id_M, tile_id_N, idx_res.p, idx_res.x] += res_tile[
                idx_res.p, idx_res.x
            ]


def save_result_block(result, result_tiles, m_ofs, n_ofs):
    num_m_tiles, num_n_tiles, TILE_M, TILE_N = result_tiles.shape

    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for tile_id_M in nl.affine_range(num_m_tiles):
        for tile_id_N in nl.affine_range(num_n_tiles):
            nl.store(
                result[
                    m_ofs + tile_id_M * TILE_M + idx_res.p,
                    n_ofs + tile_id_N * TILE_N + idx_res.x,
                ],
                value=result_tiles[tile_id_M, tile_id_N],
            )

            # coalesce result tiles for better DMA performance
            for tile_id_N in nl.affine_range(num_n_tiles):
                result_packed[idx_res.p, tile_id_N * TILE_N + idx_res.x] = nl.copy(
                    result_tiles[block_id_M, tile_id_M, tile_id_N, idx_res.p, idx_res.x]
                )

def save_result_acc(result, result_tiles, BLOCK_M, BLOCK_N):
    NUM_BLOCK_K, NUM_BLOCK_M, NUM_BLOCK_N, num_m_tiles, num_n_tiles, TILE_M, TILE_N = (
        result_tiles.shape
    )

    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for block_id_M in nl.affine_range(NUM_BLOCK_M):
        m_ofs = block_id_M * BLOCK_M
        for block_id_N in nl.affine_range(NUM_BLOCK_N):
            n_ofs = block_id_N * BLOCK_N
            for tile_id_M in nl.affine_range(num_m_tiles):
                for tile_id_N in nl.affine_range(num_n_tiles):
                    result_acc = nl.zeros(
                        (num_m_tiles, num_n_tiles, nl.par_dim(TILE_M), TILE_N),
                        dtype=result_tiles.dtype,
                        buffer=nl.sbuf,
                    )
                    for block_id_K in nl.affine_range(NUM_BLOCK_K):
                        result_acc[tile_id_M, tile_id_N] += result_tiles[
                            block_id_K, block_id_M, block_id_N, tile_id_M, tile_id_N
                        ]

                    nl.store(
                        result[
                            m_ofs + tile_id_M * TILE_M + idx_res.p,
                            n_ofs + tile_id_N * TILE_N + idx_res.x,
                        ],
                        value=result_acc[tile_id_M, tile_id_N],
                    )


def save_result_dma(result, result_tiles, block_id, m_ofs, n_ofs, TILE_K):
    _, num_m_tiles, num_n_tiles, _, TILE_N = result_tiles.shape

    for tile_id_M in nl.affine_range(num_m_tiles):
        idx_res = nl.mgrid[0:TILE_K, 0:TILE_N]
        idx_res_packed = nl.mgrid[0:TILE_K, 0 : TILE_N * num_n_tiles]

        result_packed = nl.ndarray(
            (TILE_K, TILE_N * num_n_tiles), dtype=result_tiles.dtype, buffer=nl.sbuf
        )

        # coalesce result tiles for better DMA performance
        for tile_id_N in nl.affine_range(num_n_tiles):
            result_packed[idx_res.p, tile_id_N * TILE_N + idx_res.x] = nl.copy(
                result_tiles[block_id, tile_id_M, tile_id_N, idx_res.p, idx_res.x]
            )

        nl.store(
            result[
                (m_ofs + tile_id_M) * TILE_K + idx_res_packed.p,
                n_ofs + idx_res_packed.x,
            ],
            value=result_packed[idx_res_packed.p, idx_res_packed.x],
        )


@nki.jit
def matmul_NMK(lhsT, rhs, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N):
    mm = MatMul(
        lhsT.shape, rhs.shape, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N
    )

    result = nl.ndarray((mm.M, mm.N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
        for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
            result_tiles = nl.zeros(
                (
                    TILES_IN_BLOCK_M,
                    TILES_IN_BLOCK_N,
                    nl.par_dim(mm.TILE_M),
                    mm.TILE_N,
                ),
                dtype=result.dtype,
                buffer=nl.sbuf,
            )

            for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_M
                lhsT_tiles = load_tensor_by_par_tiles(
                    input_tensor=lhsT,
                    num_par_tiles=TILES_IN_BLOCK_K,
                    par_tile_size=mm.TILE_K,
                    free_dim_size=mm.BLOCK_M,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_M * mm.BLOCK_M,
                )
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_N
                rhs_tiles = load_tensor_by_par_tiles(
                    input_tensor=rhs,
                    num_par_tiles=TILES_IN_BLOCK_K,
                    par_tile_size=mm.TILE_K,
                    free_dim_size=mm.BLOCK_N,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_N * mm.BLOCK_N,
                )
                matmul_tiles(lhsT_tiles, rhs_tiles, result_tiles, result.dtype)

            save_result_block(
                result,
                result_tiles,
                m_ofs=block_id_M * mm.BLOCK_M,
                n_ofs=block_id_N * mm.BLOCK_N,
            )
    return result


@nki.jit
def matmul_MNK(lhsT, rhs, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N):
    mm = MatMul(
        lhsT.shape, rhs.shape, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N
    )

    result = nl.ndarray((mm.M, mm.N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
            result_tiles = nl.zeros(
                (
                    TILES_IN_BLOCK_M,
                    TILES_IN_BLOCK_N,
                    nl.par_dim(mm.TILE_M),
                    mm.TILE_N,
                ),
                dtype=result.dtype,
                buffer=nl.sbuf,
            )

            for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_M
                lhsT_tiles = load_tensor_by_par_tiles(
                    input_tensor=lhsT,
                    num_par_tiles=TILES_IN_BLOCK_K,
                    par_tile_size=mm.TILE_K,
                    free_dim_size=mm.BLOCK_M,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_M * mm.BLOCK_M,
                )
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_N
                rhs_tiles = load_tensor_by_par_tiles(
                    input_tensor=rhs,
                    num_par_tiles=TILES_IN_BLOCK_K,
                    par_tile_size=mm.TILE_K,
                    free_dim_size=mm.BLOCK_N,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_N * mm.BLOCK_N,
                )
                matmul_tiles(lhsT_tiles, rhs_tiles, result_tiles, result.dtype)

            save_result_block(
                result,
                result_tiles,
                m_ofs=block_id_M * mm.BLOCK_M,
                n_ofs=block_id_N * mm.BLOCK_N,
            )

    return result


@nki.jit
def matmul_KMN(lhsT, rhs, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N):
    mm = MatMul(
        lhsT.shape, rhs.shape, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N
    )

    result = nl.ndarray((mm.M, mm.N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    result_tiles = nl.zeros(
        (
            mm.NUM_BLOCK_K,
            mm.NUM_BLOCK_M,
            mm.NUM_BLOCK_N,
            TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_N,
            nl.par_dim(mm.TILE_M),
            mm.TILE_N,
        ),
        dtype=result.dtype,
        buffer=nl.sbuf,
    )

    for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
        for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
            # TILES_IN_BLOCK_K, TILE_K, BLOCK_M
            lhsT_tiles = load_tensor_by_par_tiles(
                input_tensor=lhsT,
                num_par_tiles=TILES_IN_BLOCK_K,
                par_tile_size=mm.TILE_K,
                free_dim_size=mm.BLOCK_M,
                par_ofs=block_id_K * mm.BLOCK_K,
                free_ofs=block_id_M * mm.BLOCK_M,
            )
            for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_N
                rhs_tiles = load_tensor_by_par_tiles(
                    input_tensor=rhs,
                    num_par_tiles=TILES_IN_BLOCK_K,
                    par_tile_size=mm.TILE_K,
                    free_dim_size=mm.BLOCK_N,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_N * mm.BLOCK_N,
                )

                matmul_tiles(
                    lhsT_tiles,
                    rhs_tiles,
                    result_tiles[block_id_K, block_id_M, block_id_N],
                    result.dtype,
                )

    save_result_acc(
        result,
        result_tiles,
        mm.BLOCK_M,
        mm.BLOCK_N,
    )
    return result


@nki.jit
def matmul_KNM(lhsT, rhs, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N):
    mm = MatMul(
        lhsT.shape, rhs.shape, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N
    )

    result = nl.ndarray((mm.M, mm.N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    result_tiles = nl.zeros(
        (
            mm.NUM_BLOCK_K,
            mm.NUM_BLOCK_M,
            mm.NUM_BLOCK_N,
            TILES_IN_BLOCK_M,
            TILES_IN_BLOCK_N,
            nl.par_dim(mm.TILE_M),
            mm.TILE_N,
        ),
        dtype=result.dtype,
        buffer=nl.sbuf,
    )

    for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
            # TILES_IN_BLOCK_K, TILE_K, BLOCK_N
            rhs_tiles = load_tensor_by_par_tiles(
                input_tensor=rhs,
                num_par_tiles=TILES_IN_BLOCK_K,
                par_tile_size=mm.TILE_K,
                free_dim_size=mm.BLOCK_N,
                par_ofs=block_id_K * mm.BLOCK_K,
                free_ofs=block_id_N * mm.BLOCK_N,
            )
            for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_M
                lhsT_tiles = load_tensor_by_par_tiles(
                    input_tensor=lhsT,
                    num_par_tiles=TILES_IN_BLOCK_K,
                    par_tile_size=mm.TILE_K,
                    free_dim_size=mm.BLOCK_M,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_M * mm.BLOCK_M,
                )

                matmul_tiles(
                    lhsT_tiles,
                    rhs_tiles,
                    result_tiles[block_id_K, block_id_M, block_id_N],
                    result.dtype,
                )

    save_result_acc(
        result,
        result_tiles,
        mm.BLOCK_M,
        mm.BLOCK_N,
    )
    return result


@nki.jit
def matmul_NKM(lhsT, rhs, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N):
    mm = MatMul(
        lhsT.shape, rhs.shape, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N
    )

    result = nl.ndarray((mm.M, mm.N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    # Blocking N dimension (the RHS free dimension)
    for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
        result_tiles = nl.zeros(
            (
                mm.NUM_BLOCK_M,
                TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_N,
                nl.par_dim(mm.TILE_M),
                mm.TILE_N,
            ),
            dtype=result.dtype,
            buffer=nl.sbuf,
        )

        # Blocking K dimension (the contraction dimension)
        # Use `sequential_range` because we do not want the compiler to change this loop by,
        # for example, vectorizing it
        for block_id_K in nl.sequential_range(mm.NUM_BLOCK_K):
            # Loading tiles from rhs
            # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
            rhs_tiles = load_tensor_by_par_tiles(
                input_tensor=rhs,
                num_par_tiles=TILES_IN_BLOCK_K,
                par_tile_size=mm.TILE_K,
                free_dim_size=mm.BLOCK_N,
                par_ofs=block_id_K * mm.BLOCK_K,
                free_ofs=block_id_N * mm.BLOCK_N,
            )

            # Blocking M dimension (the LHS free dimension)
            for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
                # Loading tiles from lhsT
                lhsT_tiles = load_tensor_by_par_tiles(
                    input_tensor=lhsT,
                    num_par_tiles=TILES_IN_BLOCK_K,
                    par_tile_size=mm.TILE_K,
                    free_dim_size=mm.BLOCK_M,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_M * mm.BLOCK_M,
                )

                matmul_tiles(
                    lhsT_tiles, rhs_tiles, result_tiles[block_id_M], result.dtype
                )

        for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
            """
            save_result_block(
                result,
                result_tiles[block_id_M],
                m_ofs=block_id_M * mm.BLOCK_M,
                n_ofs=block_id_N * mm.BLOCK_N,
            )
            """
            save_result_dma(
                result,
                result_tiles,
                block_id_M,
                m_ofs=block_id_M * TILES_IN_BLOCK_M,
                n_ofs=block_id_N * mm.BLOCK_N,
                TILE_K=mm.TILE_K,
            )

    return result


@nki.jit
def matmul_MKN(lhsT, rhs, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N):
    mm = MatMul(
        lhsT.shape, rhs.shape, TILES_IN_BLOCK_K, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N
    )

    result = nl.ndarray((mm.M, mm.N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    # Blocking M dimension (the LHS free dimension)
    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
        result_tiles = nl.zeros(
            (
                mm.NUM_BLOCK_N,
                TILES_IN_BLOCK_M,
                TILES_IN_BLOCK_N,
                nl.par_dim(mm.TILE_M),
                mm.TILE_N,
            ),
            dtype=result.dtype,
            buffer=nl.sbuf,
        )

        # Blocking K dimension (the contraction dimension)
        # Use `sequential_range` because we do not want the compiler to change this loop by,
        # for example, vectorizing it
        for block_id_K in nl.sequential_range(mm.NUM_BLOCK_K):
            # Loading tiles from lhsT
            lhsT_tiles = load_tensor_by_par_tiles(
                input_tensor=lhsT,
                num_par_tiles=TILES_IN_BLOCK_K,
                par_tile_size=mm.TILE_K,
                free_dim_size=mm.BLOCK_M,
                par_ofs=block_id_K * mm.BLOCK_K,
                free_ofs=block_id_M * mm.BLOCK_M,
            )

            # Blocking N dimension (the RHS free dimension)
            for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
                # Loading tiles from rhs
                # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
                rhs_tiles = load_tensor_by_par_tiles(
                    input_tensor=rhs,
                    num_par_tiles=TILES_IN_BLOCK_K,
                    par_tile_size=mm.TILE_K,
                    free_dim_size=mm.BLOCK_N,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_N * mm.BLOCK_N,
                )

                matmul_tiles(
                    lhsT_tiles, rhs_tiles, result_tiles[block_id_N], result.dtype
                )

        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
            """
            save_result_block(
                result,
                result_tiles[block_id_N],
                m_ofs=block_id_M * mm.BLOCK_M,
                n_ofs=block_id_N * mm.BLOCK_N,
            )
            """
            save_result_dma(
                result,
                result_tiles,
                block_id_N,
                m_ofs=block_id_M * TILES_IN_BLOCK_M,
                n_ofs=block_id_N * mm.BLOCK_N,
                TILE_K=mm.TILE_K,
            )

    return result
