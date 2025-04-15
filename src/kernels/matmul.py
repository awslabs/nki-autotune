# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.compiler.backends.neuron.tensors import KernelHBMTensor
from neuronxcc.nki.typing import tensor

from src.kernels.utils import (
    MatMulCompatibility,
    load_tensor_block,
    matmul_block,
    matmul_blocks_lhs,
    matmul_blocks_tile_transposed_lhs,
    transpose_tiles_in_block,
)


@nki.jit
def matmul_main(
    lhsT: tensor,
    rhs: tensor,
    NUM_BLOCK_M: int,
    NUM_BLOCK_N: int,
    NUM_BLOCK_K: int,
    BUFFER_M: int,
    BUFFER_N: int,
    BUFFER_K: int,
    loop_order: str,
):
    mm = MatMulCompatibility(lhsT.shape, rhs.shape, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K)
    assert loop_order in ["MNK", "MKN", "NMK", "NKM", "KMN", "KNM"], f"Loop order {loop_order} GEMM does not exist."
    result = nl.ndarray((mm.M, mm.N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    if loop_order == "MNK":
        matmul_MNK(lhsT, rhs, mm, result)
    elif loop_order == "MKN":
        matmul_MKN(lhsT, rhs, mm, result)
    elif loop_order == "NMK":
        matmul_NMK(lhsT, rhs, mm, result)
    elif loop_order == "NKM":
        matmul_NKM(lhsT, rhs, mm, result)
    elif loop_order == "KMN":
        matmul_KMN(lhsT, rhs, mm, result)
    elif loop_order == "KNM":
        matmul_KNM(lhsT, rhs, mm, result)
    else:
        raise NotImplementedError(f"Loop order {loop_order} GEMM does not exist.")
    return result


def save_result_block(result, result_block, m_ofs: int, n_ofs: int):
    """
    Store result_block into result
    Args:
    result: M, N
    result_block: TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, nl.par_dim(TILE_M), TILE_N
    """
    TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILE_M, TILE_N = result_block.shape

    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for tile_id_M in nl.affine_range(TILES_IN_BLOCK_M):
        m_start = m_ofs + tile_id_M * TILE_M
        for tile_id_N in nl.affine_range(TILES_IN_BLOCK_N):
            n_start = n_ofs + tile_id_N * TILE_N
            nl.store(result[m_start + idx_res.p, n_start + idx_res.x], value=result_block[tile_id_M, tile_id_N])


def save_result_acc(result, result_tiles, BLOCK_M, BLOCK_N):
    NUM_BLOCK_K, NUM_BLOCK_M, NUM_BLOCK_N, num_m_tiles, num_n_tiles, TILE_M, TILE_N = result_tiles.shape

    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for block_id_M in nl.affine_range(NUM_BLOCK_M):
        m_ofs = block_id_M * BLOCK_M
        for block_id_N in nl.affine_range(NUM_BLOCK_N):
            n_ofs = block_id_N * BLOCK_N
            for tile_id_M in nl.affine_range(num_m_tiles):
                for tile_id_N in nl.affine_range(num_n_tiles):
                    result_acc = nl.zeros(
                        (num_m_tiles, num_n_tiles, nl.par_dim(TILE_M), TILE_N), dtype=result_tiles.dtype, buffer=nl.sbuf
                    )
                    for block_id_K in nl.affine_range(NUM_BLOCK_K):
                        result_acc[tile_id_M, tile_id_N] += result_tiles[
                            block_id_K, block_id_M, block_id_N, tile_id_M, tile_id_N
                        ]

                    nl.store(
                        result[m_ofs + tile_id_M * TILE_M + idx_res.p, n_ofs + tile_id_N * TILE_N + idx_res.x],
                        value=result_acc[tile_id_M, tile_id_N],
                    )


def save_result_dma(result, result_tiles, block_id, m_ofs, n_ofs, TILE_K):
    _, num_m_tiles, num_n_tiles, _, TILE_N = result_tiles.shape

    for tile_id_M in nl.affine_range(num_m_tiles):
        idx_res = nl.mgrid[0:TILE_K, 0:TILE_N]
        idx_res_packed = nl.mgrid[0:TILE_K, 0 : TILE_N * num_n_tiles]

        result_packed = nl.ndarray((TILE_K, TILE_N * num_n_tiles), dtype=result_tiles.dtype, buffer=nl.sbuf)

        # coalesce result tiles for better DMA performance
        for tile_id_N in nl.affine_range(num_n_tiles):
            result_packed[idx_res.p, tile_id_N * TILE_N + idx_res.x] = nl.copy(
                result_tiles[block_id, tile_id_M, tile_id_N, idx_res.p, idx_res.x]
            )

        nl.store(
            result[(m_ofs + tile_id_M) * TILE_K + idx_res_packed.p, n_ofs + idx_res_packed.x],
            value=result_packed[idx_res_packed.p, idx_res_packed.x],
        )


def matmul_NMK(lhsT: tensor, rhs: tensor, mm: MatMulCompatibility, result: KernelHBMTensor):
    for block_id_N in nl.affine_range(mm.NUM_BLOCK_N, multi_buffer=mm.BUFFER_N):
        for block_id_M in nl.affine_range(mm.NUM_BLOCK_M, multi_buffer=mm.BUFFER_M):
            result_tiles = nl.zeros(
                (mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, nl.par_dim(mm.TILE_M), mm.TILE_N),
                dtype=result.dtype,
                buffer=nl.sbuf,
            )

            for block_id_K in nl.affine_range(mm.NUM_BLOCK_K, multi_buffer=mm.BUFFER_K):
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_M
                lhsT_tiles = load_tensor_block(
                    input_tensor=lhsT,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
                )
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_N
                rhs_tiles = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
                )
                matmul_block(lhsT_tiles, rhs_tiles, result_tiles)

            save_result_block(result, result_tiles, m_ofs=block_id_M * mm.BLOCK_M, n_ofs=block_id_N * mm.BLOCK_N)
    return result


def matmul_MNK(lhsT: tensor, rhs: tensor, mm: MatMulCompatibility, result: KernelHBMTensor):
    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M, multi_buffer=mm.BUFFER_M):
        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N, multi_buffer=mm.BUFFER_N):
            result_tiles = nl.zeros(
                (mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, nl.par_dim(mm.TILE_M), mm.TILE_N),
                dtype=result.dtype,
                buffer=nl.sbuf,
            )

            for block_id_K in nl.affine_range(mm.NUM_BLOCK_K, multi_buffer=mm.BUFFER_K):
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_M
                lhsT_tiles = load_tensor_block(
                    input_tensor=lhsT,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
                )
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_N
                rhs_tiles = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
                )
                matmul_block(lhsT_tiles, rhs_tiles, result_tiles)

            save_result_block(result, result_tiles, m_ofs=block_id_M * mm.BLOCK_M, n_ofs=block_id_N * mm.BLOCK_N)

    return result


def matmul_KMN(lhsT: tensor, rhs: tensor, mm: MatMulCompatibility, result: KernelHBMTensor):

    result_tiles = nl.zeros(
        (
            mm.NUM_BLOCK_K,
            mm.NUM_BLOCK_M,
            mm.NUM_BLOCK_N,
            mm.TILES_IN_BLOCK_M,
            mm.TILES_IN_BLOCK_N,
            nl.par_dim(mm.TILE_M),
            mm.TILE_N,
        ),
        dtype=result.dtype,
        buffer=nl.sbuf,
    )

    for block_id_K in nl.affine_range(mm.NUM_BLOCK_K, multi_buffer=mm.BUFFER_K):
        for block_id_M in nl.affine_range(mm.NUM_BLOCK_M, multi_buffer=mm.BUFFER_M):
            # TILES_IN_BLOCK_K, TILE_K, BLOCK_M
            lhsT_tiles = load_tensor_block(
                input_tensor=lhsT,
                ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
            )
            for block_id_N in nl.affine_range(mm.NUM_BLOCK_N, multi_buffer=mm.BUFFER_N):
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_N
                rhs_tiles = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
                )

                matmul_block(lhsT_tiles, rhs_tiles, result_tiles[block_id_K, block_id_M, block_id_N])

    save_result_acc(result, result_tiles, mm.BLOCK_M, mm.BLOCK_N)
    return result


def matmul_KNM(lhsT: tensor, rhs: tensor, mm: MatMulCompatibility, result: KernelHBMTensor):

    result_tiles = nl.zeros(
        (
            mm.NUM_BLOCK_K,
            mm.NUM_BLOCK_M,
            mm.NUM_BLOCK_N,
            mm.TILES_IN_BLOCK_M,
            mm.TILES_IN_BLOCK_N,
            nl.par_dim(mm.TILE_M),
            mm.TILE_N,
        ),
        dtype=result.dtype,
        buffer=nl.sbuf,
    )

    for block_id_K in nl.affine_range(mm.NUM_BLOCK_K, multi_buffer=mm.BUFFER_K):
        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N, multi_buffer=mm.BUFFER_N):
            # TILES_IN_BLOCK_K, TILE_K, BLOCK_N
            rhs_tiles = load_tensor_block(
                input_tensor=rhs,
                ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
            )
            for block_id_M in nl.affine_range(mm.NUM_BLOCK_M, multi_buffer=mm.BUFFER_M):
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_M
                lhsT_tiles = load_tensor_block(
                    input_tensor=lhsT,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
                )

                matmul_block(lhsT_tiles, rhs_tiles, result_tiles[block_id_K, block_id_M, block_id_N])

    save_result_acc(result, result_tiles, mm.BLOCK_M, mm.BLOCK_N)
    return result


def matmul_NKM(lhsT: tensor, rhs: tensor, mm: MatMulCompatibility, result: KernelHBMTensor):

    # Blocking N dimension (the RHS free dimension)
    for block_id_N in nl.affine_range(mm.NUM_BLOCK_N, multi_buffer=mm.BUFFER_N):
        result_tiles = nl.zeros(
            (mm.NUM_BLOCK_M, mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, nl.par_dim(mm.TILE_M), mm.TILE_N),
            dtype=result.dtype,
            buffer=nl.sbuf,
        )

        # Blocking K dimension (the contraction dimension)
        # Use `sequential_range` because we do not want the compiler to change this loop by,
        # for example, vectorizing it
        for block_id_K in nl.sequential_range(mm.NUM_BLOCK_K, multi_buffer=mm.BUFFER_K):
            # Loading tiles from rhs
            # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
            rhs_tiles = load_tensor_block(
                input_tensor=rhs,
                ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
            )

            # Blocking M dimension (the LHS free dimension)
            for block_id_M in nl.affine_range(mm.NUM_BLOCK_M, multi_buffer=mm.BUFFER_M):
                # Loading tiles from lhsT
                lhsT_tiles = load_tensor_block(
                    input_tensor=lhsT,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
                )

                matmul_block(lhsT_tiles, rhs_tiles, result_tiles[block_id_M])

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
                m_ofs=block_id_M * mm.TILES_IN_BLOCK_M,
                n_ofs=block_id_N * mm.BLOCK_N,
                TILE_K=mm.TILE_K,
            )

    return result


def matmul_MKN(lhsT: tensor, rhs: tensor, mm: MatMulCompatibility, result: KernelHBMTensor):
    # Blocking M dimension (the LHS free dimension)
    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M, multi_buffer=mm.BUFFER_M):
        result_tiles = nl.zeros(
            (mm.NUM_BLOCK_N, mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, nl.par_dim(mm.TILE_M), mm.TILE_N),
            dtype=result.dtype,
            buffer=nl.sbuf,
        )

        # Blocking K dimension (the contraction dimension)
        # Use `sequential_range` because we do not want the compiler to change this loop by,
        # for example, vectorizing it
        for block_id_K in nl.sequential_range(mm.NUM_BLOCK_K, multi_buffer=mm.BUFFER_K):
            # Loading tiles from lhsT
            lhsT_tiles = load_tensor_block(
                input_tensor=lhsT,
                ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
            )

            # Blocking N dimension (the RHS free dimension)
            for block_id_N in nl.affine_range(mm.NUM_BLOCK_N, multi_buffer=mm.BUFFER_N):
                # Loading tiles from rhs
                # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
                rhs_tiles = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
                )

                matmul_block(lhsT_tiles, rhs_tiles, result_tiles[block_id_N])

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
                m_ofs=block_id_M * mm.TILES_IN_BLOCK_M,
                n_ofs=block_id_N * mm.BLOCK_N,
                TILE_K=mm.TILE_K,
            )

    return result


@nki.jit
def gemm_with_non_transposed_lhs_MNK(
    lhs: tensor,  # Shape (M, K)
    rhs: tensor,  # Shape (K, N)
    NUM_BLOCK_M: int,
    NUM_BLOCK_N: int,
    NUM_BLOCK_K: int,
    BUFFER_M: int,
    BUFFER_N: int,
    BUFFER_K: int,
):
    mm = MatMulCompatibility(
        lhs.shape[::-1], rhs.shape, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K
    )
    result = nl.ndarray((mm.M, mm.N), dtype=lhs.dtype, buffer=nl.shared_hbm)
    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
            result_block = nl.zeros(
                (mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, nl.par_dim(mm.TILE_M), mm.TILE_N),
                dtype=lhs.dtype,
                buffer=nl.sbuf,
            )
            for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
                lhsT_block = load_tensor_block(
                    input_tensor=lhs,
                    ofs=(block_id_M * mm.BLOCK_M, block_id_K * mm.BLOCK_K),
                    load_shape=(mm.TILES_IN_BLOCK_M, mm.TILE_M, mm.BLOCK_K),
                )
                # transpose_tiles_in_block(lhsT_block)
                rhs_block = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
                )
                # matmul_blocks_tile_transposed_lhs(lhsT_block, rhs_block, result_block)
                matmul_blocks_lhs(lhsT_block, rhs_block, result_block)
            save_result_block(result, result_block, m_ofs=block_id_M * mm.BLOCK_M, n_ofs=block_id_N * mm.BLOCK_N)
    return result


@nki.jit
def gemm_with_non_transposed_lhs_MN(
    lhs: tensor,  # Shape (M, K)
    rhs: tensor,  # Shape (K, N)
    NUM_BLOCK_M: int,
    NUM_BLOCK_N: int,
    NUM_BLOCK_K: int,
    BUFFER_M: int,
    BUFFER_N: int,
    BUFFER_K: int,
):
    mm = MatMulCompatibility(
        lhs.shape[::-1], rhs.shape, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K
    )
    result = nl.ndarray((mm.M, mm.N), dtype=lhs.dtype, buffer=nl.shared_hbm)
    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
        lhs_block = load_tensor_block(
            input_tensor=lhs, ofs=(block_id_M * mm.BLOCK_M, 0), load_shape=(mm.TILES_IN_BLOCK_M, mm.TILE_M, mm.K)
        )
        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
            result_block = nl.zeros(
                (mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, nl.par_dim(mm.TILE_M), mm.TILE_N),
                dtype=lhs.dtype,
                buffer=nl.sbuf,
            )
            rhs_block = load_tensor_block(
                input_tensor=rhs, ofs=(0, block_id_N * mm.BLOCK_N), load_shape=(mm.TILES_IN_K, mm.TILE_K, mm.BLOCK_N)
            )
            matmul_blocks_lhs(lhs_block, rhs_block, result_block)
            save_result_block(result, result_block, m_ofs=block_id_M * mm.BLOCK_M, n_ofs=block_id_N * mm.BLOCK_N)
    return result


@nki.jit
def baseline(
    lhsT,
    rhs,
    # Meta-parameters
    TILES_IN_BLOCK_M=16,
    TILES_IN_BLOCK_N=2,
    TILES_IN_BLOCK_K=8,
):
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
    Returns:
        result: the resulting output tensor of shape [M,N]
    """

    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N
    BLOCK_K = TILE_K * TILES_IN_BLOCK_K

    # the size has to be multiple of block size
    assert M % BLOCK_M == 0, f"M {M} is not multiples of BLOCK_M {BLOCK_M}"
    assert N % BLOCK_N == 0, f"N {N} is not multiples of BLOCK_N {BLOCK_N}"
    assert K % BLOCK_K == 0, f"K {K} is not multiples of BLOCK_K {BLOCK_K}"

    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N
    NUM_BLOCK_K = K // BLOCK_K

    # Blocking N dimension (the RHS free dimension)
    for n in nl.affine_range(NUM_BLOCK_N):
        result_tiles = nl.zeros(
            (NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, nl.par_dim(TILE_M), TILE_N),
            dtype=lhsT.dtype,
            buffer=nl.sbuf,
        )

        # Blocking K dimension (the contraction dimension)
        # Use `sequential_range` because we do not want the compiler to change this loop by,
        # for example, vectorizing it
        for k in nl.sequential_range(NUM_BLOCK_K):
            # Loading tiles from rhs
            # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
            i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
            rhs_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N), dtype=rhs.dtype, buffer=nl.sbuf)

            for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
                rhs_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
                    rhs[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p, BLOCK_N * n + i_rhs.x]
                )

            # Blocking M dimension (the LHS free dimension)
            for m in nl.affine_range(NUM_BLOCK_M):
                # Loading tiles from lhsT
                i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
                lhsT_tiles = nl.ndarray(
                    (TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M), dtype=lhsT.dtype, buffer=nl.sbuf
                )
                for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
                    lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
                        lhsT[(TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i_lhsT.p, BLOCK_M * m + i_lhsT.x]
                    )

                # Do matmul with all tiles in the blocks
                i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
                i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
                i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    for bm in nl.affine_range(TILES_IN_BLOCK_M):
                        res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                        for bk in nl.affine_range(TILES_IN_BLOCK_K):
                            res_tile[...] += nisa.nc_matmul(
                                lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                                rhs_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x],
                            )

                        # Accumulate on corresponding SBUF tile
                        result_tiles[m, bm, bn, i_res_mm.p, i_res_mm.x] += res_tile[i_res_mm.p, i_res_mm.x]

        # Copying the result from SBUF to HBM
        for m in nl.affine_range(NUM_BLOCK_M):
            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                i_res = nl.mgrid[0:TILE_K, 0:TILE_N]
                i_res_packed = nl.mgrid[0:TILE_K, 0:BLOCK_N]
                result_packed = nl.ndarray((TILE_K, BLOCK_N), dtype=result_tiles.dtype, buffer=nl.sbuf)

                # coalesce result tiles for better DMA performance
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    result_packed[i_res.p, bn * TILE_N + i_res.x] = nl.copy(result_tiles[m, bm, bn, i_res.p, i_res.x])
                nl.store(
                    result[(TILES_IN_BLOCK_M * m + bm) * TILE_K + i_res_packed.p, BLOCK_N * n + i_res_packed.x],
                    value=result_packed[i_res_packed.p, i_res_packed.x],
                )

    return result
