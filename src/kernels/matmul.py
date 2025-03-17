# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki as nki
from neuronxcc.nki.compiler.backends.neuron.tensors import TensorRef, KernelHBMTensor
from typing import Tuple

from src.kernels.utils import load_tensor_block, matmul_block, matmul_non_transposed_blocks, MatMulCompatibility


@nki.jit
def matmul_main(
    lhsT: TensorRef,
    rhs: TensorRef,
    NUM_BLOCK_M: int,
    NUM_BLOCK_N: int,
    NUM_BLOCK_K: int,
    BUFFER_M: int,
    BUFFER_N: int,
    BUFFER_K: int,
    loop_order: str,
):
    mm = MatMulCompatibility(
        lhsT.shape[::-1], rhs.shape, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K
    )
    assert loop_order in ["MNK", "MKN", "NMK", "NKM", "KMN", "KNM"], f"Loop order {loop_order} GEMM does not exist."
    result = nl.ndarray((mm.M, mm.N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    if loop_order == "MNK":
        return matmul_MNK(lhsT, rhs, mm, result)
    elif loop_order == "MKN":
        return matmul_MKN(lhsT, rhs, mm, result)
    elif loop_order == "NMK":
        return matmul_NMK(lhsT, rhs, mm, result)
    elif loop_order == "NKM":
        return matmul_NKM(lhsT, rhs, mm, result)
    elif loop_order == "KMN":
        return matmul_KMN(lhsT, rhs, mm, result)
    elif loop_order == "KNM":
        return matmul_KNM(lhsT, rhs, mm, result)
    else:
        raise NotImplementedError(f"Loop order {loop_order} GEMM does not exist.")


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


def matmul_NMK(lhsT: TensorRef, rhs: TensorRef, mm: MatMulCompatibility, result: KernelHBMTensor):
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
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_M * mm.BLOCK_M,
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
                )
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_N
                rhs_tiles = load_tensor_block(
                    input_tensor=rhs,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_N * mm.BLOCK_N,
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
                )
                matmul_block(lhsT_tiles, rhs_tiles, result_tiles)

            save_result_block(result, result_tiles, m_ofs=block_id_M * mm.BLOCK_M, n_ofs=block_id_N * mm.BLOCK_N)
    return result


def matmul_MNK(lhsT: TensorRef, rhs: TensorRef, mm: MatMulCompatibility, result: KernelHBMTensor):
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
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_M * mm.BLOCK_M,
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
                )
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_N
                rhs_tiles = load_tensor_block(
                    input_tensor=rhs,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_N * mm.BLOCK_N,
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
                )
                matmul_block(lhsT_tiles, rhs_tiles, result_tiles)

            save_result_block(result, result_tiles, m_ofs=block_id_M * mm.BLOCK_M, n_ofs=block_id_N * mm.BLOCK_N)

    return result


def matmul_KMN(lhsT: TensorRef, rhs: TensorRef, mm: MatMulCompatibility, result: KernelHBMTensor):

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
                par_ofs=block_id_K * mm.BLOCK_K,
                free_ofs=block_id_M * mm.BLOCK_M,
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
            )
            for block_id_N in nl.affine_range(mm.NUM_BLOCK_N, multi_buffer=mm.BUFFER_N):
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_N
                rhs_tiles = load_tensor_block(
                    input_tensor=rhs,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_N * mm.BLOCK_N,
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
                )

                matmul_block(lhsT_tiles, rhs_tiles, result_tiles[block_id_K, block_id_M, block_id_N])

    save_result_acc(result, result_tiles, mm.BLOCK_M, mm.BLOCK_N)
    return result


def matmul_KNM(lhsT: TensorRef, rhs: TensorRef, mm: MatMulCompatibility, result: KernelHBMTensor):

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
                par_ofs=block_id_K * mm.BLOCK_K,
                free_ofs=block_id_N * mm.BLOCK_N,
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
            )
            for block_id_M in nl.affine_range(mm.NUM_BLOCK_M, multi_buffer=mm.BUFFER_M):
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_M
                lhsT_tiles = load_tensor_block(
                    input_tensor=lhsT,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_M * mm.BLOCK_M,
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
                )

                matmul_block(lhsT_tiles, rhs_tiles, result_tiles[block_id_K, block_id_M, block_id_N])

    save_result_acc(result, result_tiles, mm.BLOCK_M, mm.BLOCK_N)
    return result


def matmul_NKM(lhsT: TensorRef, rhs: TensorRef, mm: MatMulCompatibility, result: KernelHBMTensor):

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
                par_ofs=block_id_K * mm.BLOCK_K,
                free_ofs=block_id_N * mm.BLOCK_N,
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
            )

            # Blocking M dimension (the LHS free dimension)
            for block_id_M in nl.affine_range(mm.NUM_BLOCK_M, multi_buffer=mm.BUFFER_M):
                # Loading tiles from lhsT
                lhsT_tiles = load_tensor_block(
                    input_tensor=lhsT,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_M * mm.BLOCK_M,
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


def matmul_MKN(lhsT: TensorRef, rhs: TensorRef, mm: MatMulCompatibility, result: KernelHBMTensor):
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
                par_ofs=block_id_K * mm.BLOCK_K,
                free_ofs=block_id_M * mm.BLOCK_M,
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
            )

            # Blocking N dimension (the RHS free dimension)
            for block_id_N in nl.affine_range(mm.NUM_BLOCK_N, multi_buffer=mm.BUFFER_N):
                # Loading tiles from rhs
                # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
                rhs_tiles = load_tensor_block(
                    input_tensor=rhs,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_N * mm.BLOCK_N,
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
    lhs: TensorRef,  # Shape (M, K)
    rhs: TensorRef,  # Shape (K, N)
    NUM_BLOCK_M: int,
    NUM_BLOCK_N: int,
    NUM_BLOCK_K: int,
    BUFFER_M: int,
    BUFFER_N: int,
    BUFFER_K: int,
):
    mm = MatMulCompatibility(lhs.shape, rhs.shape, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K)
    result = nl.ndarray((mm.M, mm.N), dtype=lhs.dtype, buffer=nl.shared_hbm)
    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
            result_block = nl.zeros(
                (mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, nl.par_dim(mm.TILE_M), mm.TILE_N),
                dtype=lhs.dtype,
                buffer=nl.sbuf,
            )
            for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
                lhs_block = load_tensor_block(
                    input_tensor=lhs,
                    ofs=(block_id_M * mm.BLOCK_M, block_id_K * mm.BLOCK_K),
                    load_shape=(mm.TILES_IN_BLOCK_M, mm.TILE_M, mm.BLOCK_K),
                )
                rhs_block = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
                )

                # Perform matrix multiplication
                matmul_non_transposed_blocks(lhs_block, rhs_block, result_block)

            # Store result
            save_result_block(result, result_block, m_ofs=block_id_M * mm.BLOCK_M, n_ofs=block_id_N * mm.BLOCK_N)

    return result


@nki.jit
def gemm_with_non_transposed_lhs_MN(
    lhs: TensorRef,  # Shape (M, K)
    rhs: TensorRef,  # Shape (K, N)
    NUM_BLOCK_M: int,
    NUM_BLOCK_N: int,
    NUM_BLOCK_K: int,
    BUFFER_M: int,
    BUFFER_N: int,
    BUFFER_K: int,
):
    mm = MatMulCompatibility(lhs.shape, rhs.shape, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K)
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

            # Perform matrix multiplication
            matmul_non_transposed_blocks(lhs_block, rhs_block, result_block)

            # Store result
            save_result_block(result, result_block, m_ofs=block_id_M * mm.BLOCK_M, n_ofs=block_id_N * mm.BLOCK_N)

    return result
