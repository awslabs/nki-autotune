# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.compiler.backends.neuron.tensors import KernelHBMTensor
from neuronxcc.nki.typing import tensor

from autotune.modules.dma import load_tensor_block, save_result_acc, save_result_block, save_result_dma
from autotune.modules.matmul import GEMMCompatibility, matmul_block


@nki.jit
def lhsT_rhs_gemm(lhsT: tensor, rhs: tensor, NUM_BLOCK_M: int, NUM_BLOCK_N: int, NUM_BLOCK_K: int, loop_order: str):
    mm = GEMMCompatibility(transposed_lhs=True)
    mm((lhsT, rhs), {"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "NUM_BLOCK_K": NUM_BLOCK_K})
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


def matmul_NMK(lhsT: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):
    for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
        for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
            # Does it matter if we fold into free without block dimensions?
            # Possible workaround for without blocking dimensions: list of tensors
            result_block = nl.zeros(
                (mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, nl.par_dim(mm.TILE_M), mm.TILE_N),
                dtype=result.dtype,
                buffer=nl.sbuf,
            )
            # Hoist or not
            for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
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
                matmul_block(lhsT_tiles, rhs_tiles, result_block)

            save_result_block(result, result_block, m_ofs=block_id_M * mm.BLOCK_M, n_ofs=block_id_N * mm.BLOCK_N)
    return result


def matmul_MNK(lhsT: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):
    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
            result_tiles = nl.zeros(
                (mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, nl.par_dim(mm.TILE_M), mm.TILE_N),
                dtype=result.dtype,
                buffer=nl.sbuf,
            )

            for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
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


def matmul_KMN(lhsT: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):

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

    for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
        for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
            # TILES_IN_BLOCK_K, TILE_K, BLOCK_M
            lhsT_tiles = load_tensor_block(
                input_tensor=lhsT,
                ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
            )
            for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_N
                rhs_tiles = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
                )

                matmul_block(lhsT_tiles, rhs_tiles, result_tiles[block_id_K, block_id_M, block_id_N])

    save_result_acc(result, result_tiles, mm.BLOCK_M, mm.BLOCK_N)
    return result


def matmul_KNM(lhsT: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):

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

    for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
            # TILES_IN_BLOCK_K, TILE_K, BLOCK_N
            rhs_tiles = load_tensor_block(
                input_tensor=rhs,
                ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
            )
            for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_M
                lhsT_tiles = load_tensor_block(
                    input_tensor=lhsT,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
                )

                matmul_block(lhsT_tiles, rhs_tiles, result_tiles[block_id_K, block_id_M, block_id_N])

    save_result_acc(result, result_tiles, mm.BLOCK_M, mm.BLOCK_N)
    return result


def matmul_NKM(lhsT: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):

    # Blocking N dimension (the RHS free dimension)
    for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
        result_tiles = nl.zeros(
            (mm.NUM_BLOCK_M, mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, nl.par_dim(mm.TILE_M), mm.TILE_N),
            dtype=result.dtype,
            buffer=nl.sbuf,
        )

        # Blocking K dimension (the contraction dimension)
        # Use `sequential_range` because we do not want the compiler to change this loop by,
        # for example, vectorizing it
        for block_id_K in nl.sequential_range(mm.NUM_BLOCK_K):
            # Loading tiles from rhs
            # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
            rhs_tiles = load_tensor_block(
                input_tensor=rhs,
                ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
            )

            # Blocking M dimension (the LHS free dimension)
            for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
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


def matmul_MKN(lhsT: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):
    # Blocking M dimension (the LHS free dimension)
    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
        result_tiles = nl.zeros(
            (mm.NUM_BLOCK_N, mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, nl.par_dim(mm.TILE_M), mm.TILE_N),
            dtype=result.dtype,
            buffer=nl.sbuf,
        )

        # Blocking K dimension (the contraction dimension)
        # Use `sequential_range` because we do not want the compiler to change this loop by,
        # for example, vectorizing it
        for block_id_K in nl.sequential_range(mm.NUM_BLOCK_K):
            # Loading tiles from lhsT
            lhsT_tiles = load_tensor_block(
                input_tensor=lhsT,
                ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
            )

            # Blocking N dimension (the RHS free dimension)
            for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
                # Loading tiles from rhs
                # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
                rhs_tiles = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
                )

                matmul_block(lhsT_tiles, rhs_tiles, result_tiles[block_id_N])

        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
            save_result_dma(
                result,
                result_tiles,
                block_id_N,
                m_ofs=block_id_M * mm.TILES_IN_BLOCK_M,
                n_ofs=block_id_N * mm.BLOCK_N,
                TILE_K=mm.TILE_K,
            )

    return result
