# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.compiler.backends.neuron.tensors import KernelHBMTensor
from neuronxcc.nki.typing import tensor

from autotune.modules.dma import load_tensor_block, save_result_block
from autotune.modules.matmul import GEMMCompatibility, matmul_blocks_lhsT


@nki.jit
def lhsT_rhs_gemm(lhsT: tensor, rhs: tensor, NUM_BLOCK_M: int, NUM_BLOCK_N: int, NUM_BLOCK_K: int, loop_order: str):
    mm = GEMMCompatibility(transposed_lhs=True)
    mm(
        input_tensors=(lhsT, rhs),
        kernel_kwargs={"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "NUM_BLOCK_K": NUM_BLOCK_K},
    )
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
            result_block = nl.zeros(
                (nl.par_dim(mm.TILE_M), mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                dtype=result.dtype,
                buffer=nl.sbuf,
            )
            for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
                lhsT_block = load_tensor_block(
                    input_tensor=lhsT,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                    load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_M, mm.TILE_M),
                )
                rhs_block = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                )
                matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=(0, 0))
            save_result_block(
                result,
                result_block,
                tile_index_ofs=(block_id_M * mm.TILES_IN_BLOCK_M, block_id_N * mm.TILES_IN_BLOCK_N),
            )
    return result


def matmul_MNK(lhsT: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):
    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
            result_block = nl.zeros(
                (nl.par_dim(mm.TILE_M), mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                dtype=result.dtype,
                buffer=nl.sbuf,
            )
            for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
                lhsT_block = load_tensor_block(
                    input_tensor=lhsT,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                    load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_M, mm.TILE_M),
                )
                rhs_block = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                )
                matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=(0, 0))
            save_result_block(
                result,
                result_block,
                tile_index_ofs=(block_id_M * mm.TILES_IN_BLOCK_M, block_id_N * mm.TILES_IN_BLOCK_N),
            )
    return result


def matmul_KMN(lhsT: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):
    result_block = nl.zeros(
        (nl.par_dim(mm.TILE_M), mm.TILES_IN_M, mm.TILES_IN_N, mm.TILE_N), dtype=result.dtype, buffer=nl.sbuf
    )
    for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
        for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
            lhsT_block = load_tensor_block(
                input_tensor=lhsT,
                ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_M, mm.TILE_M),
            )
            for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
                rhs_block = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                )
                matmul_blocks_lhsT(
                    lhsT_block, rhs_block, result_block, ofs=(block_id_M * mm.BLOCK_M, block_id_N * mm.BLOCK_N)
                )
    save_result_block(result, result_block, tile_index_ofs=(0, 0))
    return result


def matmul_KNM(lhsT: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):
    result_block = nl.zeros(
        (nl.par_dim(mm.TILE_M), mm.TILES_IN_M, mm.TILES_IN_N, mm.TILE_N), dtype=result.dtype, buffer=nl.sbuf
    )
    for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
            rhs_block = load_tensor_block(
                input_tensor=rhs,
                ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_N, mm.TILE_N),
            )
            for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
                lhsT_block = load_tensor_block(
                    input_tensor=lhsT,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                    load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_M, mm.TILE_M),
                )
                matmul_blocks_lhsT(
                    lhsT_block, rhs_block, result_block, ofs=(block_id_M * mm.BLOCK_M, block_id_N * mm.BLOCK_N)
                )
    save_result_block(result, result_block, tile_index_ofs=(0, 0))
    return result


def matmul_NKM(lhsT: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):
    for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
        result_block = nl.zeros(
            (nl.par_dim(mm.TILE_M), mm.TILES_IN_M, mm.TILES_IN_BLOCK_N, mm.TILE_N), dtype=result.dtype, buffer=nl.sbuf
        )
        for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
            rhs_block = load_tensor_block(
                input_tensor=rhs,
                ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_N, mm.TILE_N),
            )
            for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
                lhsT_block = load_tensor_block(
                    input_tensor=lhsT,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                    load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_M, mm.TILE_M),
                )
                matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=(block_id_M * mm.BLOCK_M, 0))
        save_result_block(result, result_block, tile_index_ofs=(0, block_id_N * mm.TILES_IN_BLOCK_N))
    return result


def matmul_MKN(lhsT: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):
    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
        result_block = nl.zeros(
            (nl.par_dim(mm.TILE_M), mm.TILES_IN_BLOCK_M, mm.TILES_IN_N, mm.TILE_N), dtype=result.dtype, buffer=nl.sbuf
        )
        for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
            lhsT_block = load_tensor_block(
                input_tensor=lhsT,
                ofs=(block_id_K * mm.BLOCK_K, block_id_M * mm.BLOCK_M),
                load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_M, mm.TILE_M),
            )
            for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
                rhs_block = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                )
                matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=(0, block_id_N * mm.BLOCK_N))
        save_result_block(result, result_block, tile_index_ofs=(block_id_M * mm.TILES_IN_BLOCK_M, 0))
    return result
