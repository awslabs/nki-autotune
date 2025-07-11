import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.compiler.backends.neuron.tensors import KernelHBMTensor
from neuronxcc.nki.typing import tensor

from autotune.core.dma import load_tensor_block, save_result_block, save_result_dma
from autotune.core.layout import transpose_tiles_in_block
from autotune.core.utils import GEMMCompatibility, matmul_blocks_lhs, matmul_blocks_tile_transposed_lhs
from autotune.legacy import dma as legacy_dma
from autotune.legacy import layout as legacy_layout
from autotune.legacy import utils as legacy_utils


@nki.jit
def lhs_rhs_gemm(lhs: tensor, rhs: tensor, NUM_BLOCK_M: int, NUM_BLOCK_N: int, NUM_BLOCK_K: int, template: str):
    mm = GEMMCompatibility(transposed_lhs=False)
    mm((lhs, rhs), {"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "NUM_BLOCK_K": NUM_BLOCK_K})

    result = nl.ndarray((mm.M, mm.N), dtype=lhs.dtype, buffer=nl.shared_hbm)
    if template == "MNK":
        gemm_lhs_MNK(lhs, rhs, mm, result)
    elif template == "MN":
        gemm_lhs_MN(lhs, rhs, mm, result)
    elif template == "MKN":
        gemm_lhs_MKN(lhs, rhs, mm, result)
    elif template == "legacy_MKN":
        legacy_gemm_lhs_MKN(lhs, rhs, mm, result)
    else:
        raise NotImplementedError(f"template {template} not implemented")
    return result


def gemm_lhs_MNK(lhs: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):
    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
            result_block = nl.zeros(
                (nl.par_dim(mm.TILE_M), mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                dtype=lhs.dtype,
                buffer=nl.sbuf,
            )
            for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
                lhs_block = load_tensor_block(
                    input_tensor=lhs,
                    ofs=(block_id_M * mm.BLOCK_M, block_id_K * mm.BLOCK_K),
                    load_shape=(mm.TILE_M, mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_K, mm.TILE_K),
                )
                rhs_block = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                )
                matmul_blocks_lhs(lhs_block, rhs_block, result_block)
            save_result_block(result, result_block, m_ofs=block_id_M * mm.BLOCK_M, n_ofs=block_id_N * mm.BLOCK_N)
    return result


def gemm_lhs_MN(lhs: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):
    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
        # FIXME: fix the load shape
        lhs_block = load_tensor_block(
            input_tensor=lhs,
            ofs=(block_id_M * mm.BLOCK_M, 0),
            load_shape=(mm.TILE_M, mm.TILES_IN_BLOCK_M, mm.TILES_IN_K, mm.TILE_K),
        )
        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
            result_block = nl.zeros(
                (nl.par_dim(mm.TILE_M), mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                dtype=lhs.dtype,
                buffer=nl.sbuf,
            )
            rhs_block = load_tensor_block(
                input_tensor=rhs,
                ofs=(0, block_id_N * mm.BLOCK_N),
                load_shape=(mm.TILE_K, mm.TILES_IN_K, mm.TILES_IN_BLOCK_N, mm.TILE_N),
            )
            matmul_blocks_lhs(lhs_block, rhs_block, result_block)
            save_result_block(result, result_block, m_ofs=block_id_M * mm.BLOCK_M, n_ofs=block_id_N * mm.BLOCK_N)
    return result


def gemm_lhs_MKN(lhs: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):
    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
        # TODO: tune different free dimension layout
        result_blocks = nl.zeros(
            (nl.par_dim(mm.TILE_M), mm.NUM_BLOCK_N, mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, mm.TILE_N),
            dtype=result.dtype,
            buffer=nl.sbuf,
        )
        for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
            # TODO: tune different free dimension layout
            lhs_block = load_tensor_block(
                input_tensor=lhs,
                ofs=(block_id_M * mm.BLOCK_M, block_id_K * mm.BLOCK_K),
                load_shape=(mm.TILE_M, mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_K, mm.TILE_K),
            )
            transpose_tiles_in_block(lhs_block)
            for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
                rhs_block = load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILE_K, mm.TILES_IN_BLOCK_K, mm.TILES_IN_BLOCK_N, mm.TILE_N),
                )
                # TODO: handle different lhs, rhs layouts
                matmul_blocks_tile_transposed_lhs(lhs_block, rhs_block, result_blocks, block_id_N)
        save_result_dma(result, result_blocks, block_id_M)

    return result


def legacy_gemm_lhs_MKN(lhs: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):
    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
        result_blocks = nl.zeros(
            (mm.NUM_BLOCK_N, mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, nl.par_dim(mm.TILE_M), mm.TILE_N),
            dtype=result.dtype,
            buffer=nl.sbuf,
        )
        for block_id_K in nl.sequential_range(mm.NUM_BLOCK_K):
            lhs_block = legacy_dma.load_tensor_block(
                input_tensor=lhs,
                ofs=(block_id_M * mm.BLOCK_M, block_id_K * mm.BLOCK_K),
                load_shape=(mm.TILES_IN_BLOCK_M, mm.TILE_M, mm.BLOCK_K),
            )
            legacy_layout.transpose_tiles_in_block(lhs_block)
            for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
                rhs_block = legacy_dma.load_tensor_block(
                    input_tensor=rhs,
                    ofs=(block_id_K * mm.BLOCK_K, block_id_N * mm.BLOCK_N),
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
                )
                legacy_utils.matmul_blocks_tile_transposed_lhs(lhs_block, rhs_block, result_blocks[block_id_N])

        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):
            legacy_dma.save_result_dma(
                result,
                result_blocks,
                block_id_N,
                m_ofs=block_id_M * mm.TILES_IN_BLOCK_M,
                n_ofs=block_id_N * mm.BLOCK_N,
                TILE_K=mm.TILE_K,
            )

    return result
