import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.compiler.backends.neuron.tensors import KernelHBMTensor
from neuronxcc.nki.typing import tensor

from autotune.modules.dma import load_tensor_block, save_result_block, save_result_dma
from autotune.modules.layout import transpose_tiles_in_block
from autotune.modules.matmul import GEMMCompatibility, matmul_blocks_lhs, matmul_blocks_tile_transposed_lhs


@nki.jit
def lhs_rhs_gemm(lhs: tensor, rhs: tensor, NUM_BLOCK_M: int, NUM_BLOCK_N: int, NUM_BLOCK_K: int, loop_order: str):
    mm = GEMMCompatibility(transposed_lhs=False)
    mm((lhs, rhs), {"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "NUM_BLOCK_K": NUM_BLOCK_K})

    result = nl.ndarray((mm.M, mm.N), dtype=lhs.dtype, buffer=nl.shared_hbm)
    if loop_order == "MNK":
        gemm_lhs_MNK(lhs, rhs, mm, result)
    elif loop_order == "MKN":
        gemm_lhs_MKN(lhs, rhs, mm, result)
    else:
        raise NotImplementedError(f"loop_order {loop_order} not implemented")
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
            save_result_block(
                result,
                result_block,
                tile_index_ofs=(block_id_M * mm.TILES_IN_BLOCK_M, block_id_N * mm.TILES_IN_BLOCK_N),
            )
    return result


def gemm_lhs_MKN(lhs: tensor, rhs: tensor, mm: GEMMCompatibility, result: KernelHBMTensor):
    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):
        result_block = nl.zeros(
            (nl.par_dim(mm.TILE_M), mm.TILES_IN_BLOCK_M, mm.TILES_IN_N, mm.TILE_N), dtype=result.dtype, buffer=nl.sbuf
        )
        for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):
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
                matmul_blocks_tile_transposed_lhs(lhs_block, rhs_block, result_block, block_id_N)
        save_result_dma(result, result_block, block_id_M)
    return result
