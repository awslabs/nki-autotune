import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor

from autotune.core.dma import save_result_block
from autotune.core.utils import GEMMCompatibility, load_tensor_block, matmul_blocks_lhs


@nki.jit
def gemm_with_non_transposed_lhs_MNK(
    lhs: tensor, rhs: tensor, NUM_BLOCK_M: int, NUM_BLOCK_N: int, NUM_BLOCK_K: int  # Shape (M, K)  # Shape (K, N)
):
    mm = GEMMCompatibility(transposed_lhs=False)
    mm(lhs.shape, rhs.shape, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K)
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
    lhs: tensor, rhs: tensor, NUM_BLOCK_M: int, NUM_BLOCK_N: int, NUM_BLOCK_K: int  # Shape (M, K)  # Shape (K, N)
):
    mm = GEMMCompatibility(transposed_lhs=False)
    mm(lhs.shape, rhs.shape, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K)
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
