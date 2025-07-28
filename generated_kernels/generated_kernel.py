# This is auto generated kernel codes. Do not modify directly.
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor

from autotune.modules.dma import load_tensor_block, save_result_block
from autotune.modules.matmul import GEMMCompatibility, matmul_blocks_lhsT


@nki.jit
def lhs_rhs_gemm(lhs: tensor, rhs: tensor, NUM_BLOCK_M: int, NUM_BLOCK_N: int, NUM_BLOCK_K: int):

    kernel_kwargs = {"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "NUM_BLOCK_K": NUM_BLOCK_K}
    mm = GEMMCompatibility(transposed_lhs=True)
    mm((lhs, rhs), kernel_kwargs)
    result = nl.ndarray((mm.M, mm.N), dtype=lhs.dtype, buffer=nl.shared_hbm)

    for block_id_M in nl.affine_range(mm.NUM_BLOCK_M):

        lhs_block = load_tensor_block(
            input_tensor=lhs, dim_0=(mm.TILE_K, 0, mm.TILES_IN_K), dim_1=(mm.TILE_M, block_id_M, mm.TILES_IN_BLOCK_M)
        )

        for block_id_N in nl.affine_range(mm.NUM_BLOCK_N):

            result_block = nl.zeros(
                (mm.TILE_M, mm.TILES_IN_BLOCK_M, mm.TILES_IN_BLOCK_N, mm.TILE_N), dtype=result.dtype, buffer=nl.sbuf
            )

            for block_id_K in nl.affine_range(mm.NUM_BLOCK_K):

                rhs_block = load_tensor_block(
                    input_tensor=rhs,
                    dim_0=(mm.TILE_K, block_id_K, mm.TILES_IN_BLOCK_K),
                    dim_1=(mm.TILE_N, block_id_N, mm.TILES_IN_BLOCK_N),
                )

                matmul_blocks_lhsT(lhs_block, rhs_block, result_block, ofs=xxx)

            save_result_block(result, result_block, xxx)

    return result
