import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def matmul_ref_2k(lhsT, rhs):
    TILE_M = 128
    TILE_K = 128
    TILE_N = 512
    TILES_IN_BLOCK_M = 4
    TILES_IN_BLOCK_N = 1
    TILES_IN_BLOCK_K = 8

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N
    BLOCK_K = TILE_K * TILES_IN_BLOCK_K

    NUM_BLOCK_M = 2048 // BLOCK_M
    NUM_BLOCK_N = 2048 // BLOCK_N
    NUM_BLOCK_K = 2048 // BLOCK_K

    result = nl.ndarray((2048, 2048), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    for n in nl.affine_range(NUM_BLOCK_N):
        result_tiles = [
            [
                [nl.ndarray((TILE_M, TILE_N), dtype=lhsT.dtype, buffer=nl.sbuf) for _ in range(TILES_IN_BLOCK_N)]
                for _ in range(TILES_IN_BLOCK_M)
            ]
            for _ in range(NUM_BLOCK_M)
        ]
        for mm in range(NUM_BLOCK_M):
            for bmm in range(TILES_IN_BLOCK_M):
                for bnn in range(TILES_IN_BLOCK_N):
                    nisa.memset(result_tiles[mm][bmm][bnn][0:TILE_M, 0:TILE_N], 0.0)

        for k in nl.sequential_range(NUM_BLOCK_K):
            rhs_tiles = [
                nl.ndarray((TILE_K, BLOCK_N), dtype=rhs.dtype, buffer=nl.sbuf) for _ in range(TILES_IN_BLOCK_K)
            ]
            for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
                nisa.dma_copy(
                    rhs_tiles[bk_r][0:TILE_K, 0:BLOCK_N],
                    rhs[
                        (TILES_IN_BLOCK_K * k + bk_r) * TILE_K : (TILES_IN_BLOCK_K * k + bk_r) * TILE_K + TILE_K,
                        BLOCK_N * n : BLOCK_N * n + BLOCK_N,
                    ],
                )

            for m in nl.affine_range(NUM_BLOCK_M):
                lhsT_tiles = [
                    nl.ndarray((TILE_K, BLOCK_M), dtype=lhsT.dtype, buffer=nl.sbuf) for _ in range(TILES_IN_BLOCK_K)
                ]
                for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
                    nisa.dma_copy(
                        lhsT_tiles[bk_l][0:TILE_K, 0:BLOCK_M],
                        lhsT[
                            (TILES_IN_BLOCK_K * k + bk_l) * TILE_K : (TILES_IN_BLOCK_K * k + bk_l) * TILE_K + TILE_K,
                            BLOCK_M * m : BLOCK_M * m + BLOCK_M,
                        ],
                    )

                res_tile = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                acc = nl.ndarray((TILE_M, TILE_N), dtype=lhsT.dtype, buffer=nl.sbuf)
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    for bm in nl.affine_range(TILES_IN_BLOCK_M):
                        nisa.memset(res_tile[0:TILE_M, 0:TILE_N], 0.0)
                        for bk in nl.affine_range(TILES_IN_BLOCK_K):
                            nisa.nc_matmul(
                                dst=res_tile[0:TILE_M, 0:TILE_N],
                                stationary=lhsT_tiles[bk][0:TILE_K, bm * TILE_M : bm * TILE_M + TILE_M],
                                moving=rhs_tiles[bk][0:TILE_K, bn * TILE_N : bn * TILE_N + TILE_N],
                            )
                        nisa.tensor_copy(acc[0:TILE_M, 0:TILE_N], res_tile[0:TILE_M, 0:TILE_N])
                        nisa.tensor_tensor(
                            result_tiles[m][bm][bn][0:TILE_M, 0:TILE_N],
                            result_tiles[m][bm][bn][0:TILE_M, 0:TILE_N],
                            acc[0:TILE_M, 0:TILE_N],
                            op=nl.add,
                        )

        for m in nl.affine_range(NUM_BLOCK_M):
            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                result_packed = nl.ndarray((TILE_K, BLOCK_N), dtype=lhsT.dtype, buffer=nl.sbuf)
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    nisa.tensor_copy(
                        result_packed[0:TILE_K, bn * TILE_N : bn * TILE_N + TILE_N],
                        result_tiles[m][bm][bn][0:TILE_M, 0:TILE_N],
                    )
                nisa.dma_copy(
                    result[
                        (TILES_IN_BLOCK_M * m + bm) * TILE_K : (TILES_IN_BLOCK_M * m + bm) * TILE_K + TILE_K,
                        BLOCK_N * n : BLOCK_N * n + BLOCK_N,
                    ],
                    result_packed[0:TILE_K, 0:BLOCK_N],
                )

    return result
