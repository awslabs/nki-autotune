# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki as nki
from neuronxcc.nki.compiler.backends.neuron.tensors import TensorRef, KernelHBMTensor
from typing import Tuple

from src.kernels.utils import load_tensor, matmul_tiles


class MatMulCompatibility:
    """
    Inputs compatibility checks for GEMM kernels
    """

    def __init__(
        self,
        lhs_shape: Tuple,
        rhs_shape: Tuple,
        NUM_BLOCK_M: int,
        NUM_BLOCK_N: int,
        NUM_BLOCK_K: int,
        BUFFER_M: int,
        BUFFER_N: int,
        BUFFER_K: int,
        loop_order: str,
    ):
        # Input sizes
        self.M, self.K = lhs_shape
        K_, self.N = rhs_shape

        # Single tile sizes
        self.TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
        self.TILE_N = nl.tile_size.gemm_moving_fmax  # 512
        self.TILE_K = nl.tile_size.pmax  # 128

        # Number of blocks
        self.NUM_BLOCK_M = NUM_BLOCK_M
        self.NUM_BLOCK_N = NUM_BLOCK_N
        self.NUM_BLOCK_K = NUM_BLOCK_K

        # Tiles in a block
        self.TILES_IN_BLOCK_M = self.M // self.NUM_BLOCK_M // self.TILE_M
        self.TILES_IN_BLOCK_N = self.N // self.NUM_BLOCK_N // self.TILE_N
        self.TILES_IN_BLOCK_K = self.K // self.NUM_BLOCK_K // self.TILE_K

        # Block sizes
        self.BLOCK_K = self.TILE_K * self.TILES_IN_BLOCK_K
        self.BLOCK_M = self.TILE_M * self.TILES_IN_BLOCK_M
        self.BLOCK_N = self.TILE_N * self.TILES_IN_BLOCK_N

        # Buffer degrees
        self.BUFFER_K = BUFFER_K
        self.BUFFER_M = BUFFER_M
        self.BUFFER_N = BUFFER_N

        self.loop_order = loop_order

        self._check(K_)

    def _check(self, K_):
        assert self.K == K_, f"lhs and rhs contraction dimension mismatch, got {self.K} and {K_}"
        assert (
            self.NUM_BLOCK_K * self.TILES_IN_BLOCK_K * self.TILE_K == self.K
        ), f"NUM_BLOCK_K {self.NUM_BLOCK_K} * TILES_IN_BLOCK_K {self.TILES_IN_BLOCK_K} * TILE_K {self.TILE_K} != K {self.K}"
        assert (
            self.NUM_BLOCK_M * self.TILES_IN_BLOCK_M * self.TILE_M == self.M
        ), f"NUM_BLOCK_M {self.NUM_BLOCK_M} * TILES_IN_BLOCK_M {self.TILES_IN_BLOCK_M} * TILE_M {self.TILE_M} != M {self.M}"
        assert (
            self.NUM_BLOCK_N * self.TILES_IN_BLOCK_N * self.TILE_N == self.N
        ), f"NUM_BLOCK_N {self.NUM_BLOCK_N} * TILES_IN_BLOCK_N {self.TILES_IN_BLOCK_N} * TILE_N {self.TILE_N} != N {self.N}"

        assert (
            self.BUFFER_M <= self.NUM_BLOCK_M
        ), f"M buffer degree {self.BUFFER_M} cannot be larger than number of blocks {self.NUM_BLOCK_M}"
        assert (
            self.BUFFER_N <= self.NUM_BLOCK_N
        ), f"N buffer degree {self.BUFFER_N} cannot be larger than number of blocks {self.NUM_BLOCK_N}"
        assert (
            self.BUFFER_K <= self.NUM_BLOCK_K
        ), f"K buffer degree {self.BUFFER_K} cannot be larger than number of blocks {self.NUM_BLOCK_K}"

        assert self.loop_order in [
            "MNK",
            "MKN",
            "NMK",
            "NKM",
            "KMN",
            "KNM",
        ], f"Loop order {self.loop_order} GEMM does not exist."


@nki.jit
def matmul_main(
    lhsT: TensorRef,
    rhs: TensorRef,
    TILES_IN_BLOCK_K: int,
    TILES_IN_BLOCK_M: int,
    TILES_IN_BLOCK_N: int,
    BUFFER_K: int,
    BUFFER_M: int,
    BUFFER_N: int,
    loop_order: str,
):
    mm = MatMulCompatibility(
        lhsT.shape[::-1],
        rhs.shape,
        TILES_IN_BLOCK_K,
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
        BUFFER_K,
        BUFFER_M,
        BUFFER_N,
        loop_order,
    )
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


def save_result_block(result, result_tiles, m_ofs, n_ofs):
    num_m_tiles, num_n_tiles, TILE_M, TILE_N = result_tiles.shape

    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for tile_id_M in nl.affine_range(num_m_tiles):
        for tile_id_N in nl.affine_range(num_n_tiles):
            nl.store(
                result[m_ofs + tile_id_M * TILE_M + idx_res.p, n_ofs + tile_id_N * TILE_N + idx_res.x],
                value=result_tiles[tile_id_M, tile_id_N],
            )


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
                lhsT_tiles = load_tensor(
                    input_tensor=lhsT,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_M * mm.BLOCK_M,
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
                )
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_N
                rhs_tiles = load_tensor(
                    input_tensor=rhs,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_N * mm.BLOCK_N,
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
                )
                matmul_tiles(lhsT_tiles, rhs_tiles, result_tiles)

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
                lhsT_tiles = load_tensor(
                    input_tensor=lhsT,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_M * mm.BLOCK_M,
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
                )
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_N
                rhs_tiles = load_tensor(
                    input_tensor=rhs,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_N * mm.BLOCK_N,
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
                )
                matmul_tiles(lhsT_tiles, rhs_tiles, result_tiles)

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
            lhsT_tiles = load_tensor(
                input_tensor=lhsT,
                par_ofs=block_id_K * mm.BLOCK_K,
                free_ofs=block_id_M * mm.BLOCK_M,
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
            )
            for block_id_N in nl.affine_range(mm.NUM_BLOCK_N, multi_buffer=mm.BUFFER_N):
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_N
                rhs_tiles = load_tensor(
                    input_tensor=rhs,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_N * mm.BLOCK_N,
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
                )

                matmul_tiles(lhsT_tiles, rhs_tiles, result_tiles[block_id_K, block_id_M, block_id_N])

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
            rhs_tiles = load_tensor(
                input_tensor=rhs,
                par_ofs=block_id_K * mm.BLOCK_K,
                free_ofs=block_id_N * mm.BLOCK_N,
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
            )
            for block_id_M in nl.affine_range(mm.NUM_BLOCK_M, multi_buffer=mm.BUFFER_M):
                # TILES_IN_BLOCK_K, TILE_K, BLOCK_M
                lhsT_tiles = load_tensor(
                    input_tensor=lhsT,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_M * mm.BLOCK_M,
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
                )

                matmul_tiles(lhsT_tiles, rhs_tiles, result_tiles[block_id_K, block_id_M, block_id_N])

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
            rhs_tiles = load_tensor(
                input_tensor=rhs,
                par_ofs=block_id_K * mm.BLOCK_K,
                free_ofs=block_id_N * mm.BLOCK_N,
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
            )

            # Blocking M dimension (the LHS free dimension)
            for block_id_M in nl.affine_range(mm.NUM_BLOCK_M, multi_buffer=mm.BUFFER_M):
                # Loading tiles from lhsT
                lhsT_tiles = load_tensor(
                    input_tensor=lhsT,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_M * mm.BLOCK_M,
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
                )

                matmul_tiles(lhsT_tiles, rhs_tiles, result_tiles[block_id_M])

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
            lhsT_tiles = load_tensor(
                input_tensor=lhsT,
                par_ofs=block_id_K * mm.BLOCK_K,
                free_ofs=block_id_M * mm.BLOCK_M,
                load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_M),
            )

            # Blocking N dimension (the RHS free dimension)
            for block_id_N in nl.affine_range(mm.NUM_BLOCK_N, multi_buffer=mm.BUFFER_N):
                # Loading tiles from rhs
                # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
                rhs_tiles = load_tensor(
                    input_tensor=rhs,
                    par_ofs=block_id_K * mm.BLOCK_K,
                    free_ofs=block_id_N * mm.BLOCK_N,
                    load_shape=(mm.TILES_IN_BLOCK_K, mm.TILE_K, mm.BLOCK_N),
                )

                matmul_tiles(lhsT_tiles, rhs_tiles, result_tiles[block_id_N])

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
