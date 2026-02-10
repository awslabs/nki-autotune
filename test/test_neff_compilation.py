"""Tests for NEFF compilation backend.

This module contains tests for the compile_kernel function that uses the
compile_nki_ir_kernel_to_neff API to compile NKI kernels to NEFF format.

Run with: pytest test/test_neff_compilation.py -v

Note: These tests require a properly configured NeuronX compiler environment
with Trainium hardware or simulation support.
"""

import os
import subprocess

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest

from autotune.compiler.compile import compile_kernel


def _neuron_devices_available() -> bool:
    """Check if Neuron devices are available by running neuron-ls."""
    try:
        result = subprocess.run(["neuron-ls"], capture_output=True, timeout=10)
        return result.returncode == 0
    except FileNotFoundError:
        return False


NEURON_DEVICES_AVAILABLE = _neuron_devices_available()


@nki.jit
def nki_matmul_block_free_dimension_(lhsT, rhs):
    """NKI kernel to compute matrix multiplication with blocked free dimensions.

    Computes C = lhsT.T @ rhs where lhsT is the transposed left-hand-side matrix.
    Blocking the free dimensions improves memory access patterns.

    Args:
        lhsT: Input tensor of shape [K, M], where K and M are multiples of 128.
            Left-hand-side argument delivered transposed for optimal performance.
        rhs: Input tensor of shape [K, N], where K is a multiple of 128 and N
            is a multiple of 512.

    Returns:
        result: Output tensor of shape [M, N].
    """
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"

    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_K = nl.tile_size.pmax
    TILE_N = nl.tile_size.gemm_moving_fmax

    TILES_IN_BLOCK_M = 2
    TILES_IN_BLOCK_N = 2

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // BLOCK_M):
        lhsT_tiles = []
        for bm in nl.affine_range(TILES_IN_BLOCK_M):
            lhsT_tiles_internal = []
            for k in nl.affine_range(K // TILE_K):
                lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=lhsT_tile,
                    src=lhsT[
                        k * TILE_K : (k + 1) * TILE_K,
                        (m * TILES_IN_BLOCK_M + bm) * TILE_M : ((m * TILES_IN_BLOCK_M + bm) + 1) * TILE_M,
                    ],
                )
                lhsT_tiles_internal.append(lhsT_tile)
            lhsT_tiles.append(lhsT_tiles_internal)

        for n in nl.affine_range(N // BLOCK_N):
            rhs_tiles = []
            for bn in nl.affine_range(TILES_IN_BLOCK_N):
                rhs_tiles_internal = []
                for k in nl.affine_range(K // TILE_K):
                    rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=rhs_tile,
                        src=rhs[
                            k * TILE_K : (k + 1) * TILE_K,
                            (n * TILES_IN_BLOCK_N + bn) * TILE_N : ((n * TILES_IN_BLOCK_N + bn) + 1) * TILE_N,
                        ],
                    )
                    rhs_tiles_internal.append(rhs_tile)
                rhs_tiles.append(rhs_tiles_internal)

            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    result_tile = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                    for k in nl.affine_range(K // TILE_K):
                        nisa.nc_matmul(dst=result_tile, stationary=lhsT_tiles[bm][k], moving=rhs_tiles[bn][k])

                    result_tmp = nl.ndarray(shape=result_tile.shape, dtype=result.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=result_tmp, src=result_tile)

                    nisa.dma_copy(
                        dst=result[
                            (m * TILES_IN_BLOCK_M + bm) * TILE_M : ((m * TILES_IN_BLOCK_M + bm) + 1) * TILE_M,
                            (n * TILES_IN_BLOCK_N + bn) * TILE_N : ((n * TILES_IN_BLOCK_N + bn) + 1) * TILE_N,
                        ],
                        src=result_tmp,
                    )

    return result


def matmul_transposed_lhs_golden(lhsT: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Compute golden reference for transposed-LHS matmul.

    Args:
        lhsT: Transposed left-hand side matrix of shape [K, M].
        rhs: Right-hand side matrix of shape [K, N].

    Returns:
        Expected result of lhsT.T @ rhs.
    """
    return lhsT.T @ rhs


class TestCompileKernel:
    """Tests for compile_kernel function.

    This class verifies that compile_kernel correctly compiles NKI kernels
    to NEFF format using the compile_nki_ir_kernel_to_neff API.
    """

    @pytest.mark.skipif(not NEURON_DEVICES_AVAILABLE, reason="Requires Neuron devices (neuron-ls must succeed).")
    def test_compile_kernel_produces_neff(self, tmp_path):
        """Verify compile_kernel produces a valid NEFF file.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        M = 256
        N = 1024
        K = 128

        lhsT = np.zeros((K, M), dtype=np.float32)
        rhs = np.zeros((K, N), dtype=np.float32)

        input_tensors = {"lhsT": lhsT, "rhs": rhs}
        output_tensors = [("result", (M, N), np.float32)]

        kernel_file = os.path.abspath(__file__)
        kernel_name = (kernel_file, "nki_matmul_block_free_dimension_")

        neff_path = compile_kernel(
            kernel_name=kernel_name,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            kernel_kwargs={},
            compiler_flags="",
            output_dir=str(tmp_path),
        )

        assert neff_path.endswith(".neff"), f"Expected path ending with .neff, got: {neff_path}"
        assert os.path.exists(neff_path), f"NEFF file does not exist at: {neff_path}"


class TestMultiKernelWorkload:
    """Tests for multi-kernel workload compilation using ProfileJobs infrastructure.

    This class verifies that the autotune infrastructure can compile multiple
    kernel configurations with different matrix sizes in a single batch.
    """

    @pytest.mark.skipif(not NEURON_DEVICES_AVAILABLE, reason="Requires Neuron devices (neuron-ls must succeed).")
    def test_multi_kernel_compilation(self, tmp_path):
        """Verify compilation succeeds for multiple kernel configurations.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        from autotune.job import ProfileJobs, compile_jobs

        jobs = ProfileJobs(cache_root_dir=str(tmp_path))

        test_sizes = [(256, 1024, 128), (512, 1024, 256), (256, 2048, 128)]

        for M, N, K in test_sizes:
            lhsT = np.zeros((K, M), dtype=np.float32)
            rhs = np.zeros((K, N), dtype=np.float32)
            jobs.add_job(
                kernel=nki_matmul_block_free_dimension_,
                kernel_kwargs={"lhsT": lhsT, "rhs": rhs},
                output_shapes={"result": (M, N)},
                compiler_flags="",
                correctness_check=(matmul_transposed_lhs_golden, 1e-3, 1e-3),
            )

        compiled_jobs = compile_jobs(jobs)

        for job_index, job in compiled_jobs.jobs.items():
            assert not job.has_error, f"Job {job_index} failed with error: {getattr(job, 'error', 'unknown')}"
            assert hasattr(job, "neff"), f"Job {job_index} missing neff attribute"
            assert job.neff.endswith(".neff"), f"Job {job_index} neff path invalid: {job.neff}"
