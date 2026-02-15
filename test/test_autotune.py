"""Tests for autotune benchmarking and NEFF compilation backend."""

import os

import autotune_golden
import numpy as np
import pytest
from autotune_golden import (
    ATOL,
    ITERS,
    NEURON_DEVICES_AVAILABLE,
    RTOL,
    SCALAR_VALUES,
    SHAPES,
    WARMUP,
    golden_add_scalar,
    matmul_transposed_lhs_golden,
    nki_matmul_block_free_dimension_,
    nki_tensor_add_scalar_,
)

from autotune import Benchmark, BenchmarkResults, ProfileJobs, load_results
from autotune.compiler.compile import compile_kernel


class TestBenchmarkTensorAddScalar:
    """End-to-end test for the autotune pipeline using a tensor-add-scalar kernel."""

    @pytest.mark.skipif(not NEURON_DEVICES_AVAILABLE, reason="Requires Neuron devices (neuron-ls must succeed).")
    def test_full_pipeline(self, tmp_path):
        """Verify the full benchmark pipeline: compile, run, check correctness, analyze results.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        jobs = ProfileJobs(cache_root_dir=str(tmp_path))

        for shape in SHAPES:
            for scalar in SCALAR_VALUES:
                a = np.random.randn(*shape).astype(np.float32)
                b = np.random.randn(*shape).astype(np.float32)
                jobs.add_job(
                    kernel=nki_tensor_add_scalar_,
                    kernel_kwargs={"a_input": a, "b_input": b, "c": scalar},
                    output_shapes={"result": shape},
                    compiler_flags="",
                    correctness_check=(golden_add_scalar, ATOL, RTOL),
                )

        assert len(jobs.jobs) == 9

        results = Benchmark(jobs, warmup=WARMUP, iters=ITERS).run()

        assert isinstance(results, BenchmarkResults)
        assert len(results.workloads) == len(SHAPES)

        for wl_name, wl in results.workloads.items():
            assert len(wl.entries) == len(SCALAR_VALUES)
            for entry in wl.entries:
                assert not entry.has_error, f"Entry has error: {entry.data.get('error')}"
                assert entry.data.get("correctness_result") is True
                assert entry.data["min_ms"] > 0
                assert entry.data["mean_ms"] > 0
                assert entry.data["max_ms"] >= entry.data["min_ms"]
                assert entry.data["iterations"] == ITERS
                assert entry.data["warmup_iterations"] == WARMUP

        reloaded = BenchmarkResults.load(str(tmp_path))
        assert len(reloaded.workloads) == len(results.workloads)

        reloaded_alias = load_results(str(tmp_path))
        assert len(reloaded_alias.workloads) == len(results.workloads)

        results.summary()

        best = results.best_configs()
        assert len(best) == len(SHAPES)
        for wl_name, entry in best.items():
            assert entry.min_ms > 0

        filtered = results.filter(lambda e: e.min_ms > 0)
        assert len(filtered.workloads) == len(SHAPES)

        empty = results.filter(lambda e: e.min_ms < 0)
        assert len(empty.workloads) == 0


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

        kernel_file = os.path.abspath(autotune_golden.__file__)
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
