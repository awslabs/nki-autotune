"""End-to-end tests for the autotune benchmarking pipeline.

Exercises the full profiling pipeline on real Neuron hardware: job creation,
compilation, execution, correctness checking, and results analysis.

Run with: pytest test/test_autotune_e2e.py -v
"""

import math
import subprocess
from pathlib import Path

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from conftest import assert_arrays_close, make_random_array

import nkigym
from autotune import Benchmark, BenchmarkResults, ProfileJobs, load_results
from nkigym.codegen import lower_ir_to_nki
from nkigym.ir import ir_to_callable, ir_to_source
from nkigym.ops import NKIMatmul
from nkigym.search import search
from nkigym.tiling import generate_tiled_ir
from nkigym.transforms import DataReuseTransform, OperandMergeTransform
from nkigym.utils.source import get_source


def _neuron_devices_available() -> bool:
    """Check if Neuron devices are available by running neuron-ls."""
    try:
        result = subprocess.run(["neuron-ls"], capture_output=True, timeout=10)
        return result.returncode == 0
    except FileNotFoundError:
        return False


NEURON_DEVICES_AVAILABLE = _neuron_devices_available()

SHAPES = [(128, 128), (128, 256), (128, 512)]
SCALAR_VALUES = [0.0, 1.5, -2.0]
ATOL, RTOL = 1e-5, 1e-5
WARMUP, ITERS = 2, 5


@nki.jit
def nki_tensor_add_scalar_(a_input, b_input, c):
    """NKI kernel that computes a_input + b_input + c.

    Args:
        a_input: First input tensor of shape [P, F] where P <= 128.
        b_input: Second input tensor of shape [P, F] where P <= 128.
        c: Scalar value to add.

    Returns:
        result: Output tensor of shape [P, F].
    """
    P, F = a_input.shape
    result = nl.ndarray((P, F), dtype=a_input.dtype, buffer=nl.shared_hbm)

    a_tile = nl.ndarray((P, F), dtype=a_input.dtype, buffer=nl.sbuf)
    b_tile = nl.ndarray((P, F), dtype=b_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_tile, src=a_input)
    nisa.dma_copy(dst=b_tile, src=b_input)

    sum_tile = nl.ndarray((P, F), dtype=a_input.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sum_tile, data1=a_tile, data2=b_tile, op=nl.add)

    result_tile = nl.ndarray((P, F), dtype=a_input.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(result_tile, sum_tile, nl.add, c)

    nisa.dma_copy(dst=result, src=result_tile)
    return result


def golden_add_scalar(a_input: np.ndarray, b_input: np.ndarray, c: float) -> np.ndarray:
    """Golden reference for nki_tensor_add_scalar_.

    Args:
        a_input: First input array.
        b_input: Second input array.
        c: Scalar value to add.

    Returns:
        Result of a_input + b_input + c, cast to a_input's dtype.
    """
    return (a_input + b_input + c).astype(a_input.dtype)


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


class TestMatmulE2EPipeline:
    """End-to-end matmul test through the full nkigym pipeline.

    Exercises: define gym function -> tile -> search/transforms -> verify
    numerical correctness -> lower to NKI -> verify compilable output.
    """

    @pytest.mark.parametrize(
        "a_shape,b_shape",
        [((128, 128), (128, 128)), ((128, 256), (128, 256)), ((128, 256), (128, 128))],
        ids=["1x1", "1x2_square", "1x2_rect"],
    )
    def test_tile_transform_codegen(self, a_shape: tuple, b_shape: tuple) -> None:
        """Full pipeline: define -> tile -> data_reuse -> operand_merge -> NKI codegen.

        Args:
            a_shape: Shape of first input matrix (K, M).
            b_shape: Shape of second input matrix (K, N).
        """

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute matrix multiplication."""
            return nkigym.nc_matmul(a, b)

        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        expected = matmul(a, b)

        program = generate_tiled_ir(matmul, {"a": a_shape, "b": b_shape}, np.float32)
        tiled = ir_to_callable(program)
        assert_arrays_close(tiled(a, b), expected)

        reuse = DataReuseTransform()
        for group in reuse.analyze_ir(program):
            program = reuse.transform_ir(program, group)
        assert_arrays_close(ir_to_callable(program)(a, b), expected)

        merge = OperandMergeTransform()
        while True:
            opps = merge.analyze_ir(program)
            if not opps:
                break
            program = merge.transform_ir(program, opps[0])
        assert_arrays_close(ir_to_callable(program)(a, b), expected)

        nki_source = lower_ir_to_nki(program)
        compiled = compile(nki_source, "<test>", "exec")
        assert compiled is not None

    def test_search_produces_lowerable_variants(self, tmp_path: Path) -> None:
        """Search over tiled matmul produces unique variants that lower to valid NKI."""

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute matrix multiplication."""
            return nkigym.nc_matmul(a, b)

        a_shape, b_shape = (128, 128), (128, 256)
        kernel_kwargs = {"a": make_random_array(a_shape, seed=10), "b": make_random_array(b_shape, seed=11)}

        transforms = [DataReuseTransform(), OperandMergeTransform()]
        variants = search(matmul, transforms, math.inf, 42, 0, tmp_path, kernel_kwargs)

        assert len(variants) >= 1
        sources = {ir_to_source(v) for v in variants}
        assert len(sources) == len(variants)

        for variant in variants:
            nki_source = lower_ir_to_nki(variant)
            compiled = compile(nki_source, "<test>", "exec")
            assert compiled is not None


class TestReductionTilingE2EPipeline:
    """End-to-end reduction tiling test through the full nkigym pipeline.

    Exercises the reduction tiling path (K > 128) where the K dimension is
    split into 128-wide tiles with accumulation (+=).
    """

    @pytest.mark.parametrize(
        "a_shape,b_shape", [((256, 128), (256, 128)), ((384, 128), (384, 128))], ids=["k256", "k384"]
    )
    def test_reduction_tile_transform_codegen(self, a_shape: tuple, b_shape: tuple) -> None:
        """Full pipeline with reduction tiling: define -> tile -> verify -> NKI codegen.

        Args:
            a_shape: Shape of first input matrix (K, M) where K > 128.
            b_shape: Shape of second input matrix (K, N) where K > 128.
        """

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute matrix multiplication."""
            return nkigym.nc_matmul(a, b)

        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        expected = matmul(a, b)

        program = generate_tiled_ir(matmul, {"a": a_shape, "b": b_shape}, np.float32)
        tiled = ir_to_callable(program)
        assert_arrays_close(tiled(a, b), expected)

        source = get_source(tiled)
        assert "+=" in source

        compute_stmts = [s for s in program.stmts if isinstance(s[0], NKIMatmul)]
        k_tiles = a_shape[0] // 128
        assert len(compute_stmts) == k_tiles

        nki_source = lower_ir_to_nki(program)
        compiled = compile(nki_source, "<test>", "exec")
        assert compiled is not None

    def test_reduction_with_multi_n_tiles(self, tmp_path: Path) -> None:
        """Reduction tiling combined with N-dimension tiling and transforms.

        Shape (256, 128) x (256, 256) has K=256 (2 reduction tiles) and
        N=256 (2 N tiles), exercising both reduction and spatial tiling
        through search and transform application.
        """

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute matrix multiplication."""
            return nkigym.nc_matmul(a, b)

        a_shape, b_shape = (256, 128), (256, 256)
        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        expected = matmul(a, b)

        kernel_kwargs = {"a": make_random_array(a_shape, seed=10), "b": make_random_array(b_shape, seed=11)}
        transforms = [DataReuseTransform(), OperandMergeTransform()]
        variants = search(matmul, transforms, 5, 42, 0, tmp_path, kernel_kwargs)

        assert len(variants) >= 1
        for variant in variants:
            nki_source = lower_ir_to_nki(variant)
            compiled = compile(nki_source, "<test>", "exec")
            assert compiled is not None
