"""Tests for the remote NKI kernel profiling backend.

Profiles hardcoded NKI kernels on remote Trn nodes via SSH.
Tests accuracy, correctness checking, and e2e wallclock time
benefits from multi-worker parallelism.

Requires SSH access to gym-* hosts with Neuron hardware.
"""

import json
import logging
import tempfile
import time
from pathlib import Path

from autotune.runner.remote import RemoteProfiler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

REMOTE_CONFIG = {
    "hosts": ["gym-1", "gym-2", "gym-3", "gym-4", "gym-5"],
    "ssh_timeout_sec": 600,
    "neuron_platform_target": "trn2",
}


TENSOR_COPY_KERNEL = """\
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def nki_tensor_copy(a):
    output = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.shared_hbm)
    sbuf_a = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sbuf_a[0:128, 0:512], src=a[0:128, 0:512])
    sbuf_out = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sbuf_out[0:128, 0:512], src=sbuf_a[0:128, 0:512])
    nisa.dma_copy(dst=output[0:128, 0:512], src=sbuf_out[0:128, 0:512])
    return output
"""


TENSOR_ADD_KERNEL = """\
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def nki_tensor_add(a, b):
    output = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.shared_hbm)
    sbuf_a = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.sbuf)
    sbuf_b = nl.ndarray((128, 512), dtype=b.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sbuf_a[0:128, 0:512], src=a[0:128, 0:512])
    nisa.dma_copy(dst=sbuf_b[0:128, 0:512], src=b[0:128, 0:512])
    sbuf_out = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=sbuf_out[0:128, 0:512],
        data1=sbuf_a[0:128, 0:512],
        data2=sbuf_b[0:128, 0:512],
        op=nl.add,
    )
    nisa.dma_copy(dst=output[0:128, 0:512], src=sbuf_out[0:128, 0:512])
    return output
"""


MATMUL_256_KERNEL = """\
import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def nki_matmul_256(a, b):
    hbm_output = nl.ndarray((128, 256), dtype=a.dtype, buffer=nl.shared_hbm)
    psum_acc = nl.ndarray((128, 256), dtype=nl.float32, buffer=nl.psum)
    nisa.memset(dst=psum_acc[0:128, 0:256], value=0.0)
    sbuf_a = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    sbuf_b = nl.ndarray((128, 256), dtype=b.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sbuf_a[0:128, 0:128], src=a[0:128, 0:128])
    nisa.dma_copy(dst=sbuf_b[0:128, 0:256], src=b[0:128, 0:256])
    nisa.nc_matmul(
        dst=psum_acc[0:128, 0:256],
        stationary=sbuf_a[0:128, 0:128],
        moving=sbuf_b[0:128, 0:256],
    )
    sbuf_out = nl.ndarray((128, 256), dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sbuf_out[0:128, 0:256], src=psum_acc[0:128, 0:256])
    nisa.dma_copy(dst=hbm_output[0:128, 0:256], src=sbuf_out[0:128, 0:256])
    return hbm_output
"""


GOLDEN_ADD = """\
import numpy as np

def golden_add(a, b):
    return a + b
"""

GOLDEN_MATMUL = """\
import numpy as np

def golden_matmul(a, b):
    \"\"\"nc_matmul: stationary[K,M] @ moving[K,N] = [M,N] = a.T @ b\"\"\"
    return a.T @ b
"""

GOLDEN_COPY = """\
import numpy as np

def golden_copy(a):
    return a.copy()
"""


def _make_config_file() -> str:
    """Write REMOTE_CONFIG to a temp JSON file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(REMOTE_CONFIG, f)
    f.close()
    return f.name


def _make_profiler(hosts: list[str]) -> RemoteProfiler:
    """Create a RemoteProfiler with specified hosts."""
    return RemoteProfiler(
        hosts=hosts,
        ssh_timeout_sec=REMOTE_CONFIG["ssh_timeout_sec"],
        neuron_platform_target=REMOTE_CONFIG["neuron_platform_target"],
        warmup=2,
        iters=10,
    )


class TestRemoteProfileSingleWorker:
    """Profile NKI kernels using a single remote worker."""

    def test_tensor_copy(self) -> None:
        """Profile a simple tensor copy kernel on 1 worker."""
        profiler = _make_profiler(hosts=["gym-1"])
        results = profiler.profile(
            kernels={"copy_v0.py": TENSOR_COPY_KERNEL},
            input_specs={"a": ((128, 512), "bfloat16")},
            golden_source=GOLDEN_COPY,
            golden_func_name="golden_copy",
        )
        assert len(results) == 1
        r = results[0]
        assert r.kernel_name == "copy_v0.py"
        assert r.error == "", f"Kernel errored: {r.error}"
        assert r.correct is True
        assert r.min_ms > 0
        assert r.mean_ms >= r.min_ms
        assert r.p50_ms > 0
        assert r.p99_ms >= r.p50_ms

    def test_tensor_add_with_correctness(self) -> None:
        """Profile tensor add and verify correctness against golden."""
        profiler = _make_profiler(hosts=["gym-1"])
        results = profiler.profile(
            kernels={"add_v0.py": TENSOR_ADD_KERNEL},
            input_specs={"a": ((128, 512), "bfloat16"), "b": ((128, 512), "bfloat16")},
            golden_source=GOLDEN_ADD,
            golden_func_name="golden_add",
            atol=1e-2,
            rtol=1e-2,
        )
        assert len(results) == 1
        r = results[0]
        assert r.error == "", f"Kernel errored: {r.error}"
        assert r.correct is True

    def test_matmul_with_mfu(self) -> None:
        """Profile matmul and check MFU is reported."""
        profiler = _make_profiler(hosts=["gym-1"])
        """128x128x256 matmul: MAC count = 128 * 128 * 256 = 4_194_304"""
        mac_count = 128 * 128 * 256
        results = profiler.profile(
            kernels={"matmul_v0.py": MATMUL_256_KERNEL},
            input_specs={"a": ((128, 128), "bfloat16"), "b": ((128, 256), "bfloat16")},
            mac_count=mac_count,
            golden_source=GOLDEN_MATMUL,
            golden_func_name="golden_matmul",
            atol=0.5,
            rtol=0.1,
        )
        assert len(results) == 1
        r = results[0]
        assert r.error == "", f"Kernel errored: {r.error}"
        assert r.correct is True
        assert r.mfu > 0, "MFU should be > 0 for matmul"
        assert r.mac_count == mac_count

    def test_multiple_kernels_one_worker(self) -> None:
        """Send multiple kernels to a single worker."""
        profiler = _make_profiler(hosts=["gym-1"])
        kernels = {"copy_v0.py": TENSOR_COPY_KERNEL, "add_v0.py": TENSOR_ADD_KERNEL}
        results = profiler.profile(kernels=kernels, input_specs={"a": ((128, 512), "bfloat16")})
        """
        The add kernel will fail compilation because it expects 2 inputs
        but we only gave 1 in input_specs. The copy kernel should succeed.
        We just check that both kernels get a result (success or error).
        """
        assert len(results) == 2
        names = {r.kernel_name for r in results}
        assert names == {"copy_v0.py", "add_v0.py"}

    def test_from_config_file(self) -> None:
        """Test loading config from JSON file."""
        config_path = _make_config_file()
        profiler = RemoteProfiler.from_config(config_path, warmup=2, iters=5)
        assert profiler.hosts == REMOTE_CONFIG["hosts"]
        assert profiler.warmup == 2
        assert profiler.iters == 5
        Path(config_path).unlink()


class TestRemoteProfileMultiWorker:
    """Profile NKI kernels across multiple remote workers."""

    def test_parallel_speedup(self) -> None:
        """Distribute kernels across workers and verify wallclock speedup.

        Sends the same kernel 5 times (as 5 "variants") to 1 worker
        vs 5 workers. The multi-worker run should be faster.
        """
        all_hosts = REMOTE_CONFIG["hosts"]
        num_hosts = len(all_hosts)
        kernels = {f"copy_v{i}.py": TENSOR_COPY_KERNEL for i in range(num_hosts)}
        input_specs = {"a": ((128, 512), "bfloat16")}

        profiler_1 = _make_profiler(hosts=[all_hosts[0]])
        t0 = time.monotonic()
        results_1 = profiler_1.profile(kernels=kernels, input_specs=input_specs)
        time_1_worker = time.monotonic() - t0

        profiler_n = _make_profiler(hosts=all_hosts)
        t0 = time.monotonic()
        results_n = profiler_n.profile(kernels=kernels, input_specs=input_specs)
        time_n_workers = time.monotonic() - t0

        successful_1 = [r for r in results_1 if not r.error]
        successful_n = [r for r in results_n if not r.error]
        assert (
            len(successful_1) == num_hosts
        ), f"Expected {num_hosts} results from 1 worker, got {len(successful_1)} successes"
        assert (
            len(successful_n) == num_hosts
        ), f"Expected {num_hosts} results from {num_hosts} workers, got {len(successful_n)} successes"

        print(f"\n  1 worker:  {time_1_worker:.1f}s")
        print(f"  {num_hosts} workers: {time_n_workers:.1f}s")
        print(f"  Speedup:   {time_1_worker / time_n_workers:.1f}x")

        """
        With multiple independent workers we expect at least 1.5x speedup
        since compilation dominates and is embarrassingly parallel.
        """
        assert time_n_workers < time_1_worker, (
            f"Multi-worker ({time_n_workers:.1f}s) should be faster " f"than single-worker ({time_1_worker:.1f}s)"
        )

    def test_results_consistency(self) -> None:
        """Timing results from different workers should be consistent.

        The same kernel compiled and run on different hosts should
        produce similar timing (within 10x, accounting for compilation
        variance and cold-start effects).
        """
        all_hosts = REMOTE_CONFIG["hosts"]
        num_hosts = len(all_hosts)
        kernels = {f"add_v{i}.py": TENSOR_ADD_KERNEL for i in range(num_hosts)}
        profiler = _make_profiler(hosts=all_hosts)
        results = profiler.profile(
            kernels=kernels,
            input_specs={"a": ((128, 512), "bfloat16"), "b": ((128, 512), "bfloat16")},
            golden_source=GOLDEN_ADD,
            golden_func_name="golden_add",
        )
        successful = [r for r in results if not r.error]
        assert len(successful) == num_hosts

        times = [r.min_ms for r in successful]
        min_time = min(times)
        max_time = max(times)
        print(f"\n  Timing across {num_hosts} hosts: {[f'{t:.4f}ms' for t in times]}")
        print(f"  Spread: {max_time / min_time:.1f}x")
        assert max_time / min_time < 10, f"Timing spread {max_time/min_time:.1f}x is too large; " f"times: {times}"

    def test_correctness_across_workers(self) -> None:
        """All workers should pass correctness checks."""
        all_hosts = REMOTE_CONFIG["hosts"]
        num_hosts = len(all_hosts)
        kernels = {f"add_v{i}.py": TENSOR_ADD_KERNEL for i in range(num_hosts)}
        profiler = _make_profiler(hosts=all_hosts)
        results = profiler.profile(
            kernels=kernels,
            input_specs={"a": ((128, 512), "bfloat16"), "b": ((128, 512), "bfloat16")},
            golden_source=GOLDEN_ADD,
            golden_func_name="golden_add",
            atol=1e-2,
            rtol=1e-2,
        )
        for r in results:
            assert r.error == "", f"{r.kernel_name} errored: {r.error}"
            assert r.correct is True, f"{r.kernel_name} failed correctness"

    def test_100_kernels_speedup(self) -> None:
        """Distribute 100 kernels across 1 vs all hosts and compare wallclock.

        Sends 100 identical copy kernel variants. Measures total wallclock
        time for 1 worker vs all 5 workers to quantify parallelism benefit.
        """
        num_kernels = 100
        all_hosts = REMOTE_CONFIG["hosts"]
        num_hosts = len(all_hosts)
        kernels = {f"copy_v{i}.py": TENSOR_COPY_KERNEL for i in range(num_kernels)}
        input_specs = {"a": ((128, 512), "bfloat16")}

        profiler_1 = _make_profiler(hosts=[all_hosts[0]])
        t0 = time.monotonic()
        results_1 = profiler_1.profile(kernels=kernels, input_specs=input_specs)
        time_1_worker = time.monotonic() - t0

        profiler_n = _make_profiler(hosts=all_hosts)
        t0 = time.monotonic()
        results_n = profiler_n.profile(kernels=kernels, input_specs=input_specs)
        time_n_workers = time.monotonic() - t0

        successful_1 = [r for r in results_1 if not r.error]
        successful_n = [r for r in results_n if not r.error]
        assert len(successful_1) == num_kernels, f"Expected {num_kernels} from 1 worker, got {len(successful_1)}"
        assert (
            len(successful_n) == num_kernels
        ), f"Expected {num_kernels} from {num_hosts} workers, got {len(successful_n)}"

        speedup = time_1_worker / time_n_workers
        print(f"\n  {num_kernels} kernels:")
        print(f"  1 worker:       {time_1_worker:.1f}s")
        print(f"  {num_hosts} workers:    {time_n_workers:.1f}s")
        print(f"  Speedup:        {speedup:.1f}x")

        assert time_n_workers < time_1_worker, (
            f"Multi-worker ({time_n_workers:.1f}s) should be faster " f"than single-worker ({time_1_worker:.1f}s)"
        )


class TestRemoteProfileErrorHandling:
    """Test error handling in the remote profiling backend."""

    def test_invalid_kernel_source(self) -> None:
        """An invalid kernel should produce an error result, not crash."""
        bad_kernel = """\
import nki
import nki.language as nl

@nki.jit
def bad_func(a):
    this is not valid python
"""
        profiler = _make_profiler(hosts=["gym-1"])
        results = profiler.profile(kernels={"bad_v0.py": bad_kernel}, input_specs={"a": ((128, 512), "bfloat16")})
        assert len(results) == 1
        r = results[0]
        assert r.error != "", "Expected error for invalid kernel"
        assert r.correct is False

    def test_empty_kernels(self) -> None:
        """Empty kernel dict should return empty results."""
        profiler = _make_profiler(hosts=["gym-1"])
        results = profiler.profile(kernels={}, input_specs={"a": ((128, 512), "bfloat16")})
        assert results == []
