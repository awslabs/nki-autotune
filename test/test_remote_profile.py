"""Tests for the remote NKI kernel profiling backend.

Profiles hardcoded NKI kernels on remote Trn nodes via SSH.
Tests cpu_sim correctness, hardware benchmarking, and e2e wallclock
time benefits from multi-worker parallelism.

Requires SSH access to gym-* hosts with Neuron hardware.
"""

import logging
import time

from autotune.runner.remote import RemoteProfiler
from autotune.runner.types import KernelJob

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


def _make_profiler(hosts: list[str]) -> RemoteProfiler:
    """Create a RemoteProfiler with specified hosts."""
    return RemoteProfiler(
        hosts=hosts,
        ssh_timeout_sec=REMOTE_CONFIG["ssh_timeout_sec"],
        neuron_platform_target=REMOTE_CONFIG["neuron_platform_target"],
        warmup=2,
        iters=10,
    )


def _copy_job() -> KernelJob:
    """Create a KernelJob for the tensor copy kernel."""
    return KernelJob(
        source=TENSOR_COPY_KERNEL,
        input_specs={"a": ((128, 512), "bfloat16")},
        golden_source=GOLDEN_COPY,
        golden_func_name="golden_copy",
    )


def _add_job() -> KernelJob:
    """Create a KernelJob for the tensor add kernel."""
    return KernelJob(
        source=TENSOR_ADD_KERNEL,
        input_specs={"a": ((128, 512), "bfloat16"), "b": ((128, 512), "bfloat16")},
        golden_source=GOLDEN_ADD,
        golden_func_name="golden_add",
    )


def _matmul_job() -> KernelJob:
    """Create a KernelJob for the 128x128x256 matmul kernel."""
    return KernelJob(
        source=MATMUL_256_KERNEL,
        input_specs={"a": ((128, 128), "bfloat16"), "b": ((128, 256), "bfloat16")},
        golden_source=GOLDEN_MATMUL,
        golden_func_name="golden_matmul",
        atol=0.5,
        rtol=0.1,
    )


def _timed_profile(hosts: list[str], kernels: dict[str, KernelJob]) -> tuple[float, list]:
    """Run profiler.profile() and return (elapsed_seconds, results)."""
    profiler = _make_profiler(hosts=hosts)
    t0 = time.monotonic()
    results = profiler.profile(kernels)
    return time.monotonic() - t0, results


class TestRemoteProfileSingleWorker:
    """Profile NKI kernels using a single remote worker."""

    def test_tensor_copy(self) -> None:
        """Profile a simple tensor copy kernel on 1 worker."""
        profiler = _make_profiler(hosts=["gym-1"])
        results = profiler.profile({"copy_v0.py": _copy_job()})
        assert len(results) == 1
        r = results[0]
        assert r.kernel_name == "copy_v0.py"
        assert r.hardware_run == "", f"Kernel errored: {r.hardware_run}"
        assert r.cpu_sim.startswith("PASS"), f"CPU sim failed: {r.cpu_sim}"
        assert r.min_ms > 0
        assert r.mean_ms >= r.min_ms
        assert r.p50_ms > 0
        assert r.p99_ms >= r.p50_ms

    def test_tensor_add_with_correctness(self) -> None:
        """Profile tensor add and verify cpu_sim passes against golden."""
        profiler = _make_profiler(hosts=["gym-1"])
        results = profiler.profile({"add_v0.py": _add_job()})
        assert len(results) == 1
        r = results[0]
        assert r.hardware_run == "", f"Kernel errored: {r.hardware_run}"
        assert r.cpu_sim.startswith("PASS"), f"CPU sim failed: {r.cpu_sim}"

    def test_matmul_with_mfu(self) -> None:
        """Profile matmul and check MFU is reported."""
        profiler = _make_profiler(hosts=["gym-1"])
        results = profiler.profile({"matmul_v0.py": _matmul_job()})
        assert len(results) == 1
        r = results[0]
        assert r.hardware_run == "", f"Kernel errored: {r.hardware_run}"
        assert r.cpu_sim.startswith("PASS"), f"CPU sim failed: {r.cpu_sim}"
        assert r.mfu > 0, "MFU should be > 0 for matmul"
        assert r.mac_count == 128 * 128 * 256

    def test_multiple_kernels_one_worker(self) -> None:
        """Send multiple kernels to a single worker."""
        profiler = _make_profiler(hosts=["gym-1"])
        kernels = {"copy_v0.py": _copy_job(), "add_v0.py": _add_job()}
        results = profiler.profile(kernels)
        assert len(results) == 2
        names = {r.kernel_name for r in results}
        assert names == {"copy_v0.py", "add_v0.py"}


class TestRemoteProfileMultiWorker:
    """Profile NKI kernels across multiple remote workers."""

    def test_parallel_speedup(self) -> None:
        """5 copy kernels on 1 vs 5 workers; multi-worker should be faster."""
        all_hosts = REMOTE_CONFIG["hosts"]
        num_hosts = len(all_hosts)
        kernels = {f"copy_v{i}.py": _copy_job() for i in range(num_hosts)}

        time_1, results_1 = _timed_profile([all_hosts[0]], kernels)
        time_n, results_n = _timed_profile(all_hosts, kernels)

        assert len([r for r in results_1 if not r.hardware_run]) == num_hosts
        assert len([r for r in results_n if not r.hardware_run]) == num_hosts
        print(f"\n  1 worker: {time_1:.1f}s, {num_hosts} workers: {time_n:.1f}s")
        assert time_n < time_1, f"Multi-worker ({time_n:.1f}s) not faster than single ({time_1:.1f}s)"

    def test_results_consistency(self) -> None:
        """Timing from different workers should be within 10x of each other."""
        all_hosts = REMOTE_CONFIG["hosts"]
        num_hosts = len(all_hosts)
        kernels = {f"add_v{i}.py": _add_job() for i in range(num_hosts)}
        profiler = _make_profiler(hosts=all_hosts)
        results = profiler.profile(kernels)
        successful = [r for r in results if not r.hardware_run]
        assert len(successful) == num_hosts
        times = [r.min_ms for r in successful]
        assert max(times) / min(times) < 10, f"Timing spread too large: {times}"

    def test_correctness_across_workers(self) -> None:
        """All workers should pass cpu_sim checks."""
        all_hosts = REMOTE_CONFIG["hosts"]
        kernels = {f"add_v{i}.py": _add_job() for i in range(len(all_hosts))}
        profiler = _make_profiler(hosts=all_hosts)
        results = profiler.profile(kernels)
        for r in results:
            assert r.hardware_run == "", f"{r.kernel_name} errored: {r.hardware_run}"
            assert r.cpu_sim.startswith("PASS"), f"{r.kernel_name} cpu_sim failed: {r.cpu_sim}"

    def test_100_kernels_speedup(self) -> None:
        """100 copy kernels on 1 vs 5 workers; multi-worker should be faster."""
        num_kernels = 100
        all_hosts = REMOTE_CONFIG["hosts"]
        kernels = {f"copy_v{i}.py": _copy_job() for i in range(num_kernels)}

        time_1, results_1 = _timed_profile([all_hosts[0]], kernels)
        time_n, results_n = _timed_profile(all_hosts, kernels)

        assert len([r for r in results_1 if not r.hardware_run]) == num_kernels
        assert len([r for r in results_n if not r.hardware_run]) == num_kernels
        print(f"\n  1 worker: {time_1:.1f}s, {len(all_hosts)} workers: {time_n:.1f}s")
        assert time_n < time_1, f"Multi-worker ({time_n:.1f}s) not faster than single ({time_1:.1f}s)"


class TestRemoteProfileErrorHandling:
    """Test error handling in the remote profiling backend."""

    def test_invalid_kernel_source(self) -> None:
        """An invalid kernel should produce an error result, not crash."""
        bad_kernel = """\
import nki
import nki.language as nl

@nki.jit
def bad_func(a):
    output = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.shared_hbm)
    this is not valid python
"""
        profiler = _make_profiler(hosts=["gym-1"])
        results = profiler.profile(
            {
                "bad_v0.py": KernelJob(
                    source=bad_kernel,
                    input_specs={"a": ((128, 512), "bfloat16")},
                    golden_source=GOLDEN_COPY,
                    golden_func_name="golden_copy",
                )
            }
        )
        assert len(results) == 1
        r = results[0]
        assert r.hardware_run != "" or r.cpu_sim.startswith("FAIL"), "Expected error for invalid kernel"

    def test_empty_kernels(self) -> None:
        """Empty kernel dict should return empty results."""
        profiler = _make_profiler(hosts=["gym-1"])
        results = profiler.profile({})
        assert results == []
