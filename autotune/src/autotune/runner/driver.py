"""In-process NKI kernel compile + benchmark pipeline.

Runs ON a Trn2 box. Given a set of KernelJobs, compiles each to NEFF
(parallel ProcessPool), benchmarks on Neuron hardware, and returns
per-kernel ProfileResults plus compiler logs. No SSH, no bundling — the
whole driver runs in-process on the box where nki + nkipy are installed.
"""

import logging
import os
import shutil
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from nkipy.runtime import BaremetalExecutor

from autotune.runner.benchmark import benchmark_one, generate_tensors
from autotune.runner.compile import compile_one, init_compile_worker
from autotune.runner.detect import detect_neuron_cores
from autotune.runner.types import (
    CompileResult,
    KernelJob,
    OutputSpec,
    ProfileResult,
    compile_failure_result,
    resolve_dtype,
)

logger = logging.getLogger(__name__)

_OUTPUT_TENSOR_NAME = "hbm_tensor_0"


def _prepare_kernel(kname: str, job: KernelJob, seed: int, nki_dir: Path) -> dict[str, Any]:
    """Write a kernel's source to disk and generate its input tensors."""
    filename = kname if kname.endswith(".py") else f"{kname}.py"
    nki_path = nki_dir / filename
    nki_path.write_text(job.source)

    tensor_specs = {name: {"shape": list(shape), "dtype": dt} for name, (shape, dt) in job.input_specs.items()}
    kwargs = generate_tensors(tensor_specs, seed)
    input_dtype_name = next(iter(job.input_specs.values()))[1]
    return {
        "nki_path": str(nki_path),
        "func_name": job.func_name,
        "output_shape": tuple(job.output_shape),
        "kwargs": kwargs,
        "input_dtype_name": input_dtype_name,
        "neuronx_cc_args": tuple(job.neuronx_cc_args),
        "lnc": int(job.lnc),
    }


def _submit_compilations(
    kernel_data: dict[str, dict[str, Any]], neff_dir: Path
) -> tuple[ProcessPoolExecutor, list[Future]]:
    """Submit all kernels for parallel compilation."""
    cpu_cores = os.cpu_count() or 1
    compile_workers = min(max(cpu_cores - 1, 1), len(kernel_data))
    executor = ProcessPoolExecutor(max_workers=compile_workers, initializer=init_compile_worker)
    futures: list[Future] = []
    for kname, kd in kernel_data.items():
        compile_dir = neff_dir / Path(kname).stem
        compile_dir.mkdir(parents=True, exist_ok=True)
        input_shapes = {k: v.shape for k, v in kd["kwargs"].items() if hasattr(v, "ndim") and v.ndim > 0}
        futures.append(
            executor.submit(
                compile_one,
                kname,
                kd["nki_path"],
                kd["func_name"],
                input_shapes,
                kd["input_dtype_name"],
                _OUTPUT_TENSOR_NAME,
                kd["output_shape"],
                kd["input_dtype_name"],
                str(compile_dir),
                {},
                kd["neuronx_cc_args"],
                kd["lnc"],
            )
        )
    return executor, futures


def _benchmark_compiled(
    compile_futures: list[Future],
    spike: BaremetalExecutor,
    kernel_data: dict[str, dict[str, Any]],
    collect_detailed_profile: bool,
) -> tuple[list[CompileResult], list[ProfileResult], list[ProfileResult]]:
    """Benchmark each kernel as it finishes compiling."""
    compile_results: list[CompileResult] = []
    compile_errors: list[ProfileResult] = []
    hw_results: list[ProfileResult] = []
    for f in as_completed(compile_futures):
        cr = f.result()
        compile_results.append(cr)
        kd = kernel_data[cr.kernel_name]
        if cr.error:
            compile_errors.append(compile_failure_result(cr))
            continue
        out = OutputSpec(
            name=_OUTPUT_TENSOR_NAME, shape=kd["output_shape"], dtype=resolve_dtype(kd["input_dtype_name"])
        )
        hw_results.append(
            benchmark_one(
                spike, cr, kd["func_name"], kd["kwargs"], out, collect_detailed_profile=collect_detailed_profile
            )
        )
    return compile_results, compile_errors, hw_results


def _collect_compiler_logs(compile_results: list[CompileResult], neff_dir: Path, collect: bool) -> dict[str, str]:
    """Gather compiler log files if collection is enabled."""
    logs: dict[str, str] = {}
    for cr in compile_results if collect else []:
        stem = Path(cr.kernel_name).stem
        log_path = neff_dir / stem / "log-neuron-cc.txt"
        if log_path.exists():
            logs[cr.kernel_name] = log_path.read_text()
    return logs


def run_pipeline(
    kernels: dict[str, KernelJob],
    seed: int,
    collect_compiler_logs: bool,
    collect_detailed_profile: bool,
) -> tuple[list[ProfileResult], dict[str, str]]:
    """Compile + benchmark every kernel in-process on this box.

    Returns (results, compiler_logs).
    """
    lncs = {int(job.lnc) for job in kernels.values()}
    if len(lncs) > 1:
        raise RuntimeError(f"Batch mixes lnc values {lncs}; submit one lnc per profile() call")
    lnc = next(iter(lncs))
    os.environ["NEURON_RT_VISIBLE_CORES"] = "0" if lnc == 1 else "0,1"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(lnc)

    first_job = next(iter(kernels.values()))
    work_dir = Path(f"/tmp/autotune-{first_job.func_name}")
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    nki_dir = work_dir / "nki"
    nki_dir.mkdir()
    neff_dir = work_dir / "neff"
    neff_dir.mkdir()

    neuron_cores = detect_neuron_cores()
    logger.info("Driver ready: %d kernels, %d CPU, %d NC", len(kernels), os.cpu_count() or 1, neuron_cores)

    kernel_data = {kname: _prepare_kernel(kname, job, seed, nki_dir) for kname, job in kernels.items()}

    executor, futures = _submit_compilations(kernel_data, neff_dir)
    with BaremetalExecutor(verbose=0) as spike:
        compile_results, compile_errors, hw_results = _benchmark_compiled(
            futures, spike, kernel_data, collect_detailed_profile
        )
    executor.shutdown(wait=True)

    compiler_logs = _collect_compiler_logs(compile_results, neff_dir, collect_compiler_logs)
    return compile_errors + hw_results, compiler_logs
