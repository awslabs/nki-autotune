"""Compile and benchmark a plain numpy function on Trainium as a baseline.

nkipy's ``baremetal_jit`` path (numpy -> HLO -> neuronx-cc -> NEFF) is
the zero-NKI reference for any tuned kernel. ``profile_numpy_baseline``
runs that pipeline and returns a ``ProfileResult`` with the same
timing / MFU / roofline columns as ``benchmark.benchmark_one``, so the
baseline lines up next to autotune-generated NKI kernels in any
comparison table.

Worker-only module -- imports nkipy at top level and requires the
Neuron runtime via ``BaremetalExecutor``. Callers on non-Trn hosts
should use :func:`remote_numpy_baseline` instead, which SSHs a
coordinator-side payload into this module's ``baseline_worker_main``
over the existing worker-bundle transport.
"""

import json
import logging
import os
import shutil
import sys
from typing import Any, Callable

import numpy as np
from nkipy.core.compile import CompilationTarget, compile_to_neff, lower_to_nki, trace
from nkipy.runtime import BaremetalExecutor, CompiledKernel

logger = logging.getLogger(__name__)

from autotune.runner.benchmark import _collect_roofline, _run_timing, generate_tensors
from autotune.runner.types import (
    BenchmarkConfig,
    ProfileResult,
    RooflineMetrics,
    _sim_not_run,
    capture_error,
    ensure_venv_on_path,
)


def _compile_numpy_fn(
    func: Callable[..., np.ndarray],
    kernel_kwargs: dict[str, np.ndarray],
    output_dir: str,
    additional_compiler_args: str,
) -> tuple[CompiledKernel, str]:
    """Trace ``func`` against ``kernel_kwargs`` and compile the HLO to NEFF.

    Mirrors the first half of
    ``nkipy.runtime.execute.baremetal_run_traced_kernel``
    (``specialize`` + ``compile_to_neff``) but stops short of hardware
    execution so the caller can reuse the NEFF across the timing and
    roofline passes.

    Also invokes ``lower_to_nki`` on the same traced kernel, which runs
    the compiler's ``tensorize`` pipeline with ``--print-nki`` to emit
    the post-tensorizer NKI source — returned alongside the NEFF so the
    caller can record the compiler-generated kernel next to the tuned
    NKI variants in the same cache layout.
    """
    traced = trace(func)
    traced.specialize(**kernel_kwargs)
    neff_path = compile_to_neff(
        trace_kernel=traced,
        output_dir=output_dir,
        neff_name=f"{traced.__name__}.neff",
        save_artifacts=True,
        additional_compiler_args=additional_compiler_args,
        target=CompilationTarget.TRN2,
    )
    try:
        nki_source = lower_to_nki(
            trace_kernel=traced,
            output_dir=os.path.join(output_dir, "nki_lowering"),
            target=CompilationTarget.TRN2,
            additional_compiler_args=additional_compiler_args,
            save_artifacts=True,
        )
    except RuntimeError as e:
        logger.warning("lower_to_nki failed: %s", e)
        nki_source = ""
    return CompiledKernel(traced, neff_path), nki_source


def profile_numpy_baseline(
    func: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    mac_count: int,
    output_dir: str,
    kernel_name: str = "baseline",
    additional_compiler_args: str = "--lnc 1 --internal-tensorizer-opt-level=2",
) -> tuple[ProfileResult, str]:
    """Compile a numpy function via nkipy + neuronx-cc and benchmark on Trainium.

    Produces the same timing / MFU / roofline fields as ``benchmark_one``
    so the baseline slots into any comparison table of tuned NKI kernels.
    Must be run on a Trainium host -- ``BaremetalExecutor`` requires the
    Neuron runtime.

    ``input_specs`` iteration order MUST match the positional parameter
    order of ``func``; Python dicts preserve insertion order, so callers
    that build the dict in the natural argument order are fine.

    Args:
        func: Numpy function to profile, e.g.
            ``def mm(a, b): return a @ b``.
        input_specs: Map of parameter name to ``(shape, dtype_str)``.
        mac_count: Theoretical MAC count for MFU computation.
        output_dir: Directory for compilation artifacts (NEFF, HLO, logs).
        kernel_name: Identifier stored on the resulting ``ProfileResult``.
        additional_compiler_args: Forwarded to neuronx-cc. The default
            matches nkipy's ``baremetal_jit`` baseline for an NKIPy
            kernel (``--lnc 1 --internal-tensorizer-opt-level=2``), so
            the returned MFU is an apples-to-apples reference and the
            produced NEFF loads against the same 1-core config the rest
            of the pipeline uses.

    Returns:
        A ``(ProfileResult, nki_source)`` pair. The ``ProfileResult``'s
        ``cpu_sim`` field is marked "not run" -- the numpy function *is*
        the golden reference, so there is nothing to compare against.
        ``nki_source`` is the NKI text emitted by the compiler's
        tensorizer (``lower_to_nki``); empty on failure.
    """
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
    tensor_specs = {name: {"shape": list(shape), "dtype": dt} for name, (shape, dt) in input_specs.items()}
    kernel_kwargs = generate_tensors(tensor_specs, seed=42)
    first_dtype = next(iter(input_specs.values()))[1]
    config = BenchmarkConfig(warmup=10, iters=100, mac_count=mac_count, input_dtype_name=first_dtype)

    min_ms: float | None = None
    mean_ms: float | None = None
    p50_ms: float | None = None
    p99_ms: float | None = None
    mfu: float | None = None
    roofline = RooflineMetrics(
        mbu_estimated_percent=None, mfu_max_achievable_estimated_percent=None, roofline_efficiency=None
    )
    hardware_output = "baseline"
    nki_source = ""
    try:
        compiled, nki_source = _compile_numpy_fn(func, kernel_kwargs, output_dir, additional_compiler_args)
        with BaremetalExecutor(verbose=0) as spike:
            min_ms, mean_ms, p50_ms, p99_ms, mfu = _run_timing(spike, compiled, kernel_kwargs, config)
            roofline = _collect_roofline(spike, compiled, kernel_kwargs, mfu)
    except Exception as e:
        hardware_output = capture_error(e)
    result = ProfileResult(
        kernel_name=kernel_name,
        min_ms=min_ms,
        mean_ms=mean_ms,
        p50_ms=p50_ms,
        p99_ms=p99_ms,
        mac_count=mac_count,
        mfu=mfu,
        cpu_sim=_sim_not_run(),
        hardware_output=hardware_output,
        mbu_estimated_percent=roofline.mbu_estimated_percent,
        mfu_max_achievable_estimated_percent=roofline.mfu_max_achievable_estimated_percent,
        roofline_efficiency=roofline.roofline_efficiency,
    )
    return result, nki_source


def baseline_worker_main() -> None:
    """Remote worker entry point for numpy-baseline profiling.

    Mirrors ``worker.worker_main``: reads a JSON payload from stdin,
    exec()s the caller-supplied source to resolve the numpy function,
    runs ``profile_numpy_baseline``, writes a JSON ``ProfileResult`` to
    stdout. Stderr holds log lines.

    Payload schema::

        {
          "host": str,
          "neuron_platform_target": str,
          "func_source": str,     # exec'd in a fresh namespace
          "func_name": str,       # callable to lookup in that namespace
          "input_specs": {name: [[shape...], dtype_str]},
          "mac_count": int,
          "kernel_name": str,
          "additional_compiler_args": str,
        }
    """
    payload: dict[str, Any] = json.loads(sys.stdin.buffer.read())
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = payload["neuron_platform_target"]
    ensure_venv_on_path()

    namespace: dict[str, Any] = {"__builtins__": __builtins__, "__name__": "__baseline_source__"}
    exec(payload["func_source"], namespace)  # noqa: S102
    func = namespace[payload["func_name"]]

    input_specs: dict[str, tuple[tuple[int, ...], str]] = {
        name: (tuple(shape), dt) for name, (shape, dt) in payload["input_specs"].items()
    }

    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        result, nki_source = profile_numpy_baseline(
            func=func,
            input_specs=input_specs,
            mac_count=payload["mac_count"],
            output_dir=f"/tmp/autotune-baseline-{payload['kernel_name']}",
            kernel_name=payload["kernel_name"],
            additional_compiler_args=payload["additional_compiler_args"],
        )
    finally:
        sys.stdout = real_stdout

    sys.stdout.buffer.write(
        json.dumps({"host": payload["host"], "result": result._asdict(), "nki_source": nki_source}).encode("utf-8")
    )
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    baseline_worker_main()
