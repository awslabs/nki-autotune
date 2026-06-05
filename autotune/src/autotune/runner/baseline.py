"""Compile and benchmark a plain numpy function on Trainium as a baseline.

nkipy's ``baremetal_jit`` path (numpy -> HLO -> neuronx-cc -> NEFF) is
the zero-NKI reference for any tuned kernel. ``profile_numpy_baseline``
runs that pipeline and returns a ``ProfileResult`` with the same
timing / MFU / roofline columns as ``benchmark.benchmark_one``, so the
baseline lines up next to autotune-generated NKI kernels in any
comparison table.

On-box module -- imports nkipy at top level and requires the Neuron
runtime via ``BaremetalExecutor``. Must run on a Trainium host; to run
from a laptop, drive it through a transport shell (see AGENTS.md).
"""

import base64
import logging
import os
import shutil
from typing import Any, Callable

import numpy as np
from nkipy.core.compile import CompilationTarget, compile_to_neff, lower_to_nki, trace
from nkipy.runtime import BaremetalExecutor, CompiledKernel

logger = logging.getLogger(__name__)

from autotune.runner.benchmark import _collect_profiler_outputs, generate_tensors
from autotune.runner.types import ProfileResult, capture_error


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
    output_dir: str,
    kernel_name: str = "baseline",
    additional_compiler_args: str = "--lnc 1 --internal-tensorizer-opt-level=2",
    collect_detailed_profile: bool = False,
) -> tuple[ProfileResult, str]:
    """Compile a numpy function via nkipy + neuronx-cc and benchmark on Trainium.

    Returns a :class:`ProfileResult` with the post-compiler
    ``profiler_summary`` so the baseline slots into any comparison
    table of tuned NKI kernels. Must be run on a Trainium host --
    ``BaremetalExecutor`` requires the Neuron runtime.

    ``input_specs`` iteration order MUST match the positional parameter
    order of ``func``; Python dicts preserve insertion order, so callers
    that build the dict in the natural argument order are fine.

    Args:
        func: Numpy function to profile, e.g.
            ``def mm(a, b): return a @ b``.
        input_specs: Map of parameter name to ``(shape, dtype_str)``.
        output_dir: Directory for compilation artifacts (NEFF, HLO, logs).
        kernel_name: Identifier stored on the resulting ``ProfileResult``.
        additional_compiler_args: Forwarded to neuronx-cc. The default
            matches nkipy's ``baremetal_jit`` baseline for an NKIPy
            kernel (``--lnc 1 --internal-tensorizer-opt-level=2``), so
            the returned profiler summary is an apples-to-apples
            reference and the produced NEFF loads against the same
            1-core config the rest of the pipeline uses.

    Returns:
        A ``(ProfileResult, nki_source)`` pair. ``nki_source`` is the
        NKI text emitted by the compiler's tensorizer (``lower_to_nki``);
        empty on failure.
    """
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
    tensor_specs = {name: {"shape": list(shape), "dtype": dt} for name, (shape, dt) in input_specs.items()}
    kernel_kwargs = generate_tensors(tensor_specs, seed=42)

    profiler_summary: dict[str, Any] | None = None
    profile_detailed: dict[str, Any] | None = None
    neff_b64: str | None = None
    ntff_b64: str | None = None
    hardware_output = "baseline"
    nki_source = ""
    try:
        compiled, nki_source = _compile_numpy_fn(func, kernel_kwargs, output_dir, additional_compiler_args)
        with BaremetalExecutor(verbose=0) as spike:
            profiler_summary, profile_detailed, ntff_b64 = _collect_profiler_outputs(
                spike, compiled, kernel_kwargs, collect_detailed_profile
            )
        if collect_detailed_profile:
            with open(compiled.compiled_artifact, "rb") as f:
                neff_b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as e:
        hardware_output = capture_error(e)
    result = ProfileResult(
        kernel_name=kernel_name,
        hardware_output=hardware_output,
        profiler_summary=profiler_summary,
        profile_detailed=profile_detailed,
        neff_b64=neff_b64,
        ntff_b64=ntff_b64,
    )
    return result, nki_source
