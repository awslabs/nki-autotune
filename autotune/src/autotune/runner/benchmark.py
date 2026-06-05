"""Hardware benchmarking for NKI kernels.

On-box module — imports nki and nkipy at top level. Requires a Trainium
host with nki + nkipy installed.
"""

import base64
import json
import os
import subprocess
import tempfile
from typing import Any

import numpy as np
from nki.compiler.ncc_driver import extract_perf_metrics
from nkipy.core.backend.hlo import HLOModule, HLOTensor
from nkipy.runtime import BaremetalExecutor, CompiledKernel

from autotune.runner.compile import load_kernel
from autotune.runner.types import CompileResult, OutputSpec, ProfileResult, capture_error, resolve_dtype, tensor_inputs


class TracedKernel:
    """Minimal traced kernel wrapper required by CompiledKernel."""

    def __init__(self, func: Any, code: Any) -> None:
        """Initialize with kernel function and HLO module."""
        self.func = func
        self._code = code

    @property
    def __name__(self) -> str:
        """Return the kernel function name."""
        return self.func.__name__


def create_compiled_kernel(
    neff_path: str, nki_path: str, func_name: str, kernel_kwargs: dict[str, Any], out: OutputSpec
) -> CompiledKernel:
    """Create a CompiledKernel from a NEFF for BaremetalExecutor.

    Args:
        neff_path: Path to the compiled NEFF binary.
        nki_path: Path to the kernel source file.
        func_name: Name of the kernel function.
        kernel_kwargs: Input tensors and scalar params.
        out: Output tensor specification.
    """
    kernel = load_kernel(nki_path, func_name)
    hlo = HLOModule(name=func_name)
    for name, tensor in kernel_kwargs.items():
        if hasattr(tensor, "ndim") and tensor.ndim > 0:
            hlo.add_parameter(tensor.shape, tensor.dtype, name=name)
    hlo.set_results([HLOTensor(shape=out.shape, dtype=out.dtype, name=out.name)])
    return CompiledKernel(TracedKernel(kernel, hlo), neff_path)


def _collect_profiler_outputs(
    spike: BaremetalExecutor, compiled: CompiledKernel, kernel_kwargs: dict[str, Any], collect_detailed: bool
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Run the kernel once with ``save_trace=True`` and parse the NTFF.

    Returns ``(summary, detailed, ntff_b64)`` where:

    * ``summary`` = the ``neuron-profile view --output-format summary-json``
      dict (~100-key aggregate, always collected).
    * ``detailed`` = the ``--output-format json`` dict (per-instruction
      trace, per-engine active intervals, per-layer summaries) when
      ``collect_detailed`` is ``True``; else ``None``. Tens of MB.
    * ``ntff_b64`` = base64-encoded NTFF bytes when ``collect_detailed``
      is ``True``; else ``None``. Callers can decode and re-feed into
      ``neuron-profile view`` offline.
    """
    detailed: dict[str, Any] | None = None
    ntff_b64: str | None = None
    with tempfile.TemporaryDirectory(prefix="autotune-profile-") as artifacts_dir:
        try:
            spike.run(compiled, *tensor_inputs(kernel_kwargs), save_trace=True, artifacts_dir=artifacts_dir)
            ntff_path = os.path.join(artifacts_dir, "profile.ntff")
            summary = extract_perf_metrics(compiled.compiled_artifact, ntff_path, verbose=False)
            if collect_detailed:
                detailed = _extract_detailed_profile(compiled.compiled_artifact, ntff_path, artifacts_dir)
                with open(ntff_path, "rb") as f:
                    ntff_b64 = base64.b64encode(f.read()).decode("ascii")
        except Exception as e:
            raise RuntimeError(f"Profiler summary collection failed: {e}") from e
    return summary, detailed, ntff_b64


def _extract_detailed_profile(neff_path: str, ntff_path: str, work_dir: str) -> dict[str, Any] | None:
    """Invoke ``neuron-profile view --output-format json`` and load result."""
    out_json = os.path.join(work_dir, "profile_detailed.json")
    result = subprocess.run(
        [
            "neuron-profile",
            "view",
            "-n",
            neff_path,
            "-s",
            ntff_path,
            "--output-format",
            "json",
            "--output-file",
            out_json,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    with open(out_json) as f:
        return json.load(f)


def benchmark_one(
    spike: BaremetalExecutor,
    cr: CompileResult,
    func_name: str,
    kernel_kwargs: dict[str, Any],
    out: OutputSpec,
    collect_detailed_profile: bool = False,
) -> ProfileResult:
    """Run the compiled kernel once with ``save_trace=True`` and attach the
    profiler's JSON output(s) to a :class:`ProfileResult`.

    Wall-clock timing and utilization come entirely from the profiler
    summary. When ``collect_detailed_profile`` is ``True`` the full
    per-instruction trace is also attached as ``profile_detailed``.
    """
    profiler_summary: dict[str, Any] | None = None
    profile_detailed: dict[str, Any] | None = None
    neff_b64: str | None = None
    ntff_b64: str | None = None
    hardware_output = f"{list(out.shape)} {out.dtype}"
    try:
        compiled = create_compiled_kernel(cr.neff_path, cr.nki_path, func_name, kernel_kwargs, out)
        profiler_summary, profile_detailed, ntff_b64 = _collect_profiler_outputs(
            spike, compiled, kernel_kwargs, collect_detailed_profile
        )
        if collect_detailed_profile:
            with open(cr.neff_path, "rb") as f:
                neff_b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as e:
        hardware_output = capture_error(e)
    return ProfileResult(
        kernel_name=cr.kernel_name,
        hardware_output=hardware_output,
        profiler_summary=profiler_summary,
        profile_detailed=profile_detailed,
        neff_b64=neff_b64,
        ntff_b64=ntff_b64,
    )


def generate_tensors(tensor_specs: dict[str, dict], seed: int) -> dict[str, np.ndarray]:
    """Generate random input tensors from shapes, dtypes, and a seed.

    Args:
        tensor_specs: Map of param name to {"shape": [...], "dtype": "..."}.
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    tensors: dict[str, np.ndarray] = {}
    for name, spec in tensor_specs.items():
        shape = tuple(spec["shape"])
        dtype = resolve_dtype(spec["dtype"])
        tensors[name] = rng.standard_normal(shape).astype(dtype)
    return tensors
