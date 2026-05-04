"""Hardware benchmarking and CPU simulation for NKI kernels.

Worker-only module — imports nki and nkipy at top level. Not safe to
import on the coordinator machine.
"""

import base64
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import nki
import numpy as np
from nki.compiler.ncc_driver import extract_perf_metrics
from nkipy.core.backend.hlo import HLOModule, HLOTensor
from nkipy.runtime import BaremetalExecutor, CompiledKernel

from autotune.runner.compile import load_kernel
from autotune.runner.types import (
    CompileResult,
    OutputSpec,
    ProfileResult,
    _sim_not_run,
    capture_error,
    resolve_dtype,
    tensor_inputs,
)


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
    cpu_sim: dict | None = None,
    collect_detailed_profile: bool = False,
) -> ProfileResult:
    """Run the compiled kernel once with ``save_trace=True`` and attach the
    profiler's JSON output(s) to a :class:`ProfileResult`.

    Correctness is already gated by CPU sim upstream; wall-clock timing
    and utilization come entirely from the profiler summary. When
    ``collect_detailed_profile`` is ``True`` the full per-instruction
    trace is also attached as ``profile_detailed``.
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
        cpu_sim=cpu_sim if cpu_sim is not None else _sim_not_run(),
        hardware_output=hardware_output,
        profiler_summary=profiler_summary,
        profile_detailed=profile_detailed,
        neff_b64=neff_b64,
        ntff_b64=ntff_b64,
    )


def simulate_one(nki_path: str, func_name: str, kernel_kwargs: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Run a kernel through the NKI CPU simulator in float32 end-to-end.

    Everything runs at float32: inputs are cast, and the kernel source
    is rewritten to a sibling ``.sim.py`` so every ``dtype=nl.<name>``
    token declares ``nl.float32``. This keeps the numerical check
    apples-to-apples with ``compute_golden()`` and bypasses
    ``nisa.dma_transpose``'s strict dtype assert — the hardware path
    still reads ``nki_path`` with the user's declared dtypes.

    Args:
        nki_path: Path to the kernel source file.
        func_name: Name of the @nki.jit decorated function.
        kernel_kwargs: Input tensors and scalar params.

    Returns:
        Tuple of (output array, error string). Output is empty on failure.
    """
    output = np.empty(0)
    error = ""
    try:
        sim_path = _rewrite_to_fp32(nki_path)
        kernel_func = load_kernel(sim_path, func_name)
        sim_kwargs: dict[str, Any] = {}
        for k, v in kernel_kwargs.items():
            if hasattr(v, "ndim") and v.ndim > 0:
                sim_kwargs[k] = v.astype(np.float32)
            else:
                sim_kwargs[k] = v
        result = nki.simulate(kernel_func)(**sim_kwargs)
        output = result[0] if isinstance(result, tuple) else result
    except Exception as e:
        error = capture_error(e)
    return output, error


def _rewrite_to_fp32(nki_path: str) -> str:
    """Return a sibling ``.sim.py`` with every ``dtype=nl.<name>`` forced to ``nl.float32``.

    Targets the codegen's dtype-bearing tokens without touching
    unrelated ``nl.*`` names (``nl.ndarray``, ``nl.sbuf``,
    ``nl.psum``, ``nl.shared_hbm``, etc.).
    """
    source = Path(nki_path).read_text()
    rewritten = re.sub(r"dtype=nl\.\w+", "dtype=nl.float32", source)
    sim_path = Path(nki_path).with_suffix(".sim.py")
    sim_path.write_text(rewritten)
    return str(sim_path)


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


def compute_golden(nkigym_source: str, func_name: str, kernel_kwargs: dict[str, np.ndarray]) -> np.ndarray:
    """Execute the nkigym math function as the golden reference, in float32.

    The source carries its own imports (``from nkigym.ops.matmul
    import NKIMatmul`` etc.). Each ``NKIOp`` has a pure-numpy
    ``__call__`` — invoking the math function therefore runs a
    numpy simulation of the kernel at float32. Inputs are cast to
    float32 to match ``simulate_one``'s fp32-end-to-end path.

    Args:
        nkigym_source: Source code of the nkigym math function.
        func_name: Name of the nkigym function within the source.
        kernel_kwargs: Input arrays (will be cast to float32).

    Returns:
        Expected output at float32 precision.
    """
    g: dict[str, Any] = {"__builtins__": __builtins__, "__name__": "__nkigym_golden__"}
    exec(nkigym_source, g)  # noqa: S102
    func = g[func_name]
    f32_kwargs = {k: v.astype(np.float32) for k, v in kernel_kwargs.items() if hasattr(v, "ndim") and v.ndim > 0}
    return func(**f32_kwargs)
