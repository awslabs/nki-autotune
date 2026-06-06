"""Profile the 90.92%-MFU hand matmul on Trn2 to confirm it reproduces and
to read its engine-active / DMA-overlap breakdown.

Reads ``kernel_library/matmul/lhsT_rhs/kernel_hand_90.92mfu.py`` verbatim as
the kernel source (helpers + the ``@nki.jit`` entry), CPU-sim's it at fp32
against the numpy golden, then profiles on Trn2 next to the neuronx-cc
compiler baseline. The point is the raw ``tensor_engine_active_time_percent``
and DMA fields — to quantify how much of the 84%->91% gap (vs our Tier-B
double-buffer kernel) is a prologue-load bubble vs matmul efficiency.

Usage::

    source ~/venvs/kernel-env/bin/activate
    PYTHONPATH=.:nkigym/src:autotune/src \
        python examples/profile_hand_90mfu.py --cache /abs/path
"""

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "nkigym", "src"), os.path.join(_REPO_ROOT, "autotune", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from autotune.runner.api import profile
from autotune.runner.baseline import profile_numpy_baseline
from autotune.runner.types import KernelJob, ProfileResult, profiler_percent
from nkigym.synthesis.simulate_nki import simulate_fp32

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
KERNEL_PATH = os.path.join(_REPO_ROOT, "kernel_library", "matmul", "lhsT_rhs", "kernel_hand_90.92mfu.py")
RENDERED_FUNC_NAME = "matmul_lhsT_rhs_nkigym"
SEED = 0
NEURON_PLATFORM_TARGET = "trn2"


def _sim_check(source: str, scratch_path: str, atol: float = 5e-3, rtol: float = 5e-3) -> None:
    """CPU-sim ``source`` at fp32 and assert it matches ``lhs_T.T @ rhs``."""
    rng = np.random.default_rng(SEED)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    with open(scratch_path, "w", encoding="utf-8") as handle:
        handle.write(source)
    spec = importlib.util.spec_from_file_location("dumped_kernel", scratch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = np.asarray(simulate_fp32(getattr(module, RENDERED_FUNC_NAME))(**inputs))
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def _numpy_matmul(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """``lhs_T.T @ rhs`` — the compiler-baseline reference."""
    return lhs_T.T @ rhs


def _print_baseline(baseline: ProfileResult) -> None:
    """Print the compiler baseline below the hand-kernel row, same scale."""
    base_mfu = profiler_percent(baseline.profiler_summary, "mfu_estimated_percent")
    base_time = (baseline.profiler_summary or {}).get("total_time")
    if base_mfu is None or not isinstance(base_time, (int, float)):
        print(f"\n[hand90] compiler baseline unavailable: {baseline.hardware_output.splitlines()[0]}")
    else:
        print(f"  compiler baseline: {float(base_time):.6f} s   {base_mfu:.2f}% MFU")


def main() -> None:
    """Sim-validate the hand kernel, profile it + the compiler baseline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--detailed", action="store_true", help="collect per-instruction trace + NEFF/NTFF")
    args = parser.parse_args()

    cache_dir = os.path.join(args.cache, "profile_hand_90mfu")
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)
    scratch_path = os.path.join(tempfile.gettempdir(), "hand90_sim_scratch.py")

    with open(KERNEL_PATH, encoding="utf-8") as handle:
        source = handle.read()
    _sim_check(source, scratch_path)
    print("[hand90] kernel_hand_90.92mfu: sim PASS")

    """Per kernel_library/README.md, this kernel hand-places PSUM allocation
    and REQUIRES the neuronx-cc scheduler + linear-scan allocator OFF;
    scheduler-on OOMs PSUM even with per-iteration alloc."""
    neuronx_cc_args = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")
    jobs = {
        "hand_90mfu": KernelJob(
            source=source,
            func_name=RENDERED_FUNC_NAME,
            output_shape=(M, N),
            input_specs=INPUT_SPECS,
            neuronx_cc_args=neuronx_cc_args,
        )
    }
    print(f"\n[hand90] profiling on {NEURON_PLATFORM_TARGET} ...\n")
    output = profile(
        jobs,
        cache_dir=cache_dir,
        seed=SEED,
        neuron_platform_target=NEURON_PLATFORM_TARGET,
        collect_detailed_profile=args.detailed,
    )
    print(output)

    print(f"\n[hand90] compiling numpy baseline via neuronx-cc on {NEURON_PLATFORM_TARGET} ...")
    baseline, _nki_source = profile_numpy_baseline(
        _numpy_matmul, INPUT_SPECS, os.path.join(cache_dir, "compiler_baseline"), kernel_name="compiler_baseline"
    )
    _print_baseline(baseline)


if __name__ == "__main__":
    main()
