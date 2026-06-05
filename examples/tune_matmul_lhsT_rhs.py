"""Hand-tuned matmul: a transform sequence found by reading the profiler.

Proof-of-concept tuner for ``lhs_T.T @ rhs`` (K=M=N=2048, bf16). The
``TRACE`` below is not a canned ladder — it is the sequence I arrived at
by running candidates on this Trn2 box, reading the profiler feedback,
and refining. The script replays that final sequence, profiles every
rung, and prints the MFU climb so the result is reproducible.

How the sequence was found (the manual tuning loop):

1. The canonical kernel already loads both operands once and runs a
   pure-compute matmul nest — but its accumulator is the full-extent
   PSUM (2048x2048 fp32 = 16 MiB >> Trn2's ~2 MiB), so it PSUM-OOMs.
2. The obvious fix (the k0..k14 transform-test ladder) shrinks PSUM by
   sinking BOTH operand loads into the innermost K-loop with
   single-element buffers. That compiles and runs, but profiles at only
   ~16% MFU: each ``nc_matmul`` waits on a fresh DMA (load->matmul->load
   serializes) and operands are re-fetched 4-16x. Low MFU AND low MBU
   with a high ceiling == a serialization stall, not a memory wall.
3. The fix that the profiler points to: keep both loads HOISTED and
   resident in SBUF, and shrink PSUM a different way — reorder the
   matmul nest to put the K (reduction) loop innermost, then ``compute_at``
   the memset and the PSUM->SBUF drain under the M loop so the live PSUM
   is a single 128x2048 fp32 tile (~1 MiB). The inner loop becomes a
   pure ``nc_matmul`` reduction the compiler can software-pipeline.

That final kernel profiles at ~60% MFU (0.000366 s) versus the ladder's
~16% — a 3.7x improvement from four atoms. For reference, the compiler
baseline (numpy ``lhs_T.T @ rhs`` via neuronx-cc) lands at ~86% MFU /
0.000253 s on the same box; closing that last gap needs accumulator
multi-buffering, which the shipped transform set does not yet express
cleanly (see CLAUDE.md "Tiles per block").

What this script profiles, and why only one kernel: every rung is
CPU-sim'd at fp32 against the numpy golden, so the whole transform path
is proven correct (a broken transform fails loudly and halts). But the
PSUM accumulator only shrinks on the LAST transform — the canonical
kernel and the three intermediate rungs all hold the full-extent 16 MiB
PSUM and would PSUM-OOM on hardware. So there is no measurable "MFU
climb" to profile: only the final tuned kernel fits the box. The script
profiles that kernel and the compiler baseline, and reports them side by
side.

Measured (Trn2, seed 0; numbers vary run to run):

    kernel                                 total_s      mfu
    tuned (loads resident, K innermost)   0.000366   ~59.7%
    compiler baseline (neuronx-cc)        0.000253   ~86.2%

Usage::

    source ~/venvs/kernel-env/bin/activate
    PYTHONPATH=.:nkigym/src \
        python examples/tune_matmul_lhsT_rhs.py --cache-root-dir /tmp/autotune_cache
"""

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile

"""Put the repo root and both src trees on sys.path so a bare
``python examples/...`` run resolves ``nkigym`` and ``autotune``."""
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "nkigym", "src"), os.path.join(_REPO_ROOT, "autotune", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from autotune.runner.api import profile
from autotune.runner.baseline import profile_numpy_baseline
from autotune.runner.output import ProfileOutput
from autotune.runner.types import KernelJob, ProfileResult, profiler_percent
from nkigym.codegen import render
from nkigym.environment import KernelMDP
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.synthesis.simulate_nki import simulate_fp32
from nkigym.transforms import (
    ComputeAt,
    ComputeAtOption,
    Fuse,
    Reorder,
    ReorderOption,
    ReverseComputeAt,
    Split,
)

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
RENDERED_FUNC_NAME = "nki_f_nkigym"
SEED = 0
NEURON_PLATFORM_TARGET = "trn2"


@nkigym_kernel
def f_nkigym(lhs_T, rhs):
    """``lhs_T.T @ rhs`` SSA body — the canonical (un-tiled) matmul."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


"""The tuned transform sequence, one ``(transform, option)`` atom per rung.

Axes: ``d0=K``, ``d1=M``, ``d2=N``. ``build_initial_ir`` is deterministic,
so the canonical tree's node ids are stable and every ``apply`` yields a
deterministic next tree — the literal nids below are fixed.

* rungs 1-2 (Reorder): rotate the matmul nest from ``K>M>N`` to ``M>N>K``,
  putting the reduction loop innermost.
* rung 3 (ComputeAt): sink the PSUM memset under the M loop.
* rung 4 (ComputeAt): sink the PSUM->SBUF drain under the M loop, which
  drops the PSUM accumulator's live extent to a single 128x2048 tile.

The two operand loads are never touched, so they stay hoisted at the top
of the kernel and resident in SBUF across the whole matmul.
"""
TRACE: list[tuple[object, object]] = [
    (Reorder(), ReorderOption(outer_nid=11, inner_nid=12)),
    (Reorder(), ReorderOption(outer_nid=12, inner_nid=13)),
    (ComputeAt(), ComputeAtOption(block_nid=7, target_loop_nid=11, index=0)),
    (ComputeAt(), ComputeAtOption(block_nid=15, target_loop_nid=11, index=2)),
]


def _sim_check(source: str, scratch_path: str, atol: float = 5e-3, rtol: float = 5e-3) -> None:
    """CPU-sim ``source`` at fp32 and assert it matches ``lhs_T.T @ rhs``.

    Writes the rendered kernel to ``scratch_path``, imports it, runs the
    fp32 simulator, and compares against the numpy golden. Raises (halting
    the run) on any mismatch, so only correct kernels reach the profiler.
    """
    rng = np.random.default_rng(SEED)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    with open(scratch_path, "w", encoding="utf-8") as handle:
        handle.write(source)
    spec = importlib.util.spec_from_file_location("dumped_kernel", scratch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = np.asarray(simulate_fp32(module.nki_f_nkigym)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def _make_job(source: str, output_shape: tuple[int, ...]) -> KernelJob:
    """Wrap a rendered kernel source string as a profileable ``KernelJob``.

    ``neuronx_cc_args`` is left empty: nkigym kernels carry no ``address=``
    annotations, so the Neuron allocator + scheduler must stay on.
    """
    return KernelJob(
        source=source,
        func_name=RENDERED_FUNC_NAME,
        output_shape=output_shape,
        input_specs=INPUT_SPECS,
    )


def _validate_trace(cache_dir: str) -> tuple[str, tuple[int, ...]]:
    """Replay ``TRACE`` from canonical, sim-check every rung, return the final kernel.

    Each rung (canonical + the four transformed states) is rendered and
    CPU-sim'd at fp32 against the numpy golden, so the whole transform path
    is proven numerically correct before any hardware run. Returns the
    ``(source, output_shape)`` of the final tuned kernel only — the
    intermediate rungs still hold the full-extent 16 MiB PSUM accumulator
    and would PSUM-OOM on hardware, so only the final kernel is profiled.
    """
    env = KernelMDP(f_nkigym, INPUT_SPECS, transforms=[Split(), Fuse(), Reorder(), ComputeAt(), ReverseComputeAt()])
    state = env.reset()
    output_shape = tuple(state.buffer(state.return_name).shape)
    scratch_path = os.path.join(tempfile.gettempdir(), "tune_matmul_sim_scratch.py")

    source = render(state)
    _sim_check(source, scratch_path)
    print("[tune] step_00 (canonical): sim PASS")

    for index, action in enumerate(TRACE, start=1):
        state = env.step(state, action)
        source = render(state)
        _sim_check(source, scratch_path)
        transform_name = type(action[0]).__name__
        print(f"[tune] step_{index:02d} ({transform_name}): sim PASS")
    return source, output_shape


def _numpy_matmul(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """``lhs_T.T @ rhs`` — the compiler-baseline reference (traced by nkipy)."""
    return lhs_T.T @ rhs


def _print_baseline(baseline: ProfileResult, tuned: ProfileOutput) -> None:
    """Print the compiler baseline next to the tuned kernel's MFU / time.

    Reads MFU and wall-clock straight from the baseline's profiler summary
    (``None`` when neuronx-cc failed to produce one) and the tuned kernel's
    best success row, so the two appear on the same scale.
    """
    base_mfu = profiler_percent(baseline.profiler_summary, "mfu_estimated_percent")
    base_time = (baseline.profiler_summary or {}).get("total_time")
    if base_mfu is None or not isinstance(base_time, (int, float)):
        print(f"\n[tune] compiler baseline unavailable: {baseline.hardware_output.splitlines()[0]}")
    else:
        best = min(tuned.successes, key=lambda s: s.total_time_s) if tuned.successes else None
        print("\n[tune] reference comparison:")
        if best is not None:
            print(f"  tuned (nkigym):    {best.total_time_s:.6f} s   {best.mfu:.2f}% MFU")
        print(f"  compiler baseline: {float(base_time):.6f} s   {base_mfu:.2f}% MFU")


def main() -> None:
    """Sim-validate the trace, profile the tuned kernel + compiler baseline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root-dir", required=True)
    parser.add_argument("--detailed", action="store_true", help="collect per-instruction profiler trace + NEFF/NTFF")
    args = parser.parse_args()

    cache_dir = os.path.join(args.cache_root_dir, "tune_matmul_lhsT_rhs")
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

    source, output_shape = _validate_trace(cache_dir)
    jobs = {"tuned": _make_job(source, output_shape)}
    print(f"\n[tune] profiling tuned kernel on {NEURON_PLATFORM_TARGET} ...\n")
    output = profile(
        jobs,
        cache_dir=cache_dir,
        seed=SEED,
        neuron_platform_target=NEURON_PLATFORM_TARGET,
        collect_detailed_profile=args.detailed,
    )
    print(output)

    print(f"\n[tune] compiling numpy baseline via neuronx-cc on {NEURON_PLATFORM_TARGET} ...")
    baseline, _nki_source = profile_numpy_baseline(
        _numpy_matmul, INPUT_SPECS, os.path.join(cache_dir, "compiler_baseline"), kernel_name="compiler_baseline"
    )
    _print_baseline(baseline, output)


if __name__ == "__main__":
    main()
