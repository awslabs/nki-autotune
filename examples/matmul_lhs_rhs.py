"""Apply, verify, dump, and optionally profile a fixed ``lhs @ rhs`` ladder."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType

import numpy as np

from autotune.runner.types import KernelJob
from nkigym.codegen import render
from nkigym.environment import Action, KernelMDP
from nkigym.ir import KernelIR
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.transpose import NKITranspose
from nkigym.synthesis import simulate_fp32
from nkigym.transforms import (
    BufferCompaction,
    BufferCompactionOption,
    BufferLayout,
    BufferLayoutOption,
    CodeMotion,
    CodeMotionOption,
    Reorder,
    ReorderOption,
    RFactor,
    RFactorOption,
    Split,
    SplitOption,
    TransposeThroughLoad,
    TransposeThroughLoadOption,
)

SIZE = 2048
SEED = 0
ATOL = 5e-3
RTOL = 5e-3
INPUT_SPECS = {"lhs": ((SIZE, SIZE), "bfloat16"), "rhs": ((SIZE, SIZE), "bfloat16")}
SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")


def f_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Compute ``lhs @ rhs`` in FP32."""
    return lhs.astype(np.float32) @ rhs.astype(np.float32)


@nkigym_kernel
def f_nkigym(lhs, rhs):
    """Define ``lhs @ rhs`` as an SSA operator graph."""
    sbuf_lhs = NKILoad()(src=lhs)
    psum_lhs_T = NKITranspose()(data=sbuf_lhs)
    sbuf_lhs_T = NKITensorCopy()(src=psum_lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


ACTIONS: tuple[Action, ...] = (
    (TransposeThroughLoad(), TransposeThroughLoadOption(target_nid=1)),
    (Reorder(), ReorderOption(outer_nid=19, inner_nid=20)),
    (Reorder(), ReorderOption(outer_nid=18, inner_nid=19)),
    (Split(), SplitOption(target_nid=19, factors=(2, 8), target_axis=None)),
    (Split(), SplitOption(target_nid=20, factors=(4, 4), target_axis=None)),
    (Reorder(), ReorderOption(outer_nid=32, inner_nid=33)),
    (Reorder(), ReorderOption(outer_nid=33, inner_nid=34)),
    (BufferLayout(), BufferLayoutOption(tensor="psum_prod", list_len=16)),
    (Split(), SplitOption(target_nid=24, factors=(4, 512), target_axis="d2")),
    (Reorder(), ReorderOption(outer_nid=23, inner_nid=35)),
    (CodeMotion(), CodeMotionOption(block_nid=22, target_loop_nid=18, index=-1)),
    (Split(), SplitOption(target_nid=27, factors=(4, 512), target_axis="d2")),
    (Reorder(), ReorderOption(outer_nid=26, inner_nid=36)),
    (CodeMotion(), CodeMotionOption(block_nid=25, target_loop_nid=18, index=-1)),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_prod")),
    (BufferLayout(), BufferLayoutOption(tensor="sbuf_prod", list_len=16)),
    (Split(), SplitOption(target_nid=16, factors=(4, 512), target_axis="d2")),
    (Reorder(), ReorderOption(outer_nid=15, inner_nid=37)),
    (CodeMotion(), CodeMotionOption(block_nid=14, target_loop_nid=18, index=0)),
    (BufferCompaction(), BufferCompactionOption(tensor="psum_prod")),
    (Split(), SplitOption(target_nid=12, factors=(2, 8), target_axis=None)),
    (Split(), SplitOption(target_nid=13, factors=(4, 512), target_axis="d2")),
    (Reorder(), ReorderOption(outer_nid=39, inner_nid=40)),
    (Reorder(), ReorderOption(outer_nid=38, inner_nid=39)),
    (CodeMotion(), CodeMotionOption(block_nid=11, target_loop_nid=31, index=0)),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_rhs")),
    (BufferLayout(), BufferLayoutOption(tensor="sbuf_rhs", list_len=8)),
    (RFactor(), RFactorOption(target_loop_nid=31, factor_axis=0)),
    (BufferCompaction(), BufferCompactionOption(tensor="psum_prod")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_rfactor")),
    (BufferLayout(), BufferLayoutOption(tensor="sbuf_lhs_T", list_len=16)),
)


def _parse_args() -> argparse.Namespace:
    """Parse the cache path and optional Trn2 profiling flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def _labels() -> list[str]:
    """Return one stable cache label per ladder state."""
    return ["00_canonical", *[f"{index:02d}_{type(action[0]).__name__}" for index, action in enumerate(ACTIONS, 1)]]


def _build_ladder() -> list[KernelIR]:
    """Apply every hardcoded action through one kernel environment."""
    environment = KernelMDP(f_nkigym, INPUT_SPECS, transforms=[transform for transform, _option in ACTIONS])
    states = [environment.reset()]
    for action in ACTIONS:
        states.append(environment.step(states[-1], action))
    return states


def _load_kernel(path: Path, module_name: str) -> ModuleType:
    """Import one dumped kernel."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_and_dump(states: list[KernelIR], cache: Path) -> dict[str, float]:
    """Dump and CPU-verify every ladder state."""
    rng = np.random.default_rng(SEED)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in INPUT_SPECS.items()}
    expected = f_numpy(**inputs)
    errors: dict[str, float] = {}
    for index, (label, state) in enumerate(zip(_labels(), states, strict=True)):
        state_dir = cache / "kernels" / label
        state.dump(state_dir)
        module = _load_kernel(state_dir / "kernel.py", f"matmul_lhs_rhs_{index}")
        actual = np.asarray(simulate_fp32(module.nki_f_nkigym)(**inputs))
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
        errors[label] = float(np.max(np.abs(actual - expected)))
        (state_dir / "accuracy.json").write_text(
            json.dumps({"max_abs_error": errors[label], "atol": ATOL, "rtol": RTOL}, indent=2) + "\n", encoding="utf-8"
        )
    return errors


def _profile(states: list[KernelIR], cache: Path) -> dict[str, object]:
    """Profile every state on the local Trn2 device."""
    from autotune.runner.api import profile

    jobs = {
        f"{label}.py": KernelJob(
            source=render(state),
            func_name="nki_f_nkigym",
            output_shape=(SIZE, SIZE),
            input_specs=INPUT_SPECS,
            neuronx_cc_args=SCHEDULER_OFF_ARGS,
            lnc=1,
        )
        for label, state in zip(_labels(), states, strict=True)
    }
    output = profile(
        jobs, cache_dir=str(cache / "mfu"), seed=SEED, neuron_platform_target="trn2", collect_detailed_profile=False
    )
    measurements = {
        Path(row.kernel_name).stem: {"mfu_percent": row.mfu, "latency_ms": row.total_time_s * 1000.0}
        for row in output.successes
    }
    final_label = _labels()[-1]
    if "mfu_percent" not in measurements.get(final_label, {}):
        failure = next(result for result in output.failures if Path(result.kernel_name).stem == final_label)
        raise RuntimeError(f"final state did not profile successfully: {failure.hardware_output[-1000:]}")
    return {"successes": measurements, "failure_count": len(output.failures), "results": "mfu/results.json"}


def _main() -> None:
    """Build, dump, verify, and optionally profile the fixed ladder."""
    args = _parse_args()
    cache = Path(args.cache).expanduser().resolve()
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True)

    states = _build_ladder()
    errors = _verify_and_dump(states, cache)
    summary: dict[str, object] = {
        "states": len(states),
        "accuracy": {"passed": len(errors), "max_abs_error": max(errors.values()), "kernels": "kernels/"},
    }
    if args.profile:
        summary["profile"] = _profile(states, cache)
    (cache / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
