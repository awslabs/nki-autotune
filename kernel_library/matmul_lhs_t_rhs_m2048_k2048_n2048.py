"""Apply, verify, dump, and profile the M2048/K2048/N2048 ``lhs_T.T @ rhs`` ladder."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType

import numpy as np

from kernel_library import InputSpecs, Workload
from nkigym.codegen import render
from nkigym.environment import Action, KernelMDP
from nkigym.ir import KernelIR
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.profile import profile, simulate_fp32
from nkigym.transforms import (
    BufferCompaction,
    BufferCompactionOption,
    BufferLayout,
    BufferLayoutOption,
    BufferPlacement,
    BufferPlacementOption,
    CodeMotion,
    CodeMotionOption,
    Reorder,
    ReorderOption,
    RFactor,
    RFactorOption,
    Split,
    SplitOption,
)

M = 2048
K = 2048
N = 2048
SEED = 0
WORKLOAD_NAME = "matmul-lhs-t"
SHAPE = "m2048_k2048_n2048"
INPUT_SPECS = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")


def input_generator(input_specs: InputSpecs, seed: int) -> dict[str, np.ndarray]:
    """Generate Kaena-style uniform FP32 matmul inputs."""
    rng = np.random.default_rng(seed)
    return {name: rng.random(shape).astype(np.float32) for name, (shape, _dtype) in input_specs.items()}


def f_numpy(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Compute ``lhs_T.T @ rhs`` in FP32."""
    return lhs_T.astype(np.float32).T @ rhs.astype(np.float32)


@nkigym_kernel
def f_nkigym(lhs_T, rhs):
    """Define ``lhs_T.T @ rhs`` as an SSA operator graph."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


ACTIONS: tuple[Action, ...] = (
    (Reorder(), ReorderOption(outer_nid=12, inner_nid=13)),
    (Reorder(), ReorderOption(outer_nid=11, inner_nid=12)),
    (Split(), SplitOption(target_nid=12, factors=(2, 8), target_axis=None)),
    (Split(), SplitOption(target_nid=13, factors=(4, 4), target_axis=None)),
    (Reorder(), ReorderOption(outer_nid=22, inner_nid=23)),
    (Reorder(), ReorderOption(outer_nid=23, inner_nid=24)),
    (BufferLayout(), BufferLayoutOption(tensor="psum_prod", list_len=16)),
    (Split(), SplitOption(target_nid=17, factors=(4, 512), target_axis="d2")),
    (Reorder(), ReorderOption(outer_nid=16, inner_nid=25)),
    (CodeMotion(), CodeMotionOption(block_nid=15, target_loop_nid=11, index=1)),
    (Split(), SplitOption(target_nid=20, factors=(4, 512), target_axis="d2")),
    (Reorder(), ReorderOption(outer_nid=19, inner_nid=26)),
    (CodeMotion(), CodeMotionOption(block_nid=18, target_loop_nid=11, index=2)),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_prod")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_prod")),
    (BufferLayout(), BufferLayoutOption(tensor="sbuf_prod", list_len=16)),
    (Split(), SplitOption(target_nid=9, factors=(4, 512), target_axis="d2")),
    (Reorder(), ReorderOption(outer_nid=8, inner_nid=27)),
    (CodeMotion(), CodeMotionOption(block_nid=7, target_loop_nid=11, index=0)),
    (BufferPlacement(), BufferPlacementOption(tensor="psum_prod")),
    (BufferCompaction(), BufferCompactionOption(tensor="psum_prod")),
    (Split(), SplitOption(target_nid=5, factors=(2, 8), target_axis=None)),
    (Split(), SplitOption(target_nid=6, factors=(4, 512), target_axis="d2")),
    (Reorder(), ReorderOption(outer_nid=29, inner_nid=30)),
    (Reorder(), ReorderOption(outer_nid=28, inner_nid=29)),
    (CodeMotion(), CodeMotionOption(block_nid=4, target_loop_nid=21, index=0)),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_rhs")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_rhs")),
    (BufferLayout(), BufferLayoutOption(tensor="sbuf_rhs", list_len=8)),
    (Split(), SplitOption(target_nid=3, factors=(4, 512), target_axis="d1")),
    (Split(), SplitOption(target_nid=2, factors=(2, 8), target_axis=None)),
    (Reorder(), ReorderOption(outer_nid=33, inner_nid=31)),
    (CodeMotion(), CodeMotionOption(block_nid=1, target_loop_nid=22, index=0)),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_lhs_T")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_lhs_T")),
    (BufferLayout(), BufferLayoutOption(tensor="sbuf_lhs_T", list_len=8)),
    (RFactor(), RFactorOption(target_loop_nid=21, factor_axis=0)),
    (BufferLayout(), BufferLayoutOption(tensor="psum_prod", list_len=1)),
    (BufferCompaction(), BufferCompactionOption(tensor="psum_prod")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_rfactor")),
)

WORKLOAD = Workload(
    input_specs=INPUT_SPECS,
    f_numpy=f_numpy,
    f_nkigym=f_nkigym,
    input_generator=input_generator,
    atol=1e-3,
    rtol=1e-3,
    best_action_ladder=ACTIONS,
    historical_best_mfu=90.92,
)


def _parse_args() -> argparse.Namespace:
    """Parse artifact and SSH profile controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--host", required=True)
    return parser.parse_args()


def _labels(workload: Workload) -> list[str]:
    """Return one stable cache label per ladder state."""
    return [
        "00_canonical",
        *[f"{index:02d}_{type(action[0]).__name__}" for index, action in enumerate(workload.best_action_ladder, 1)],
    ]


def _build_ladder(workload: Workload) -> list[KernelIR]:
    """Apply every fixed action through one kernel environment."""
    environment = KernelMDP(
        workload.f_nkigym,
        workload.input_specs,
        transforms=[transform for transform, _option in workload.best_action_ladder],
    )
    states = [environment.reset()]
    for action in workload.best_action_ladder:
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


def _verify_and_dump(workload: Workload, states: list[KernelIR], cache: Path) -> dict[str, float]:
    """Dump and CPU-verify every ladder state."""
    inputs = workload.generate_inputs(SEED)
    expected = workload.f_numpy(**inputs)
    errors: dict[str, float] = {}
    for index, (label, state) in enumerate(zip(_labels(workload), states, strict=True)):
        state_dir = cache / "kernels" / label
        state.dump(state_dir)
        module = _load_kernel(state_dir / "kernel.py", f"matmul_lhsT_rhs_{index}")
        actual = np.asarray(simulate_fp32(module.nki_f_nkigym)(**inputs))
        np.testing.assert_allclose(actual, expected, atol=workload.atol, rtol=workload.rtol)
        errors[label] = float(np.max(np.abs(actual - expected)))
        (state_dir / "accuracy.json").write_text(
            json.dumps({"max_abs_error": errors[label], "atol": workload.atol, "rtol": workload.rtol}, indent=2) + "\n",
            encoding="utf-8",
        )
    return errors


def _profile(workload: Workload, states: list[KernelIR], cache: Path, host: str) -> dict[str, object]:
    """Profile every state on the SSH Trn2 host."""
    profile_dir = cache / "mfu"
    measurements: dict[str, dict[str, float]] = {}
    failures: dict[str, str] = {}
    for label, state in zip(_labels(workload), states, strict=True):
        try:
            mfu_percent, latency_ms = profile(
                host=host,
                kernel=render(state),
                func_name="nki_f_nkigym",
                input_specs=workload.input_specs,
                cache_dir=profile_dir / label,
                neuronx_cc_args=SCHEDULER_OFF_ARGS,
            )
            measurements[label] = {"mfu_percent": mfu_percent, "latency_ms": latency_ms}
        except RuntimeError as error:
            if not (profile_dir / label / "result.json").is_file():
                raise
            failures[label] = str(error)
    (profile_dir / "results.json").write_text(
        json.dumps({"successes": measurements, "failures": failures}, indent=2) + "\n", encoding="utf-8"
    )
    final_label = _labels(workload)[-1]
    if final_label not in measurements:
        raise RuntimeError(f"final state did not profile successfully: {failures[final_label]}")
    return {"successes": measurements, "failure_count": len(failures), "results": "mfu/results.json"}


def _main() -> None:
    """Build, dump, verify, and profile the fixed ladder."""
    args = _parse_args()
    cache = Path(args.cache).expanduser().resolve()
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True)

    states = _build_ladder(WORKLOAD)
    errors = _verify_and_dump(WORKLOAD, states, cache)
    summary: dict[str, object] = {
        "workload": WORKLOAD_NAME,
        "shape": SHAPE,
        "states": len(states),
        "accuracy": {"passed": len(errors), "max_abs_error": max(errors.values()), "kernels": "kernels/"},
    }
    summary["profile"] = _profile(WORKLOAD, states, cache, args.host)
    (cache / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
