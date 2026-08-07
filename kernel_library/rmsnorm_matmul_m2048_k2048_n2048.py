"""Apply, verify, dump, and profile the M2048/K2048/N2048 RMSNorm+matmul ladder."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np

from kernel_library import InputSpecs, Workload
from nkigym.codegen import render
from nkigym.environment import Action, KernelMDP
from nkigym.ir import KernelIR
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.profile import profile_many, simulate_fp32
from nkigym.transforms import (
    BatchPermutation,
    BatchPermutationOption,
    BufferCompaction,
    BufferCompactionOption,
    BufferLayout,
    BufferLayoutOption,
    BufferPlacement,
    BufferPlacementOption,
    CodeMotion,
    CodeMotionOption,
    CopyPropagation,
    CopyPropagationOption,
    Fuse,
    FuseOption,
    FusePointwise,
    FusePointwiseOption,
    OnlineFusion,
    OnlineFusionOption,
    Reorder,
    ReorderOption,
    SoftwarePipeline,
    SoftwarePipelineOption,
    Split,
    SplitOption,
)

M = 2048
K = 2048
N = 2048
EPSILON = 1e-6
SEED = 0
WORKLOAD_NAME = "rmsnorm-matmul"
SHAPE = "m2048_k2048_n2048"
INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}
SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")


def input_generator(input_specs: InputSpecs, seed: int) -> dict[str, np.ndarray]:
    """Generate Kaena-style small uniform FP32 RMSNorm inputs."""
    rng = np.random.default_rng(seed)
    return {
        name: rng.uniform(-0.1, 0.1, size=shape).astype(np.float32) for name, (shape, _dtype) in input_specs.items()
    }


def f_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Compute row-wise RMSNorm followed by matmul in FP32."""
    lhs_fp32 = lhs.astype(np.float32)
    normalized = lhs_fp32 / np.sqrt(np.mean(np.square(lhs_fp32), axis=1, keepdims=True) + EPSILON)
    return normalized @ rhs.astype(np.float32)


def make_f_nkigym(k: int) -> Callable[..., np.ndarray]:
    """Build an RMSNorm+matmul graph for one reduction extent."""
    scale = 1.0 / k

    @nkigym_kernel
    def f_nkigym(lhs, rhs):
        """Define ``rmsnorm(lhs) @ rhs`` as an SSA operator graph."""
        sbuf_rhs = NKILoad()(src=rhs)
        sbuf_lhs = NKILoad()(src=lhs)
        sbuf_square_sum = NKIActivationReduce(op="square", reduce_op="add")(data=sbuf_lhs)
        sbuf_rms_inverse = NKIActivation(op="rsqrt", scale=scale, bias=EPSILON)(data=sbuf_square_sum)
        sbuf_normalized = NKITensorScalar(op0="multiply")(data=sbuf_lhs, operand0=sbuf_rms_inverse)
        sbuf_normalized_T = NKIDMATranspose()(src=sbuf_normalized)
        psum_output = NKIMatmul()(stationary=sbuf_normalized_T, moving=sbuf_rhs)
        sbuf_output = NKITensorCopy()(src=psum_output)
        hbm_output = NKIStore()(src=sbuf_output)
        return hbm_output

    return f_nkigym


f_nkigym = make_f_nkigym(K)


ACTIONS: tuple[Action, ...] = (
    (Split(), SplitOption(target_nid=5, factors=(2, 8), target_axis=None)),
    (Split(), SplitOption(target_nid=8, factors=(2, 8), target_axis=None)),
    (Split(), SplitOption(target_nid=11, factors=(2, 8), target_axis=None)),
    (Split(), SplitOption(target_nid=14, factors=(2, 8), target_axis=None)),
    (Split(), SplitOption(target_nid=17, factors=(2, 8), target_axis=None)),
    (Split(), SplitOption(target_nid=21, factors=(2, 8), target_axis=None)),
    (Split(), SplitOption(target_nid=25, factors=(2, 8), target_axis=None)),
    (Split(), SplitOption(target_nid=29, factors=(2, 8), target_axis=None)),
    (OnlineFusion(), OnlineFusionOption(match_id=("d1", (9, 27)), chunk_size=2048)),
    (Split(), SplitOption(target_nid=54, factors=(4, 512), target_axis="d2")),
    (Split(), SplitOption(target_nid=75, factors=(4, 512), target_axis="d2")),
    (Split(), SplitOption(target_nid=30, factors=(4, 512), target_axis="d2")),
    (Split(), SplitOption(target_nid=32, factors=(2, 8), target_axis=None)),
    (Split(), SplitOption(target_nid=33, factors=(4, 512), target_axis="d2")),
    (Reorder(), ReorderOption(outer_nid=53, inner_nid=105)),
    (Reorder(), ReorderOption(outer_nid=77, inner_nid=78)),
    (Reorder(), ReorderOption(outer_nid=78, inner_nid=79)),
    (Reorder(), ReorderOption(outer_nid=79, inner_nid=80)),
    (CopyPropagation(), CopyPropagationOption(copy_block_nid=92, consumer_block_nid=96, consumer_operand="data")),
    (CodeMotion(), CodeMotionOption(block_nid=72, target_loop_nid=57, index=1)),
    (CodeMotion(), CodeMotionOption(block_nid=67, target_loop_nid=57, index=2)),
    (CodeMotion(), CodeMotionOption(block_nid=59, target_loop_nid=57, index=3)),
    (CodeMotion(), CodeMotionOption(block_nid=63, target_loop_nid=57, index=4)),
    (CodeMotion(), CodeMotionOption(block_nid=96, target_loop_nid=57, index=5)),
    (CodeMotion(), CodeMotionOption(block_nid=76, target_loop_nid=57, index=6)),
    (CodeMotion(), CodeMotionOption(block_nid=82, target_loop_nid=57, index=7)),
    (CodeMotion(), CodeMotionOption(block_nid=87, target_loop_nid=57, index=8)),
    (CodeMotion(), CodeMotionOption(block_nid=100, target_loop_nid=57, index=9)),
    (CodeMotion(), CodeMotionOption(block_nid=28, target_loop_nid=57, index=10)),
    (CodeMotion(), CodeMotionOption(block_nid=31, target_loop_nid=57, index=11)),
    (CopyPropagation(), CopyPropagationOption(copy_block_nid=63, consumer_block_nid=96, consumer_operand="data")),
    (CopyPropagation(), CopyPropagationOption(copy_block_nid=82, consumer_block_nid=87, consumer_operand="src")),
    (CopyPropagation(), CopyPropagationOption(copy_block_nid=87, consumer_block_nid=100, consumer_operand="data")),
    (FusePointwise(), FusePointwiseOption(pointwise_block_nid=100, consumer_block_nid=28)),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_rhs")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_square_sum_scratch")),
    (BufferPlacement(), BufferPlacementOption(tensor="online_deferred_factor_rsqrt")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_square_sum_online_chunk")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_output")),
    (BufferPlacement(), BufferPlacementOption(tensor="psum_output_online_partial")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_normalized_T")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_lhs")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_lhs")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_normalized_T")),
    (BufferCompaction(), BufferCompactionOption(tensor="psum_output_online_partial")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_output")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_square_sum_online_chunk")),
    (BufferCompaction(), BufferCompactionOption(tensor="online_deferred_factor_rsqrt")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_square_sum_scratch")),
    (Fuse(), FuseOption(target_nids=(56, 57), target_axis=None)),
    (SoftwarePipeline(), SoftwarePipelineOption(loop_nid=111, stages=(0, 1, 1, 1, 1, 1, 2, 3))),
    (BatchPermutation(), BatchPermutationOption(loop_nid=70)),
    (BufferLayout(), BufferLayoutOption(tensor="sbuf_rhs", list_len=4)),
)

WORKLOAD = Workload(
    input_specs=INPUT_SPECS,
    f_numpy=f_numpy,
    f_nkigym=f_nkigym,
    input_generator=input_generator,
    atol=1e-3,
    rtol=2e-2,
    best_action_ladder=ACTIONS,
    historical_best_mfu=86.99,
    reference_mfu=79.09,
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
        module = _load_kernel(state_dir / "kernel.py", f"rmsnorm_matmul_{index}")
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
    labels = _labels(workload)
    return profile_many(
        host=host,
        kernels={label: render(state) for label, state in zip(labels, states, strict=True)},
        func_name="nki_f_nkigym",
        input_specs=workload.input_specs,
        cache_dir=cache / "mfu",
        neuronx_cc_args=SCHEDULER_OFF_ARGS,
        required_successes=(labels[-1],),
    )


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
