"""Apply, verify, dump, and profile a fixed online-fusion ladder."""

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
from nkigym.ops.activation import NKIActivation
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.synthesis import simulate_fp32
from nkigym.transforms import (
    BatchPermutation,
    BatchPermutationOption,
    BufferCompaction,
    BufferCompactionOption,
    CodeMotion,
    CodeMotionOption,
    CommonSubexpressionElimination,
    CommonSubexpressionEliminationOption,
    CopyPropagation,
    CopyPropagationOption,
    EliminateIdentityInitializer,
    EliminateIdentityInitializerOption,
    FuseBroadcastActivation,
    FuseBroadcastActivationOption,
    FusePointwiseActivation,
    FusePointwiseActivationOption,
    FusePointwiseReduction,
    FusePointwiseReductionOption,
    OnlineFusion,
    OnlineFusionOption,
    Reorder,
    ReorderOption,
    RFactor,
    RFactorOption,
    SoftwarePipeline,
    SoftwarePipelineOption,
    Split,
    SplitOption,
)

InputSpecs = dict[str, tuple[tuple[int, ...], str]]

HEAD_DIM = 128
SEQUENCE_LENGTH = 16384
VALIDATION_QUERY_LENGTH = 512
SEED = 0
ATOL = 5e-3
RTOL = 5e-3
SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")


def _input_specs(query_length: int) -> InputSpecs:
    """Return pretransposed BF16 attention inputs."""
    return {
        "query": ((HEAD_DIM, query_length), "bfloat16"),
        "key": ((HEAD_DIM, SEQUENCE_LENGTH), "bfloat16"),
        "value": ((SEQUENCE_LENGTH, HEAD_DIM), "bfloat16"),
    }


def f_numpy(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Compute scaled dot-product attention in FP32."""
    scores = query.astype(np.float32).T @ key.astype(np.float32) / np.sqrt(HEAD_DIM)
    scores -= np.max(scores, axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities @ value.astype(np.float32)


@nkigym_kernel
def f_nkigym(query, key, value):
    """Define materialized attention as an SSA operator graph."""
    sbuf_query = NKILoad()(src=query)
    sbuf_key = NKILoad()(src=key)
    psum_scores = NKIMatmul()(stationary=sbuf_query, moving=sbuf_key)
    sbuf_scores = NKITensorCopy()(src=psum_scores)
    sbuf_scaled_scores = NKITensorScalar(op0="multiply")(data=sbuf_scores, operand0=HEAD_DIM**-0.5)
    sbuf_row_max = NKITensorReduce(op="maximum", axis=1)(data=sbuf_scaled_scores)
    sbuf_centered = NKITensorScalar(op0="subtract")(data=sbuf_scaled_scores, operand0=sbuf_row_max)
    sbuf_exp = NKIActivation(op="exp")(data=sbuf_centered)
    sbuf_row_sum = NKITensorReduce(op="add", axis=1)(data=sbuf_exp)
    sbuf_inv_sum = NKIActivation(op="reciprocal")(data=sbuf_row_sum)
    sbuf_probability = NKITensorScalar(op0="multiply")(data=sbuf_exp, operand0=sbuf_inv_sum)
    sbuf_probability_t = NKIDMATranspose()(src=sbuf_probability)
    sbuf_value = NKILoad()(src=value)
    psum_output = NKIMatmul()(stationary=sbuf_probability_t, moving=sbuf_value)
    sbuf_output = NKITensorCopy()(src=psum_output)
    hbm_output = NKIStore()(src=sbuf_output)
    return hbm_output


ACTIONS: tuple[Action, ...] = (
    (OnlineFusion(), OnlineFusionOption(match_id=("d2", (20, 29)), chunk_size=8192)),
    (OnlineFusion(), OnlineFusionOption(match_id=("d2", (20, 29, 49)), chunk_size=8192)),
    (FusePointwiseReduction(), FusePointwiseReductionOption(pointwise_block_nid=82, reduction_block_nid=85)),
    (FusePointwiseReduction(), FusePointwiseReductionOption(pointwise_block_nid=94, reduction_block_nid=97)),
    (FuseBroadcastActivation(), FuseBroadcastActivationOption(pointwise_block_nid=91, activation_block_nid=94)),
    (FusePointwiseActivation(), FusePointwiseActivationOption(pointwise_block_nid=100, activation_block_nid=103)),
    (FusePointwiseActivation(), FusePointwiseActivationOption(pointwise_block_nid=129, activation_block_nid=132)),
    (
        CommonSubexpressionElimination(),
        CommonSubexpressionEliminationOption(canonical_block_nid=103, redundant_block_nid=132),
    ),
    (Split(), SplitOption(target_nid=84, factors=(16, 512), target_axis="d2")),
    (Split(), SplitOption(target_nid=96, factors=(16, 512), target_axis="d2")),
    (Reorder(), ReorderOption(outer_nid=120, inner_nid=121)),
    (CodeMotion(), CodeMotionOption(block_nid=109, target_loop_nid=117, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=106, target_loop_nid=117, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=103, target_loop_nid=117, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=162, target_loop_nid=117, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=94, target_loop_nid=117, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=91, target_loop_nid=117, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=88, target_loop_nid=117, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=158, target_loop_nid=117, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=82, target_loop_nid=117, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=79, target_loop_nid=117, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=75, target_loop_nid=117, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=72, target_loop_nid=117, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=119, target_loop_nid=117, index=-1)),
    (CodeMotion(), CodeMotionOption(block_nid=123, target_loop_nid=117, index=-1)),
    (CodeMotion(), CodeMotionOption(block_nid=126, target_loop_nid=117, index=-1)),
    (CodeMotion(), CodeMotionOption(block_nid=135, target_loop_nid=117, index=-1)),
    (CodeMotion(), CodeMotionOption(block_nid=138, target_loop_nid=117, index=-1)),
    (CodeMotion(), CodeMotionOption(block_nid=141, target_loop_nid=117, index=-1)),
    (CodeMotion(), CodeMotionOption(block_nid=144, target_loop_nid=117, index=-1)),
    (Split(), SplitOption(target_nid=74, factors=(16, 512), target_axis="d2")),
    (Split(), SplitOption(target_nid=81, factors=(16, 512), target_axis="d2")),
    (CodeMotion(), CodeMotionOption(block_nid=72, target_loop_nid=77, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=79, target_loop_nid=77, index=-1)),
    (Split(), SplitOption(target_nid=121, factors=(4, 16), target_axis=None)),
    (RFactor(), RFactorOption(target_loop_nid=167, factor_axis=0)),
    (CodeMotion(), CodeMotionOption(block_nid=82, target_loop_nid=77, index=3)),
    (CopyPropagation(), CopyPropagationOption(copy_block_nid=79, consumer_block_nid=82, consumer_operand="data")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_row_sum_online_chunk_rfactor")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_row_max_online_chunk_rfactor")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_row_max_online_current_negative")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_scaled_scores")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_exp")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_probability_t")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_row_max_online_chunk")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_row_max_online_current")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_row_sum_online_chunk")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_row_sum_online_current")),
    (BufferCompaction(), BufferCompactionOption(tensor="psum_output_online_state")),
    (BufferCompaction(), BufferCompactionOption(tensor="psum_output_online_chunk")),
    (BufferCompaction(), BufferCompactionOption(tensor="online_stage1_correction")),
    (BufferCompaction(), BufferCompactionOption(tensor="psum_scores")),
    (BufferCompaction(), BufferCompactionOption(tensor="psum_output")),
    (BufferCompaction(), BufferCompactionOption(tensor="sbuf_rfactor")),
    (
        EliminateIdentityInitializer(),
        EliminateIdentityInitializerOption(initializer_block_nid=72, reduction_block_nid=75, tensor="psum_scores"),
    ),
    (
        SoftwarePipeline(),
        SoftwarePipelineOption(
            loop_nid=117,
            stages=(0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2),
            order=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        ),
    ),
    (Split(), SplitOption(target_nid=111, factors=(16, 4), target_axis=None)),
    (BatchPermutation(), BatchPermutationOption(loop_nid=176)),
)

STEPS: tuple[tuple[Action, ...], ...] = tuple((action,) for action in ACTIONS)


def _parse_args() -> argparse.Namespace:
    """Parse the cache path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    return parser.parse_args()


def _labels() -> list[str]:
    """Return one stable cache label per ladder state."""
    return [
        "00_canonical",
        *[
            f"{index:02d}_{'_'.join(type(action[0]).__name__ for action in step)}"
            for index, step in enumerate(STEPS, 1)
        ],
    ]


def _build_ladder(input_specs: InputSpecs) -> list[KernelIR]:
    """Apply every hardcoded action through one kernel environment."""
    environment = KernelMDP(f_nkigym, input_specs, transforms=[transform for transform, _option in ACTIONS])
    states = [environment.reset()]
    state = states[0]
    for step in STEPS:
        for action in step:
            state = environment.step(state, action)
        states.append(state)
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
    """Dump and CPU-verify every validation-shaped state."""
    input_specs = _input_specs(VALIDATION_QUERY_LENGTH)
    rng = np.random.default_rng(SEED)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in input_specs.items()}
    expected = f_numpy(**inputs)
    errors: dict[str, float] = {}
    for index, (label, state) in enumerate(zip(_labels(), states, strict=True)):
        state_dir = cache / "kernels" / label
        state.dump(state_dir)
        module = _load_kernel(state_dir / "kernel.py", f"online_fusion_attention_{index}")
        kernel = getattr(module, "nki_f_nkigym")
        with np.errstate(divide="ignore", invalid="ignore"):
            actual = np.asarray(simulate_fp32(kernel)(**inputs))
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
        errors[label] = float(np.max(np.abs(actual - expected)))
        (state_dir / "accuracy.json").write_text(
            json.dumps({"max_abs_error": errors[label], "atol": ATOL, "rtol": RTOL}, indent=2) + "\n", encoding="utf-8"
        )
    return errors


def _profile(states: list[KernelIR], cache: Path) -> dict[str, object]:
    """Profile every production-shaped state on the local Trn2 device."""
    from autotune.runner.api import profile

    input_specs = _input_specs(SEQUENCE_LENGTH)
    jobs = {
        f"{label}.py": KernelJob(
            source=render(state),
            func_name="nki_f_nkigym",
            output_shape=(SEQUENCE_LENGTH, HEAD_DIM),
            input_specs=input_specs,
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
    """Build the ladder, persist every state, verify it, and profile it."""
    args = _parse_args()
    cache = Path(args.cache).expanduser().resolve()
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True)

    validation_states = _build_ladder(_input_specs(VALIDATION_QUERY_LENGTH))
    errors = _verify_and_dump(validation_states, cache)
    summary: dict[str, object] = {
        "states": len(validation_states),
        "accuracy": {"passed": len(errors), "max_abs_error": max(errors.values()), "kernels": "kernels/"},
    }
    profile_states = _build_ladder(_input_specs(SEQUENCE_LENGTH))
    summary["profile"] = _profile(profile_states, cache)
    (cache / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
