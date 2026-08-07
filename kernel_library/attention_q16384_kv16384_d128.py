"""Apply, verify, dump, and profile the Q16384/KV16384/D128 attention ladder."""

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
from nkigym.ops.activation import NKIActivation
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.profile import profile, simulate_fp32
from nkigym.transforms import (
    BatchPermutation,
    BatchPermutationOption,
    BufferCompaction,
    BufferCompactionOption,
    BufferPlacement,
    BufferPlacementOption,
    CodeMotion,
    CodeMotionOption,
    CommonSubexpressionElimination,
    CommonSubexpressionEliminationOption,
    CopyPropagation,
    CopyPropagationOption,
    DecomposeBroadcastSubtract,
    DecomposeBroadcastSubtractOption,
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

HEAD_DIM = 128
SEQUENCE_LENGTH = 16384
VALIDATION_QUERY_LENGTH = 512
SEED = 0
ATOL = 5e-3
RTOL = 5e-3
WORKLOAD_NAME = "attention"
SHAPE = "q16384_kv16384_d128"
SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")


def _input_specs(query_length: int, sequence_length: int, head_dim: int) -> InputSpecs:
    """Return pretransposed BF16 attention inputs."""
    return {
        "query": ((head_dim, query_length), "bfloat16"),
        "key": ((head_dim, sequence_length), "bfloat16"),
        "value": ((sequence_length, head_dim), "bfloat16"),
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
    (FusePointwiseReduction(), FusePointwiseReductionOption(pointwise_block_nid=76, reduction_block_nid=79)),
    (FusePointwiseReduction(), FusePointwiseReductionOption(pointwise_block_nid=88, reduction_block_nid=91)),
    (DecomposeBroadcastSubtract(), DecomposeBroadcastSubtractOption(pointwise_block_nid=85)),
    (FuseBroadcastActivation(), FuseBroadcastActivationOption(pointwise_block_nid=85, activation_block_nid=88)),
    (FusePointwiseActivation(), FusePointwiseActivationOption(pointwise_block_nid=94, activation_block_nid=97)),
    (FusePointwiseActivation(), FusePointwiseActivationOption(pointwise_block_nid=135, activation_block_nid=138)),
    (
        CommonSubexpressionElimination(),
        CommonSubexpressionEliminationOption(canonical_block_nid=97, redundant_block_nid=138),
    ),
    (RFactor(), RFactorOption(target_loop_nid=78, factors=(16, 512), target_axis="d2")),
    (RFactor(), RFactorOption(target_loop_nid=90, factors=(16, 512), target_axis="d2")),
    (Reorder(), ReorderOption(outer_nid=126, inner_nid=127)),
    (CodeMotion(), CodeMotionOption(block_nid=115, target_loop_nid=123, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=100, target_loop_nid=123, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=97, target_loop_nid=123, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=165, target_loop_nid=123, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=88, target_loop_nid=123, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=157, target_loop_nid=123, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=82, target_loop_nid=123, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=161, target_loop_nid=123, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=76, target_loop_nid=123, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=73, target_loop_nid=123, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=69, target_loop_nid=123, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=66, target_loop_nid=123, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=125, target_loop_nid=123, index=-1)),
    (CodeMotion(), CodeMotionOption(block_nid=129, target_loop_nid=123, index=-1)),
    (CodeMotion(), CodeMotionOption(block_nid=132, target_loop_nid=123, index=-1)),
    (CodeMotion(), CodeMotionOption(block_nid=141, target_loop_nid=123, index=-1)),
    (CodeMotion(), CodeMotionOption(block_nid=144, target_loop_nid=123, index=-1)),
    (CodeMotion(), CodeMotionOption(block_nid=103, target_loop_nid=123, index=-1)),
    (CodeMotion(), CodeMotionOption(block_nid=106, target_loop_nid=123, index=-1)),
    (Split(), SplitOption(target_nid=68, factors=(16, 512), target_axis="d2")),
    (Split(), SplitOption(target_nid=75, factors=(16, 512), target_axis="d2")),
    (CodeMotion(), CodeMotionOption(block_nid=66, target_loop_nid=71, index=0)),
    (CodeMotion(), CodeMotionOption(block_nid=73, target_loop_nid=71, index=-1)),
    (Split(), SplitOption(target_nid=127, factors=(4, 16), target_axis=None)),
    (RFactor(), RFactorOption(target_loop_nid=170, factor_axis=0)),
    (CodeMotion(), CodeMotionOption(block_nid=76, target_loop_nid=71, index=3)),
    (CopyPropagation(), CopyPropagationOption(copy_block_nid=73, consumer_block_nid=76, consumer_operand="data")),
    (BufferPlacement(), BufferPlacementOption(tensor="online_deferred_factor_reciprocal")),
    (BufferPlacement(), BufferPlacementOption(tensor="online_deferred_numerator")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_output")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_rfactor")),
    (BufferPlacement(), BufferPlacementOption(tensor="psum_output")),
    (BufferPlacement(), BufferPlacementOption(tensor="psum_scores")),
    (BufferPlacement(), BufferPlacementOption(tensor="online_stage1_correction")),
    (BufferPlacement(), BufferPlacementOption(tensor="psum_output_online_chunk")),
    (BufferPlacement(), BufferPlacementOption(tensor="psum_output_online_state")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_row_sum_online_current")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_row_sum_online_chunk")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_row_max_online_current")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_row_max_online_chunk")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_probability_t")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_exp")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_scaled_scores")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_row_max_online_current_negative")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_row_max_online_chunk_rfactor")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_row_sum_online_chunk_rfactor")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_value")),
    (BufferPlacement(), BufferPlacementOption(tensor="sbuf_key")),
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
        EliminateIdentityInitializerOption(initializer_block_nid=66, reduction_block_nid=69, tensor="psum_scores"),
    ),
    (SoftwarePipeline(), SoftwarePipelineOption(loop_nid=123, stages=(0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2))),
    (Split(), SplitOption(target_nid=117, factors=(16, 4), target_axis=None)),
    (BatchPermutation(), BatchPermutationOption(loop_nid=179)),
)

INPUT_SPECS = _input_specs(SEQUENCE_LENGTH, SEQUENCE_LENGTH, HEAD_DIM)
WORKLOAD = Workload(
    input_specs=INPUT_SPECS, f_numpy=f_numpy, f_nkigym=f_nkigym, best_action_ladder=ACTIONS, historical_best_mfu=46.43
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


def _build_ladder(workload: Workload, input_specs: InputSpecs) -> list[KernelIR]:
    """Apply every fixed action through one kernel environment."""
    environment = KernelMDP(
        workload.f_nkigym, input_specs, transforms=[transform for transform, _option in workload.best_action_ladder]
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


def _validation_input_specs(workload: Workload) -> InputSpecs:
    """Use a bounded query length while preserving the selected KV shape."""
    query_shape, _query_dtype = workload.input_specs["query"]
    key_shape, _key_dtype = workload.input_specs["key"]
    query_length = min(VALIDATION_QUERY_LENGTH, query_shape[1])
    return _input_specs(query_length, key_shape[1], query_shape[0])


def _verify_and_dump(
    workload: Workload, input_specs: InputSpecs, states: list[KernelIR], cache: Path
) -> dict[str, float]:
    """Dump and CPU-verify every validation-shaped state."""
    rng = np.random.default_rng(SEED)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in input_specs.items()}
    expected = workload.f_numpy(**inputs)
    errors: dict[str, float] = {}
    for index, (label, state) in enumerate(zip(_labels(workload), states, strict=True)):
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


def _profile(workload: Workload, states: list[KernelIR], cache: Path, host: str) -> dict[str, object]:
    """Profile every production-shaped state through SSH."""
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

    validation_input_specs = _validation_input_specs(WORKLOAD)
    validation_states = _build_ladder(WORKLOAD, validation_input_specs)
    errors = _verify_and_dump(WORKLOAD, validation_input_specs, validation_states, cache)
    summary: dict[str, object] = {
        "workload": WORKLOAD_NAME,
        "shape": SHAPE,
        "states": len(validation_states),
        "accuracy": {"passed": len(errors), "max_abs_error": max(errors.values()), "kernels": "kernels/"},
    }
    profile_states = _build_ladder(WORKLOAD, WORKLOAD.input_specs)
    summary["profile"] = _profile(WORKLOAD, profile_states, cache, args.host)
    (cache / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
