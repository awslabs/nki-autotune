"""Profile fixed skewed-matmul traces with and without transpose rewrites.

Both traces start from the same canonical ``lhs @ rhs`` kernel. The baseline
keeps the explicit Tensor Engine transpose and applies three ordinary layout
transforms. The comparison uses ``TransposeThroughLoad`` on the input and
``TransposeThroughMatmul`` on the output. There is no reasoning policy or
search loop.

Run from the development machine:

    PYTHONPATH=.:nkigym/src:autotune/src \
      python examples/transpose_layout_demo.py \
      --cache /home/weittang/workplace/cache/transpose-layout-demo
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np

from autotune.search.profile_evaluator import ProfileEvaluatorConfig
from autotune.search.ssh_profile_evaluator import SSHNKIProfileEvaluator, SSHProfileConfig
from examples._matmul_workloads import TRANSPOSE_DEMO
from nkigym.codegen import render
from nkigym.environment import Action
from nkigym.ir import KernelIR, build_initial_ir
from nkigym.search.types import Evaluation
from nkigym.synthesis import simulate_fp32
from nkigym.transforms import (
    BufferCompaction,
    BufferCompactionOption,
    BufferLayout,
    BufferLayoutOption,
    CodeMotion,
    CodeMotionOption,
    InsertTransposePair,
    InsertTransposePairOption,
    TransposeThroughLoad,
    TransposeThroughLoadOption,
    TransposeThroughMatmul,
    TransposeThroughMatmulOption,
    TransposeThroughTensorCopy,
    TransposeThroughTensorCopyOption,
)

WORKLOAD = TRANSPOSE_DEMO
M, K = WORKLOAD.input_specs["lhs"][0]
_, N = WORKLOAD.input_specs["rhs"][0]
SEED = 0
NEURON_PLATFORM_TARGET = "trn2"
SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")
MIN_MFU_GAIN_PERCENTAGE_POINTS = 5.0
REMOTE_REPO_SUBDIR = ".cache/nki-autotune-transpose-demo/repo"
REMOTE_CACHE_SUBDIR = ".cache/nki-autotune-transpose-demo/profiles"
REMOTE_ACTIVATE = 'source "$HOME"/venvs/kernel-env/bin/activate'
PROFILE_TIMEOUT_S = 1800

WITHOUT_TRANSPOSE_TRACE: tuple[Action, ...] = (
    (CodeMotion(), CodeMotionOption(block_nid=8, target_loop_nid=5, index=1)),
    (BufferCompaction(), BufferCompactionOption(tensor="psum_lhs_T")),
    (BufferLayout(), BufferLayoutOption(tensor="psum_lhs_T", list_len=8)),
)

WITH_TRANSPOSE_TRACE: tuple[Action, ...] = (
    (TransposeThroughLoad(), TransposeThroughLoadOption(target_nid=1)),
    (InsertTransposePair(), InsertTransposePairOption(consumer_nid=26, operand="src", source="sbuf_prod")),
    (TransposeThroughMatmul(), TransposeThroughMatmulOption(transpose_nid=30)),
    (TransposeThroughTensorCopy(), TransposeThroughTensorCopyOption(transpose_nid=35)),
)


def _parse_args() -> argparse.Namespace:
    """Parse the cache and remote profiler host."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--host", default="gym-1")
    return parser.parse_args()


def _profile_evaluator(args: argparse.Namespace) -> SSHNKIProfileEvaluator:
    """Build the shared SSH-backed Trn2 evaluator."""
    repository = Path(__file__).resolve().parents[1]
    return SSHNKIProfileEvaluator(
        profile_config=ProfileEvaluatorConfig(
            input_specs=WORKLOAD.input_specs,
            output_shape=(M, N),
            neuron_platform_target=NEURON_PLATFORM_TARGET,
            neuronx_cc_args=SCHEDULER_OFF_ARGS,
            seed=SEED,
        ),
        ssh_config=SSHProfileConfig(
            host=args.host,
            local_repo=repository,
            remote_repo_subdir=REMOTE_REPO_SUBDIR,
            remote_cache_subdir=REMOTE_CACHE_SUBDIR,
            remote_activate=REMOTE_ACTIVATE,
            timeout_s=PROFILE_TIMEOUT_S,
        ),
    )


def _apply_trace(trace: tuple[Action, ...]) -> KernelIR:
    """Apply one fixed transform trace to a fresh canonical kernel."""
    state = build_initial_ir(WORKLOAD.f_nkigym, WORKLOAD.input_specs)
    for transform, option in trace:
        state = transform.apply(state, option)
    return state


def _simulate(state: KernelIR, cache_dir: Path, module_name: str) -> float:
    """Write and fp32-simulate one final kernel."""
    cache_dir.mkdir(parents=True)
    kernel_path = cache_dir / "kernel.py"
    kernel_path.write_text(render(state), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import generated kernel from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(SEED)
    inputs = {
        name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in WORKLOAD.input_specs.items()
    }
    expected = WORKLOAD.f_numpy(**inputs)
    generated = getattr(module, f"nki_{WORKLOAD.f_nkigym.__name__}")
    actual = np.asarray(simulate_fp32(generated)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
    return float(np.abs(actual - expected).max())


def _main() -> None:
    """Build, validate, profile, and compare the two fixed traces."""
    args = _parse_args()
    cache_dir = Path(args.cache).expanduser().resolve()
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True)

    traces = {"without_transpose": WITHOUT_TRANSPOSE_TRACE, "with_transpose": WITH_TRANSPOSE_TRACE}
    states = {name: _apply_trace(trace) for name, trace in traces.items()}
    evaluator = _profile_evaluator(args)
    evaluations: dict[str, Evaluation] = {}
    fp32_errors: dict[str, float] = {}

    for node_id, (name, state) in enumerate(states.items()):
        trace_dir = cache_dir / name
        fp32_errors[name] = _simulate(state, trace_dir, name)
        evaluations[name] = evaluator.evaluate(state, node_id, trace_dir)
        evaluation = evaluations[name]
        (trace_dir / "evaluation.json").write_text(
            json.dumps(
                {"score": evaluation.score, "metrics": evaluation.metrics, "message": evaluation.message}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )

    run_summaries: dict[str, dict[str, object]] = {}
    mfu_by_run: dict[str, float] = {}
    latency_by_run: dict[str, float] = {}
    for name, evaluation in evaluations.items():
        score = evaluation.score
        latency = evaluation.metrics.get("total_time_s")
        if score is None:
            raise RuntimeError(f"{name} failed to compile and profile: {evaluation.message}")
        if isinstance(latency, bool) or not isinstance(latency, (float, int)):
            raise RuntimeError(f"{name} has invalid total_time_s metric: {latency!r}")
        mfu_by_run[name] = score
        latency_by_run[name] = float(latency)
        run_summaries[name] = {
            "trace": [f"{type(transform).__name__}: {option!r}" for transform, option in traces[name]],
            "mfu_percent": score,
            "latency_us": float(latency) * 1e6,
            "fp32_max_abs_error": fp32_errors[name],
            "kernel": str(cache_dir / name / "kernel.py"),
        }

    gain = mfu_by_run["with_transpose"] - mfu_by_run["without_transpose"]
    if gain <= MIN_MFU_GAIN_PERCENTAGE_POINTS:
        raise RuntimeError(
            f"transpose trace MFU gain {gain:.2f} did not exceed "
            f"{MIN_MFU_GAIN_PERCENTAGE_POINTS:.2f} percentage points"
        )

    summary = {
        "workload": {
            "operation": "lhs @ rhs",
            "lhs_shape": list(WORKLOAD.input_specs["lhs"][0]),
            "rhs_shape": list(WORKLOAD.input_specs["rhs"][0]),
            "output_shape": [M, N],
            "dtype": "bfloat16",
        },
        "without_transpose": run_summaries["without_transpose"],
        "with_transpose": run_summaries["with_transpose"],
        "mfu_gain_percentage_points": gain,
        "latency_speedup": latency_by_run["without_transpose"] / latency_by_run["with_transpose"],
    }
    (cache_dir / "demonstration.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
