"""Neuron-profiler feedback refinement for transposed-LHS matmul.

The loop profiles the canonical kernel, asks GPT for one legal next transform,
applies it, profiles the result on ``gym-1``, and repeats.

Run from the development machine:

    PYTHONPATH=.:nkigym/src:autotune/src \
      python examples/matmul_lhsT_rhs_agentic_search.py --cache /tmp/agentic-search
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path

import numpy as np

from autotune.search.profile_evaluator import ProfileEvaluatorConfig
from autotune.search.ssh_profile_evaluator import SSHNKIProfileEvaluator, SSHProfileConfig
from examples._matmul_workloads import LHS_T_RHS, TRANSFORMS
from nkigym.codegen import render
from nkigym.environment import KernelMDP
from nkigym.search import ProfilerGuidedRefinement, SearchConfig, SearchResult
from nkigym.search.codex_policy import CodexPolicyConfig, CodexTransformPolicy
from nkigym.synthesis import simulate_fp32

M = 2048
N = 2048
SEED = 0
WORKLOAD = LHS_T_RHS
NEURON_PLATFORM_TARGET = "trn2"
SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")
MATMUL_GUIDANCE = """\
Workload: bf16 C[M,N] = lhs_T[K,M].T @ rhs[K,N], with K=M=N=2048 on one
Trn2 NeuronCore. Maximize measured MFU; no reference schedule or expected score
is available.

Generic NKI constraints and semantics:
- Tensor partition extents are at most 128. nc_matmul contracts a K tile of
  128 and produces an output tile with partition extent 128 and free extent up
  to 512.
- Inputs originate in shared HBM, DMA loads stage them in SBUF, matrix
  multiplication accumulates in fp32 PSUM, and output returns through SBUF to
  shared HBM.
- Reordering and placement change reuse, live ranges, transfer frequency, and
  overlap. BufferCompaction materializes a moved buffer's tighter scope and
  access bounding box.
- An on-chip buffer's list_len refactorizes its existing tile axis into
  separate ndarray allocations. SoftwarePipeline instead derives versions
  from producer and consumer stages.
- Transform legality guarantees dependence preservation, not hardware resource
  fit. Compilation failures and measured utilization are feedback.

Use the current Neuron profile to choose one next transform with a concrete
performance hypothesis.
"""


def _parse_args() -> argparse.Namespace:
    """Parse policy, remote profiler, and iteration controls."""
    parser = argparse.ArgumentParser(description="Refine canonical matmul using GPT decisions and Neuron profiles.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--host", default="gym-1")
    parser.add_argument("--model", default="openai.gpt-5.6-sol")
    parser.add_argument("--model-provider", default="amazon-bedrock")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"), default="max")
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--policy-timeout-s", type=int, default=900)
    parser.add_argument("--profile-timeout-s", type=int, default=1800)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--remote-repo-subdir", default=".cache/nki-autotune-agentic/repo")
    parser.add_argument("--remote-cache-subdir", default=".cache/nki-autotune-agentic/profiles")
    parser.add_argument("--remote-activate", default='source "$HOME"/venvs/kernel-env/bin/activate')
    return parser.parse_args()


def _profile_evaluator(args: argparse.Namespace) -> SSHNKIProfileEvaluator:
    """Build the SSH-backed Trn2 evaluator."""
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
            remote_repo_subdir=args.remote_repo_subdir,
            remote_cache_subdir=args.remote_cache_subdir,
            remote_activate=args.remote_activate,
            timeout_s=args.profile_timeout_s,
        ),
    )


async def _run_search(cache_dir: Path, args: argparse.Namespace, evaluator: SSHNKIProfileEvaluator) -> SearchResult:
    """Run one canonical linear refinement."""
    policy = CodexTransformPolicy(
        CodexPolicyConfig(
            executable=args.codex_executable,
            model=args.model,
            model_provider=args.model_provider,
            reasoning_effort=args.reasoning_effort,
            timeout_s=args.policy_timeout_s,
        )
    )
    refinement = ProfilerGuidedRefinement(
        environment=KernelMDP(WORKLOAD.f_nkigym, WORKLOAD.input_specs, transforms=TRANSFORMS),
        policy=policy,
        evaluator=evaluator,
        config=SearchConfig(cache_dir=cache_dir, max_iterations=args.max_iterations, workload_guidance=MATMUL_GUIDANCE),
    )
    return await refinement.run()


def _simulate_selected(source: str, cache_dir: Path) -> float:
    """FP32-simulate the selected generated kernel and return max error."""
    kernel_path = cache_dir / "selected_kernel.py"
    kernel_path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("agentic_selected_kernel", kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import selected kernel from {kernel_path}")
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


def _search_summary(result: SearchResult) -> dict[str, object]:
    """Return the selected state and measured refinement history."""
    best = result.best_node
    return {
        "selected_node_id": best.node_id,
        "current_node_id": result.current_node.node_id,
        "best_mfu_percent": best.evaluation.score,
        "transforms_applied": result.transforms_applied,
        "evaluations_run": result.evaluations_run,
        "finish_reason": result.finish_reason,
        "measured_history": [
            {
                "node_id": node.node_id,
                "action": node.action_description,
                "score": node.evaluation.score,
                "message": node.evaluation.message,
            }
            for node in result.nodes
        ],
    }


def _main() -> None:
    """Run refinement and validate the best measured kernel."""
    args = _parse_args()
    cache_dir = Path(args.cache).expanduser().resolve() / "matmul_lhsT_rhs" / "agentic_search"
    result = asyncio.run(_run_search(cache_dir, args, _profile_evaluator(args)))
    max_abs_error = _simulate_selected(render(result.best_node.state), cache_dir)
    summary = {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "fp32_max_abs_error": max_abs_error,
        **_search_summary(result),
    }
    (cache_dir / "demonstration.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
