"""One-call profiler-guided refinement for an nkigym kernel."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from nkigym.codegen import render
from nkigym.environment import KernelMDP
from nkigym.ir import KernelIR
from nkigym.profile import profile_metrics
from nkigym.search.codex_policy import CodexPolicyConfig, CodexTransformPolicy
from nkigym.search.engine import ProfilerGuidedRefinement
from nkigym.search.profile_feedback import evaluation_from_profile
from nkigym.search.types import Evaluation, InputSpecs, SearchConfig, SearchResult
from nkigym.transforms import public_transforms

ReasoningEffort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True)
class ProfileEvaluatorConfig:
    """Static profile controls shared by every state in one refinement."""

    host: str
    input_specs: InputSpecs
    neuronx_cc_args: tuple[str, ...]
    lnc: int
    timeout_s: int

    def __post_init__(self) -> None:
        """Reject invalid controls before the first remote profile."""
        if not self.host.strip():
            raise ValueError("profile host must not be empty")
        if self.lnc not in {1, 2}:
            raise ValueError("lnc must be 1 or 2")
        if self.timeout_s < 1:
            raise ValueError("profile timeout must be positive")


class NKIProfileEvaluator:
    """Render and profile one state, maximizing measured MFU."""

    def __init__(self, config: ProfileEvaluatorConfig) -> None:
        """Store immutable workload and backend controls."""
        self.config = config

    def evaluate(self, state: KernelIR, node_id: int, cache_dir: Path) -> Evaluation:
        """Profile one rendered state and preserve failure artifacts."""
        evaluation: Evaluation
        profile_dir = cache_dir / "profile"
        try:
            profile_result = profile_metrics(
                host=self.config.host,
                kernel=render(state),
                func_name=f"nki_{state.func_name}",
                input_specs=self.config.input_specs,
                cache_dir=profile_dir,
                neuronx_cc_args=self.config.neuronx_cc_args,
                lnc=self.config.lnc,
                timeout_s=self.config.timeout_s,
            )
        except RuntimeError as error:
            if not (profile_dir / "result.json").is_file():
                raise
            detail = str(error).strip() or type(error).__name__
            evaluation = Evaluation(
                score=None,
                metrics={"profile_succeeded": False},
                message=f"Neuron profile failed for N{node_id:03d}: {detail}",
            )
        else:
            evaluation = evaluation_from_profile(profile_result, node_id)
        return evaluation


def _workload_guidance(kernel_func: Callable[..., Any], input_specs: InputSpecs) -> str:
    """Describe the kernel and profile inputs to the reasoning policy."""
    inputs = ", ".join(f"{name}={shape}:{dtype}" for name, (shape, dtype) in input_specs.items())
    return (
        f"Optimize nkigym function {kernel_func.__name__} with inputs {inputs}. "
        "Use measured Neuron profiles to select legal transforms."
    )


def run_profiled_refinement(
    kernel_func: Callable[..., Any],
    input_specs: InputSpecs,
    profile_host: str,
    /,
    *,
    target_score: float | None = None,
    trace_dir: Path | None = None,
    neuronx_cc_args: tuple[str, ...] = (),
    reasoning_effort: ReasoningEffort = "max",
    max_reasoning_steps: int | None = None,
    profile_timeout_s: int = 1800,
    policy_timeout_s: int = 600,
    lnc: int = 1,
    codex_executable: str = "codex",
) -> SearchResult:
    """Run measured agentic refinement, retaining artifacts only when requested."""
    with ExitStack() as stack:
        cache_dir = trace_dir
        if cache_dir is None:
            cache_dir = Path(stack.enter_context(TemporaryDirectory(prefix="nkigym-refinement-")))
        evaluator = NKIProfileEvaluator(
            ProfileEvaluatorConfig(
                host=profile_host,
                input_specs=input_specs,
                neuronx_cc_args=neuronx_cc_args,
                lnc=lnc,
                timeout_s=profile_timeout_s,
            )
        )
        policy = CodexTransformPolicy(
            CodexPolicyConfig(
                executable=codex_executable, reasoning_effort=reasoning_effort, timeout_s=policy_timeout_s
            )
        )
        refinement = ProfilerGuidedRefinement(
            environment=KernelMDP(kernel_func, input_specs, transforms=public_transforms()),
            policy=policy,
            evaluator=evaluator,
            config=SearchConfig(
                cache_dir=cache_dir,
                max_reasoning_steps=max_reasoning_steps,
                workload_guidance=_workload_guidance(kernel_func, input_specs),
                target_score=target_score,
            ),
        )
        result = asyncio.run(refinement.run())
    return result


__all__ = ["NKIProfileEvaluator", "ProfileEvaluatorConfig", "ReasoningEffort", "run_profiled_refinement"]
