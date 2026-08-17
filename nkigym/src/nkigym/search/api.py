"""One-call iterative schedule refinement."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from nkigym.environment import KernelMDP
from nkigym.profile import InputSpecs
from nkigym.search.engine import IterativeRefinement, SearchConfig
from nkigym.search.types import Policy, SearchResult
from nkigym.transforms import public_transforms


def run_search(
    kernel_func: Callable[..., Any],
    input_specs: InputSpecs,
    profile_host: str,
    /,
    *,
    policy: Policy,
    max_transforms_per_evaluation: int = 10,
    target_latency_ms: float | None = None,
    trace_dir: Path | None = None,
    neuronx_cc_args: tuple[str, ...] = (),
    max_evaluations: int = 128,
    profile_timeout_s: int = 1800,
    lnc: int = 1,
) -> SearchResult:
    """Run the fixed refinement process with a caller-selected policy."""
    with ExitStack() as stack:
        output_dir = trace_dir
        if output_dir is None:
            output_dir = Path(stack.enter_context(TemporaryDirectory(prefix="nkigym-search-")))
        refinement = IterativeRefinement(
            environment=KernelMDP(kernel_func, input_specs, public_transforms()),
            policy=policy,
            config=SearchConfig(
                trace_dir=output_dir,
                profile_host=profile_host,
                input_specs=input_specs,
                neuronx_cc_args=neuronx_cc_args,
                lnc=lnc,
                profile_timeout_s=profile_timeout_s,
                max_transforms_per_evaluation=max_transforms_per_evaluation,
                max_evaluations=max_evaluations,
                target_latency_ms=target_latency_ms,
            ),
        )
        result = refinement.run()
    return result


__all__ = ["run_search"]
