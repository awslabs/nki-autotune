"""On-box Trn2 evaluator for rendered ``KernelIR`` states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from autotune.runner.output import ProfileOutput
from autotune.runner.types import KernelJob
from autotune.search.types import Evaluation
from nkigym.codegen import render
from nkigym.ir import KernelIR


@dataclass(frozen=True)
class ProfileEvaluatorConfig:
    """Static runner inputs shared by every searched state."""

    input_specs: dict[str, tuple[tuple[int, ...], str]]
    output_shape: tuple[int, ...]
    neuron_platform_target: str
    neuronx_cc_args: tuple[str, ...]
    seed: int


class ProfileRunner(Protocol):
    """Callable surface implemented by :func:`autotune.runner.api.profile`."""

    def __call__(
        self,
        kernels: dict[str, KernelJob],
        cache_dir: str,
        seed: int,
        neuron_platform_target: str,
        collect_detailed_profile: bool,
    ) -> ProfileOutput:
        """Compile and profile a batch of kernel jobs."""
        ...


class NKIProfileEvaluator:
    """Compile and profile one state, maximizing measured MFU."""

    def __init__(self, config: ProfileEvaluatorConfig, profile_runner: ProfileRunner) -> None:
        """Store runner configuration."""
        self.config = config
        self.profile_runner = profile_runner

    def evaluate(self, state: KernelIR, node_id: int, cache_dir: Path) -> Evaluation:
        """Render, compile, benchmark, and convert the runner result to a score."""
        label = f"node_{node_id:03d}"
        job = KernelJob(
            source=render(state),
            func_name=f"nki_{state.func_name}",
            output_shape=self.config.output_shape,
            input_specs=self.config.input_specs,
            neuronx_cc_args=self.config.neuronx_cc_args,
        )
        output = self.profile_runner(
            {label: job},
            cache_dir=str(cache_dir / "profile"),
            seed=self.config.seed,
            neuron_platform_target=self.config.neuron_platform_target,
            collect_detailed_profile=False,
        )
        successes = {row.kernel_name: row for row in output.successes}
        row = successes.get(label)
        if row is not None:
            evaluation = Evaluation(
                score=row.mfu,
                metrics={
                    "mfu_percent": row.mfu,
                    "total_time_s": row.total_time_s,
                    "mbu_percent": row.mbu,
                    "roofline_ceiling_percent": row.roofline_ceiling,
                },
                message=f"Trn2 success: MFU={row.mfu:.2f}%, total={row.total_time_s:.6f}s",
            )
        else:
            evaluation = Evaluation(
                score=None,
                metrics={"wallclock_s": output.elapsed_s},
                message=f"Trn2 failure: {profile_failure_message(output, label)}",
            )
        return evaluation


def profile_failure_message(output: ProfileOutput, label: str) -> str:
    """Return the most specific compiler or runner diagnostic for one job."""
    compiler_log = output.compiler_logs.get(label, "")
    lines = compiler_log.splitlines()
    diagnostic = next(
        (
            line
            for line in reversed(lines)
            if "[NCC_" in line or "Out of memory" in line or "Allocated memory out of bound" in line
        ),
        "",
    )
    if not diagnostic:
        diagnostic = next((line for line in reversed(lines) if " ERROR " in line), "")
    if not diagnostic:
        failure = next(
            (row.hardware_output for row in output.failures if row.kernel_name == label), "runner returned no result"
        )
        diagnostic = failure
    return diagnostic[-1000:]


__all__ = ["NKIProfileEvaluator", "ProfileEvaluatorConfig", "ProfileRunner", "profile_failure_message"]
