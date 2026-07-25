"""On-box Trn2 evaluator for rendered ``KernelIR`` states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from autotune.runner.output import ProfileOutput
from autotune.runner.types import KernelJob, profiler_percent
from nkigym.codegen import render
from nkigym.ir import KernelIR
from nkigym.search.types import Evaluation, EvaluationMetric


@dataclass(frozen=True)
class ProfileEvaluatorConfig:
    """Static runner inputs shared by every profiled state."""

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
        return evaluation_from_profile_output(output, label)


def evaluation_from_profile_output(output: ProfileOutput, label: str) -> Evaluation:
    """Convert one labeled runner result into search feedback."""
    return evaluations_from_profile_output(output, (label,))[label]


def evaluations_from_profile_output(output: ProfileOutput, labels: tuple[str, ...]) -> dict[str, Evaluation]:
    """Convert labeled runner results into profiler feedback."""
    successes = {row.kernel_name: row for row in output.successes}
    results = {result.kernel_name: result for result in output.results}
    evaluations: dict[str, Evaluation] = {}
    for label in labels:
        row = successes.get(label)
        if row is not None:
            result = results.get(label)
            summary = result.profiler_summary if result is not None else None
            metrics = profile_summary_metrics(summary)
            metrics.update(
                {
                    "mfu_percent": row.mfu,
                    "total_time_s": row.total_time_s,
                    "mbu_percent": row.mbu,
                    "roofline_ceiling_percent": row.roofline_ceiling,
                }
            )
            evaluation = Evaluation(
                score=row.mfu,
                metrics=metrics,
                message=f"Trn2 success: MFU={row.mfu:.2f}%, total={row.total_time_s:.6f}s",
            )
        else:
            evaluation = Evaluation(
                score=None,
                metrics={"wallclock_s": output.elapsed_s},
                message=f"Trn2 failure: {profile_failure_message(output, label)}",
            )
        evaluations[label] = evaluation
    return evaluations


def profile_summary_metrics(summary: dict | None) -> dict[str, EvaluationMetric]:
    """Extract policy-relevant utilization and instruction counters."""
    metrics: dict[str, EvaluationMetric] = {}
    percent_fields = {
        "tensor_engine_active_percent": "tensor_engine_active_time_percent",
        "vector_engine_active_percent": "vector_engine_active_time_percent",
        "dma_active_percent": "dma_active_time_percent",
        "gpsimd_engine_active_percent": "gpsimd_engine_active_time_percent",
        "total_active_percent": "total_active_time_percent",
        "throttle_average_limit_percent": "throttle_avg_util_limit_nc0_percent",
    }
    for metric_name, profiler_name in percent_fields.items():
        value = profiler_percent(summary, profiler_name)
        if value is not None:
            metrics[metric_name] = value
    counter_fields = {
        "matmul_instruction_count": "matmul_instruction_count",
        "tensor_engine_instruction_count": "tensor_engine_instruction_count",
        "vector_engine_instruction_count": "vector_engine_instruction_count",
        "dynamic_dma_packet_count": "software_dynamic_dma_packet_count",
        "hbm_read_bytes": "hbm_read_bytes",
        "hbm_write_bytes": "hbm_write_bytes",
    }
    if summary is not None:
        for metric_name, profiler_name in counter_fields.items():
            value = summary.get(profiler_name)
            if isinstance(value, (int, float)):
                metrics[metric_name] = value
    return metrics


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


__all__ = [
    "NKIProfileEvaluator",
    "ProfileEvaluatorConfig",
    "ProfileRunner",
    "evaluation_from_profile_output",
    "evaluations_from_profile_output",
    "profile_failure_message",
    "profile_summary_metrics",
]
