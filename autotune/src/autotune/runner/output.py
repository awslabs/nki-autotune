"""Profile output formatting and display."""

from dataclasses import dataclass
from typing import NamedTuple

from autotune.runner.types import ProfileResult, profiler_percent


class SuccessRow(NamedTuple):
    """Kernel result narrowed to non-None profiler fields used by the table."""

    kernel_name: str
    total_time_s: float
    mfu: float
    mbu: float | None
    roofline_ceiling: float | None


def _collect_successes(results: list[ProfileResult]) -> list[SuccessRow]:
    """Narrow to kernels that compiled, ran on HW, and produced profiler data."""
    rows: list[SuccessRow] = []
    for r in results:
        if not r.hardware_output.startswith("["):
            continue
        total = (r.profiler_summary or {}).get("total_time")
        if not isinstance(total, (int, float)):
            continue
        mfu = profiler_percent(r.profiler_summary, "mfu_estimated_percent")
        if mfu is None:
            continue
        rows.append(
            SuccessRow(
                kernel_name=r.kernel_name,
                total_time_s=float(total),
                mfu=mfu,
                mbu=profiler_percent(r.profiler_summary, "mbu_estimated_percent"),
                roofline_ceiling=profiler_percent(r.profiler_summary, "mfu_max_achievable_estimated_percent"),
            )
        )
    return rows


@dataclass
class ProfileOutput:
    """Results from a remote profiling run.

    Attributes:
        results: Per-kernel profiling results.
        compiler_logs: Map of kernel name to compiler log text.
        elapsed_s: Total wallclock time in seconds.
        hosts: SSH hostnames used for profiling.
        cache_dir: Path to cache directory, if saved.
    """

    results: list[ProfileResult]
    compiler_logs: dict[str, str]
    elapsed_s: float
    hosts: list[str]
    cache_dir: str = ""

    @property
    def successes(self) -> list[SuccessRow]:
        """Kernels that passed CPU sim, compiled, and produced HW timings."""
        return _collect_successes(self.results)

    @property
    def failures(self) -> list[ProfileResult]:
        """Kernels that failed CPU sim, compile, or hardware execution."""
        return [r for r in self.results if r.profiler_summary is None or not r.hardware_output.startswith("[")]

    def __str__(self) -> str:
        """Human-readable summary with per-kernel timing table."""
        lines: list[str] = []
        successes = self.successes
        lines.extend(_format_results_table(successes))
        lines.extend(_format_failures(self.failures))
        lines.extend(_format_summary(successes, self))
        return "\n".join(lines)


def _format_results_table(successes: list[SuccessRow]) -> list[str]:
    """Format the results table for successful kernels."""
    lines: list[str] = []
    show_mfu = any(s.mfu > 0 for s in successes)
    show_roofline = any(s.mbu is not None or s.roofline_ceiling is not None for s in successes)
    header = f"{'Kernel':<30} {'total_s':>10}"
    if show_mfu:
        header += f" {'mfu':>8}"
    if show_roofline:
        header += f" {'mbu':>8} {'ceiling':>8}"
    for s in sorted(successes, key=lambda s: s.total_time_s):
        if not lines:
            lines.append(header)
            lines.append("-" * len(header))
        row = f"{s.kernel_name:<30} {s.total_time_s:>10.6f}"
        if show_mfu:
            row += f" {s.mfu:>7.2f}%"
        if show_roofline:
            row += f" {_fmt_pct(s.mbu):>8} {_fmt_pct(s.roofline_ceiling):>8}"
        lines.append(row)
    return lines


def _fmt_pct(value: float | None) -> str:
    """Format a percentage with '-' for None."""
    return "-" if value is None else f"{value:.2f}%"


def _format_failures(failures: list[ProfileResult]) -> list[str]:
    """Format the failure summary lines."""
    lines: list[str] = []
    for i, r in enumerate(failures):
        if i == 0:
            lines.append(f"\n{len(failures)} failures:")
        error = r.hardware_output
        assert isinstance(error, str)
        lines.append(f"  {r.kernel_name}: {error}")
    return lines


def _format_summary(successes: list[SuccessRow], output: ProfileOutput) -> list[str]:
    """Format the summary footer."""
    lines: list[str] = []
    times = [s.total_time_s for s in successes]
    lines.append("\nSummary:")
    lines.append(f"  Succeeded:  {len(successes)}/{len(output.results)}")
    lines.append(f"  Hosts:      {len(output.hosts)} ({', '.join(output.hosts)})")
    lines.append(f"  Wallclock:  {output.elapsed_s:.1f}s")
    if times:
        lines.append(f"  Best:       {min(times):.6f} s ({successes[0].kernel_name})")
        lines.append(f"  Worst:      {max(times):.6f} s")
    if output.cache_dir:
        lines.append(f"  Cache:      {output.cache_dir}")
    return lines
