"""Profile output formatting and display."""

from dataclasses import dataclass
from typing import NamedTuple

from autotune.runner.types import ProfileResult


class SuccessRow(NamedTuple):
    """Kernel result with guaranteed non-None timing/MFU fields."""

    kernel_name: str
    min_ms: float
    mean_ms: float
    p50_ms: float
    p99_ms: float
    mfu: float


def _collect_successes(results: list[ProfileResult]) -> list[SuccessRow]:
    """Narrow to kernels that compiled, ran on HW, and produced timings."""
    rows: list[SuccessRow] = []
    for r in results:
        if not r.hardware_output.startswith("["):
            continue
        if r.min_ms is None or r.mean_ms is None or r.p50_ms is None or r.p99_ms is None or r.mfu is None:
            continue
        rows.append(
            SuccessRow(
                kernel_name=r.kernel_name,
                min_ms=r.min_ms,
                mean_ms=r.mean_ms,
                p50_ms=r.p50_ms,
                p99_ms=r.p99_ms,
                mfu=r.mfu,
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
        return [r for r in self.results if r.min_ms is None or not r.hardware_output.startswith("[")]

    def __str__(self) -> str:
        """Human-readable summary with per-kernel timing table."""
        lines: list[str] = []
        successes = self.successes
        lines.extend(_format_results_table(successes))
        lines.extend(_format_failures(self.failures))
        lines.extend(_format_summary(successes, self))
        return "\n".join(lines)


def _format_results_table(successes: list[SuccessRow]) -> list[str]:
    """Format the timing results table for successful kernels."""
    lines: list[str] = []
    show_mfu = any(s.mfu > 0 for s in successes)
    header = f"{'Kernel':<30} {'min_ms':>10} {'mean_ms':>10} {'p50_ms':>10} {'p99_ms':>10}"
    if show_mfu:
        header += f" {'mfu':>8}"
    for s in sorted(successes, key=lambda s: s.min_ms):
        if not lines:
            lines.append(header)
            lines.append("-" * len(header))
        row = f"{s.kernel_name:<30} {s.min_ms:>10.4f} {s.mean_ms:>10.4f} {s.p50_ms:>10.4f} {s.p99_ms:>10.4f}"
        if show_mfu:
            row += f" {s.mfu:>7.2f}%"
        lines.append(row)
    return lines


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
    times = [s.min_ms for s in successes]
    lines.append(f"\nSummary:")
    lines.append(f"  Succeeded:  {len(successes)}/{len(output.results)}")
    lines.append(f"  Hosts:      {len(output.hosts)} ({', '.join(output.hosts)})")
    lines.append(f"  Wallclock:  {output.elapsed_s:.1f}s")
    if times:
        lines.append(f"  Best:       {min(times):.4f} ms ({successes[0].kernel_name})")
        lines.append(f"  Worst:      {max(times):.4f} ms")
        lines.append(f"  Mean:       {sum(times) / len(times):.4f} ms")
    if output.cache_dir:
        lines.append(f"  Cache:      {output.cache_dir}")
    return lines
