"""Profile output formatting and display."""

from dataclasses import dataclass

from autotune.runner.types import ProfileResult


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
    def successes(self) -> list[ProfileResult]:
        """Results that completed without hardware errors."""
        return [r for r in self.results if r.hardware_output.startswith("[")]

    @property
    def failures(self) -> list[ProfileResult]:
        """Results that had hardware errors."""
        return [r for r in self.results if not r.hardware_output.startswith("[")]

    def __str__(self) -> str:
        """Human-readable summary with per-kernel timing table."""
        lines: list[str] = []
        successes = self.successes
        lines.extend(_format_results_table(successes))
        lines.extend(_format_failures(self.failures))
        lines.extend(_format_summary(successes, self))
        return "\n".join(lines)


def _format_results_table(successes: list[ProfileResult]) -> list[str]:
    """Format the timing results table for successful kernels."""
    lines: list[str] = []
    show_mfu = any(r.mfu > 0 for r in successes)
    header = f"{'Kernel':<30} {'min_ms':>10} {'mean_ms':>10} {'p50_ms':>10} {'p99_ms':>10}"
    if show_mfu:
        header += f" {'mfu':>8}"
    for r in sorted(successes, key=lambda r: r.min_ms):
        if not lines:
            lines.append(header)
            lines.append("-" * len(header))
        row = f"{r.kernel_name:<30} {r.min_ms:>10.4f} {r.mean_ms:>10.4f} {r.p50_ms:>10.4f} {r.p99_ms:>10.4f}"
        if show_mfu:
            row += f" {r.mfu:>7.2f}%"
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


def _format_summary(successes: list[ProfileResult], output: ProfileOutput) -> list[str]:
    """Format the summary footer."""
    lines: list[str] = []
    times = [r.min_ms for r in successes]
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
