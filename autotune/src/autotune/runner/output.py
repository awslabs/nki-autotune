"""Profile output formatting, display, and cache-layout writers."""

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from autotune.runner.types import KernelJob, ProfileResult, profiler_percent

_PROFILE_SUMMARY_FILE = "profile_summary.json"
_PROFILE_DETAILED_FILE = "profile.json"
_NEFF_FILE = "file.neff"
_NTFF_FILE = "profile.ntff"


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
    """Results from a local profiling run.

    Attributes:
        results: Per-kernel profiling results.
        compiler_logs: Map of kernel name to compiler log text.
        elapsed_s: Total wallclock time in seconds.
        cache_dir: Path to cache directory, if saved.
    """

    results: list[ProfileResult]
    compiler_logs: dict[str, str]
    elapsed_s: float
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
    lines.append(f"  Wallclock:  {output.elapsed_s:.1f}s")
    if times:
        lines.append(f"  Best:       {min(times):.6f} s ({successes[0].kernel_name})")
        lines.append(f"  Worst:      {max(times):.6f} s")
    if output.cache_dir:
        lines.append(f"  Cache:      {output.cache_dir}")
    return lines


def write_kernel_sources(cache_dir: str, kernels: dict[str, KernelJob]) -> None:
    """Write kernel source files to ``<cache>/<stem>/<stem>.py``."""
    for kname, job in kernels.items():
        stem = Path(kname).stem
        variant_dir = os.path.join(cache_dir, stem)
        os.makedirs(variant_dir, exist_ok=True)
        with open(os.path.join(variant_dir, f"{stem}.py"), "w") as f:
            f.write(job.source)


def write_compiler_logs(cache_dir: str, compiler_logs: dict[str, str]) -> None:
    """Write compiler logs to ``<cache>/<stem>/log-neuron-cc.txt``."""
    for kname, log_text in compiler_logs.items():
        stem = Path(kname).stem
        variant_dir = os.path.join(cache_dir, stem)
        os.makedirs(variant_dir, exist_ok=True)
        with open(os.path.join(variant_dir, "log-neuron-cc.txt"), "w") as f:
            f.write(log_text)


def _kernel_sort_key(kernel_name: str) -> tuple[int, int, str]:
    """Natural sort key for kernel names.

    Names shaped like ``<prefix>_<N>.py`` sort numerically by ``N`` (so
    ``kernel_2.py`` precedes ``kernel_10.py``); names without a numeric
    suffix sort alphabetically after all numerically-sorted names.
    """
    stem = Path(kernel_name).stem
    tail = stem.rsplit("_", 1)[-1]
    key = (0, int(tail), stem) if tail.isdigit() else (1, 0, stem)
    return key


def _write_per_kernel_profiles(cache_dir: str, results: list[ProfileResult]) -> None:
    """Move per-kernel profiler dicts into their subfolders.

    Always writes ``<cache>/<stem>/profile_summary.json``. When detailed
    collection was on, the subfolder also receives ``profile.json`` (full
    trace), ``file.neff`` (compiled binary), and ``profile.ntff`` (raw trace)
    so offline ``neuron-profile view`` runs work without a gym host.
    """
    for r in results:
        stem = Path(r.kernel_name).stem
        variant_dir = os.path.join(cache_dir, stem)
        os.makedirs(variant_dir, exist_ok=True)
        if r.profiler_summary is not None:
            with open(os.path.join(variant_dir, _PROFILE_SUMMARY_FILE), "w") as f:
                json.dump(r.profiler_summary, f, indent=2)
        if r.profile_detailed is not None:
            with open(os.path.join(variant_dir, _PROFILE_DETAILED_FILE), "w") as f:
                json.dump(r.profile_detailed, f)
        if r.neff_b64 is not None:
            with open(os.path.join(variant_dir, _NEFF_FILE), "wb") as f:
                f.write(base64.b64decode(r.neff_b64))
        if r.ntff_b64 is not None:
            with open(os.path.join(variant_dir, _NTFF_FILE), "wb") as f:
                f.write(base64.b64decode(r.ntff_b64))


def _kernel_index_row(
    r: ProfileResult, total_time_s: float | None, mfu: float | None, mbu: float | None, ceiling: float | None
) -> dict:
    """Slim per-kernel index row for results.json — no embedded dicts."""
    stem = Path(r.kernel_name).stem
    return {
        "kernel_name": r.kernel_name,
        "kernel_path": f"{stem}/{stem}.py",
        "hardware_output": r.hardware_output,
        "total_time_s": total_time_s,
        "mfu_estimated_percent": mfu,
        "mbu_estimated_percent": mbu,
        "mfu_max_achievable_estimated_percent": ceiling,
    }


def write_results_json(
    cache_dir: str, num_kernels: int, results: list[ProfileResult], wallclock_s: float
) -> None:
    """Write results.json (index) and split per-kernel JSONs into subfolders."""
    _write_per_kernel_profiles(cache_dir, results)

    extracted: dict[str, tuple[float | None, float | None, float | None, float | None]] = {}
    success = sbuf_oom = psum_oom = 0
    for r in results:
        total = (r.profiler_summary or {}).get("total_time")
        total_s = float(total) if isinstance(total, (int, float)) else None
        mfu = profiler_percent(r.profiler_summary, "mfu_estimated_percent")
        mbu = profiler_percent(r.profiler_summary, "mbu_estimated_percent")
        ceiling = profiler_percent(r.profiler_summary, "mfu_max_achievable_estimated_percent")
        extracted[r.kernel_name] = (total_s, mfu, mbu, ceiling)
        hw_ok = total_s is not None
        if hw_ok:
            success += 1
        if not hw_ok and "Out of memory in sbuf" in r.hardware_output:
            sbuf_oom += 1
        if not hw_ok and "Out of memory in psum" in r.hardware_output:
            psum_oom += 1

    successes = [(r, extracted[r.kernel_name]) for r in results if extracted[r.kernel_name][0] is not None]
    times = [e[0] for _, e in successes]
    mfus = [e[1] for _, e in successes if e[1] is not None]
    mbus = [e[2] for _, e in successes if e[2] is not None]

    kernel_entries = [
        _kernel_index_row(r, *extracted[r.kernel_name])
        for r in sorted(results, key=lambda r: _kernel_sort_key(r.kernel_name))
    ]

    best_kernel = min(successes, key=lambda s: s[1][0])[0].kernel_name if successes else None
    worst_kernel = max(successes, key=lambda s: s[1][0])[0].kernel_name if successes else None
    results_data = {
        "metadata": {"num_kernels": num_kernels, "wallclock_s": wallclock_s},
        "metrics": {
            "best_total_time_s": min(times) if times else None,
            "worst_total_time_s": max(times) if times else None,
            "best_kernel": best_kernel,
            "worst_kernel": worst_kernel,
            "best_mfu": max(mfus) if mfus else None,
            "worst_mfu": min(mfus) if mfus else None,
            "best_mbu": max(mbus) if mbus else None,
            "worst_mbu": min(mbus) if mbus else None,
            "success": success,
            "sbuf_oom": sbuf_oom,
            "psum_oom": psum_oom,
        },
        "kernels": kernel_entries,
    }
    with open(os.path.join(cache_dir, "results.json"), "w") as f:
        json.dump(results_data, f, indent=2)
