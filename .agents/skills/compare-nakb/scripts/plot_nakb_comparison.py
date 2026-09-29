"""Read saved NAKB/NKIGym measurements and export a vector PDF figure.

Run with the kernel environment:
python .agents/skills/compare-nakb/scripts/plot_nakb_comparison.py.
The development session and explicit ladder benchmark publish the input JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from typing import cast

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY))

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullLocator

from kernel_library.nakb_comparison import RESULTS_DIRECTORY, Measurements, validate_measurements

COLORS = ("#0072B2", "#D55E00")
INK = "#202B38"
MUTED = "#536170"


@dataclass(frozen=True)
class Measurement:
    """One configuration's matched latencies, in microseconds."""

    name: str
    nakb_us: float
    nkigym_us: float


def load_measurements(source: Path) -> tuple[list[Measurement], Measurements, str]:
    """Validate saved measurements and sort by decreasing NKIGym latency reduction."""
    if not source.is_file():
        raise FileNotFoundError(
            f"No measurements at {source}. Run test/test_nkigym_ladder.py with the required host options "
            "or select a saved run with --measurements PATH."
        )
    content = source.read_bytes()
    snapshot = cast(Measurements, json.loads(content))
    validate_measurements(snapshot, require_complete=True)
    rows = [
        Measurement(name, raw["nakb_latency_ms"] * 1000, raw["nkigym_latency_ms"] * 1000)
        for name, raw in snapshot["workloads"].items()
    ]
    rows.sort(key=lambda row: (row.nkigym_us / row.nakb_us, row.name))
    return rows, snapshot, sha256(content).hexdigest()


def measurement_status(snapshot: Measurements) -> str:
    """Describe the recorded run provenance without implying fresh validation."""
    status = "mixed-run diagnostic"
    if snapshot["complete_single_run"]:
        status = "single-run diagnostic" if snapshot["diagnostic_only"] else "one complete run"
    return status


def style_axis(axis: Axes) -> None:
    """Give quantitative panels light grids and unobtrusive borders."""
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#BCC5CE")
        axis.spines[side].set_linewidth(0.6)
    axis.tick_params(axis="both", length=3, width=0.6, color="#9DA9B5")


def format_latency(value: float) -> str:
    """Format bar annotations in microseconds, using k for thousands."""
    result = f"{value:.1f}" if value < 1000 else f"{value / 1000:.2f}k"
    return result


def format_configuration(name: str) -> str:
    """Wrap long IDs at an underscore while preserving every character."""
    result = name
    if len(name) > 20:
        boundary = name.rfind("_", 0, 20) + 1
        result = f"{name[:boundary]}\n{name[boundary:]}"
    return result


def draw_aggregate(figure: Figure, rows: list[Measurement]) -> None:
    """Compare arithmetic mean relative latencies with equal case weights."""
    mean_ratio = fmean(row.nkigym_us / row.nakb_us for row in rows)
    faster = sum(row.nkigym_us < row.nakb_us for row in rows)
    slower = sum(row.nkigym_us > row.nakb_us for row in rows)
    figure.text(0.060, 0.922, "Aggregate comparison", fontsize=15, weight="bold", color=INK)
    axis = figure.add_axes((0.105, 0.853, 0.335, 0.056))
    style_axis(axis)
    bars = axis.barh((1, 0), (1.0, mean_ratio), height=0.52, color=COLORS)
    axis.bar_label(bars, labels=("1.000×", f"{mean_ratio:.3f}×"), padding=7, fontsize=11, weight="bold")
    axis.set_yticks((1, 0), labels=("NAKB", "NKIGym"), fontsize=11)
    axis.set_xlim(0, max(1.0, mean_ratio) * 1.23)
    axis.set_xticks((0, 0.5, 1.0), labels=("0", "0.5", "1.0"), fontsize=9)
    axis.grid(axis="x", color="#D8DFE5", linewidth=0.6)
    axis.set_xlabel("Mean(latency / NAKB latency) · lower is better", fontsize=10, labelpad=5)
    change = 100 * (mean_ratio - 1)
    direction = "higher" if change >= 0 else "lower"
    figure.text(0.505, 0.883, f"{abs(change):.2f}% {direction}", fontsize=27, weight="bold", color=COLORS[1])
    figure.text(0.505, 0.857, "Mean relative latency vs. NAKB", fontsize=11, color=MUTED)
    figure.text(0.790, 0.883, f"{faster} / {len(rows)} faster", fontsize=22, weight="bold", color=INK)
    figure.text(0.790, 0.857, f"{slower} slower · all configurations included", fontsize=11, color=MUTED)
    figure.text(
        0.060,
        0.818,
        "Each configuration has equal weight in the aggregate. "
        "Individual bars show measured latency; every panel uses the same logarithmic scale.",
        fontsize=10,
        color=MUTED,
    )


def draw_kernel_panel(
    figure: Figure, rows: list[Measurement], panel: int, first_rank: int, panel_count: int, exponents: tuple[int, int]
) -> None:
    """Draw consecutive configurations by decreasing latency reduction."""
    height = 0.420 / panel_count
    bottom = 0.787 - 0.772 * panel / panel_count - height
    axis = figure.add_axes((0.060, bottom, 0.922, height))
    style_axis(axis)
    positions = list(range(len(rows)))
    axis.set_yscale("log")
    axis.set_ylim(10 ** exponents[0], 2 * 10 ** exponents[1])
    axis.set_xlim(-0.7, 31.7)
    axis.set_yticks([10**exponent for exponent in range(exponents[0], exponents[1] + 1)])
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, position: f"{value:,.0f}" if value >= 1 else f"{value:g}")
    )
    axis.yaxis.set_minor_locator(NullLocator())
    axis.grid(axis="y", color="#D8DFE5", linewidth=0.6)
    axis.set_ylabel("Latency (µs, log scale)", fontsize=10)
    for values, offset, color in (
        ([row.nakb_us for row in rows], -0.19, COLORS[0]),
        ([row.nkigym_us for row in rows], 0.19, COLORS[1]),
    ):
        bars = axis.bar(
            [position + offset for position in positions],
            values,
            width=0.35,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
        axis.bar_label(
            bars, labels=[format_latency(value) for value in values], padding=3, fontsize=6.7, rotation=90, color=INK
        )
    axis.set_xticks(positions, labels=[format_configuration(row.name) for row in rows])
    axis.tick_params(axis="y", labelsize=9)
    for label in axis.get_xticklabels():
        label.set_rotation(68)
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
        label.set_fontsize(8)
    first_reduction = 100 * (1 - rows[0].nkigym_us / rows[0].nakb_us)
    last_reduction = 100 * (1 - rows[-1].nkigym_us / rows[-1].nakb_us)
    axis.set_title(
        f"Configurations {first_rank}–{first_rank + len(rows) - 1}"
        f"   ·   Latency reduction vs. NAKB {first_reduction:+.1f}% to {last_reduction:+.1f}%",
        loc="left",
        fontsize=12,
        weight="bold",
        pad=12,
    )


def main() -> None:
    """Export the comparison to vector PDF with source metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=RESULTS_DIRECTORY / "measurements.json")
    parser.add_argument("--output-prefix", type=Path, default=REPOSITORY / "kernel_library/nakb_latency_comparison")
    args = parser.parse_args()
    source = cast(Path, args.measurements).expanduser().resolve()
    output = cast(Path, args.output_prefix).expanduser().resolve()
    if source in {output.with_suffix(suffix) for suffix in (".pdf", ".json")}:
        raise ValueError("The output prefix would overwrite the input measurements; choose a different prefix")
    rows, snapshot, source_sha256 = load_measurements(source)
    status = measurement_status(snapshot)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    panel_count = math.ceil(len(rows) / 32)
    figure = Figure(figsize=(22, 4.9 + 3.6 * panel_count), facecolor="white")
    FigureCanvasPdf(figure)
    figure.text(0.060, 0.977, "NAKB vs. NKIGym kernel latency", fontsize=27, weight="bold", color=INK)
    figure.text(
        0.060,
        0.951,
        f"{len(rows)} configurations · {status} · descending latency reduction vs. NAKB, "
        "left to right and then top to bottom",
        fontsize=13,
        color=MUTED,
    )
    figure.legend(
        handles=[Patch(facecolor=color, label=name) for color, name in zip(COLORS, ("NAKB", "NKIGym"))],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.988),
        ncol=2,
        frameon=False,
        fontsize=13,
        handlelength=1.4,
    )
    draw_aggregate(figure, rows)
    latencies = [value for row in rows for value in (row.nakb_us, row.nkigym_us)]
    exponents = (math.floor(math.log10(min(latencies))) - 1, math.ceil(math.log10(max(latencies))))
    for panel, start in enumerate(range(0, len(rows), 32)):
        draw_kernel_panel(figure, rows[start : start + 32], panel, start + 1, panel_count, exponents)
    timestamp = snapshot["updated_utc"].replace("T", " ").replace("+00:00", " UTC")
    figure.text(0.060, 0.013, f"Provenance: {status}.  Results saved {timestamp}.", fontsize=10, color=MUTED)
    figure.text(0.982, 0.013, "Bar labels: µs; k = 1,000 µs", ha="right", fontsize=10, color=MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    description = (
        f"{status}; run {snapshot['run_id']}; {source}; SHA-256 {source_sha256}. "
        f"Frozen NAKB baselines verified for all {len(rows)} configurations. "
        "Aggregate: arithmetic mean of per-configuration NKIGym / NAKB latency ratios. "
        f"{snapshot['description']}"
    )
    figure.savefig(
        output.with_suffix(".pdf"),
        dpi=600,
        metadata={"Title": "NAKB vs. NKIGym kernel latency", "Subject": description},
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "measurements": str(source),
                "measurements_sha256": source_sha256,
                "run_id": snapshot["run_id"],
                "matplotlib_version": mpl.__version__,
                "mean_relative_latency": snapshot["mean_relative_latency"],
                "diagnostic_only": snapshot["diagnostic_only"],
                "complete_single_run": snapshot["complete_single_run"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {output.with_suffix('.pdf')}")
    print(f"Configurations: {len(rows)}; mean relative latency: {snapshot['mean_relative_latency']}; {status}")


if __name__ == "__main__":
    main()
