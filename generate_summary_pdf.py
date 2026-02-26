"""Generate a PDF summarizing nkigym search backend performance optimizations.

All profiling data hardcoded from nkigym/dev_notes.md (2026-02-26).
Dependencies: matplotlib, fpdf2, Liberation fonts.

Usage:
    source ~/venvs/kernel-env/bin/activate
    python generate_summary_pdf.py
"""

import tempfile
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from fpdf import FPDF

FONT_DIR = "/usr/share/fonts/truetype/liberation"

C_RED = "#C62828"
C_RED_LT = "#FFCDD2"
C_ORANGE = "#EF6C00"
C_ORANGE_LT = "#FFF3E0"
C_AMBER = "#FF8F00"
C_AMBER_LT = "#FFF8E1"
C_YELLOW = "#FDD835"
C_GREEN = "#2E7D32"
C_GREEN_LT = "#E8F5E9"
C_TEAL = "#00897B"
C_BLUE = "#1565C0"
C_BLUE_LT = "#E3F2FD"
C_GREY = "#757575"
C_GREY_LT = "#EEEEEE"
C_INDIGO = "#5C6BC0"
C_INDIGO_LT = "#E8EAF6"
C_DARK = "#212121"
C_WHITE = "#FFFFFF"


def _fig_before_breakdown(path: str) -> None:
    """Horizontal stacked bar: baseline 274s breakdown."""
    labels = [
        "copy.deepcopy\n180s (66%)",
        "ast.dump/format\n64s (23%)",
        "exec() verify\n13s (5%)",
        "assert_allclose\n7s (3%)",
        "Hardware\n10s (4%)",
    ]
    widths = [180, 64, 13, 7, 10]
    colors = [C_RED, C_ORANGE, C_AMBER, C_YELLOW, C_BLUE_LT]

    fig, ax = plt.subplots(figsize=(10, 2.4))
    left = 0.0
    for w, c, lbl in zip(widths, colors, labels):
        ax.barh(0, w, left=left, height=0.6, color=c, edgecolor=C_WHITE)
        fontsize = 7.5 if w < 15 else 9
        ax.text(left + w / 2, 0, lbl, ha="center", va="center", fontsize=fontsize, color=C_DARK)
        left += w

    divider_x = sum(widths[:4])
    ax.axvline(divider_x, color=C_DARK, linewidth=1.5, linestyle="--")
    ax.text(divider_x - 1, 0.48, "Controllable: 264s (96%)", ha="right", va="bottom", fontsize=8.5)
    ax.text(divider_x + 1, 0.48, "HW: 10s (4%)", ha="left", va="bottom", fontsize=8.5, color=C_BLUE)

    ax.set_xlim(0, 280)
    ax.set_ylim(-0.5, 0.85)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


_WATERFALL_STEPS = [
    ("Baseline", 274, None),
    ("#1 deepcopy\n\u2212178s", -178, C_GREEN),
    ("#2 ast cache\n\u221236s", -36, C_GREEN),
    ("#3 interpreter\n\u221211.5s", -11.5, C_TEAL),
    ("#4 np compare\n\u22126.9s", -6.9, C_TEAL),
    ("#5 early-exit\n\u22122s", -2, C_TEAL),
    ("+ HW delta\n+47.4s", 47.4, C_GREY),
    ("Final", None, None),
]


def _draw_waterfall_bars(ax: plt.Axes, cumulative: list[float]) -> None:
    """Draw individual bars for the waterfall chart."""
    final = cumulative[-1]
    n = len(_WATERFALL_STEPS)
    for i, (_label, delta, color) in enumerate(_WATERFALL_STEPS):
        if i == 0:
            ax.bar(i, cumulative[0], color=C_RED, edgecolor=C_WHITE, width=0.6)
            ax.text(i, cumulative[0] + 3, f"{cumulative[0]:.0f}s", ha="center", fontsize=9, fontweight="bold")
        elif i == n - 1:
            ax.bar(i, final, color=C_BLUE, edgecolor=C_WHITE, width=0.6)
            ax.text(i, final + 3, f"{final:.1f}s", ha="center", fontsize=9, fontweight="bold")
        else:
            bottom = min(cumulative[i - 1], cumulative[i])
            ax.bar(i, abs(delta), bottom=bottom, color=color, edgecolor=C_WHITE, width=0.6)
            ax.plot([i - 0.7, i - 0.3], [cumulative[i - 1]] * 2, color=C_DARK, linewidth=0.8, linestyle=":")


def _fig_waterfall(path: str) -> None:
    """Waterfall chart: cumulative reduction from each optimization."""
    cumulative = [274.0]
    for _, delta, _ in _WATERFALL_STEPS[1:-1]:
        cumulative.append(cumulative[-1] + delta)
    cumulative.append(cumulative[-1])
    final = cumulative[-1]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    _draw_waterfall_bars(ax, cumulative)

    ax.annotate(
        "3.2\u00d7 speedup",
        xy=(len(_WATERFALL_STEPS) - 1, final),
        xytext=(len(_WATERFALL_STEPS) - 1.8, final + 55),
        fontsize=12,
        fontweight="bold",
        color=C_BLUE,
        arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.5),
    )

    ax.set_xticks(np.arange(len(_WATERFALL_STEPS)))
    ax.set_xticklabels([s[0] for s in _WATERFALL_STEPS], fontsize=8)
    ax.set_ylabel("End-to-end time (seconds)", fontsize=10)
    ax.set_ylim(0, 310)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


_BEFORE_FLOW = [
    ("tile_program() \u2192 GymProgram", "", C_INDIGO_LT, C_INDIGO),
    ("TransformGraph.expand_one()\n1,601 unique programs", "", C_INDIGO_LT, C_INDIGO),
    ("Verify: program_to_source()\n\u2192 exec() \u2192 assert_allclose", "20s", C_ORANGE_LT, C_ORANGE),
    ("Lower + roll_loops()\ncopy.deepcopy \u00d7117M\nast.dump \u00d7371K", "244s", C_RED_LT, C_RED),
    ("CompilationPool\n(hidden behind slow search)", "\u2014", C_GREY_LT, C_GREY),
    ("run_on_hardware()", "10s", C_BLUE_LT, C_BLUE),
]

_AFTER_FLOW = [
    ("tile_program() \u2192 GymProgram", "", C_INDIGO_LT, C_INDIGO),
    ("TransformGraph.expand_one()\n1,601 unique programs", "", C_INDIGO_LT, C_INDIGO),
    ("Verify: interpret_program()\nnp.max(np.abs(a - b)) > tol", "1.6s", C_GREEN_LT, C_GREEN),
    ("Lower + roll_loops()\nregex + dump cache", "7.2s", C_GREEN_LT, C_GREEN),
    ("CompilationPool.wait_all()\nneuronxcc \u00d7100 kernels", "47s", C_AMBER_LT, C_AMBER),
    ("run_on_hardware()", "19.5s", C_BLUE_LT, C_BLUE),
]

_FLOW_ARROWS = ["", "\u00d71,601", "\u00d7100", "", ""]


def _draw_flow_box(ax: plt.Axes, x: float, y: float, w: float, h: float, label: str, face: str, edge: str) -> None:
    """Draw a single rounded box with centered label."""
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08", facecolor=face, edgecolor=edge, linewidth=1.5
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9, linespacing=1.3)


def _draw_flow_column(ax: plt.Axes, stages: list, title: str, arrows: list) -> None:
    """Draw a vertical pipeline of boxes connected by arrows."""
    n = len(stages)
    box_w, box_h = 3.5, 1.15
    stride = box_h + 0.5
    x_left = 0.3
    x_mid = x_left + box_w / 2

    ax.set_xlim(-0.2, 5.6)
    ax.set_ylim(n * stride + 0.1, -0.6)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")

    for i, (label, time_str, face, edge) in enumerate(stages):
        y = i * stride
        _draw_flow_box(ax, x_left, y, box_w, box_h, label, face, edge)
        if time_str:
            ax.text(
                x_left + box_w + 0.2,
                y + box_h / 2,
                time_str,
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=edge,
            )
        if i < n - 1:
            y_tail = y + box_h + 0.02
            y_tip = (i + 1) * stride - 0.02
            ax.annotate(
                "", xy=(x_mid, y_tip), xytext=(x_mid, y_tail), arrowprops=dict(arrowstyle="->", color="#424242", lw=1.2)
            )
            if arrows[i]:
                ax.text(
                    x_mid + 0.15,
                    (y_tail + y_tip) / 2,
                    arrows[i],
                    fontsize=8,
                    color="#616161",
                    va="center",
                    style="italic",
                )


def _fig_flow_diagrams(path: str) -> None:
    """Before/after pipeline flow diagrams side by side."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 7.5))
    _draw_flow_column(ax_l, _BEFORE_FLOW, "Before (274s)", _FLOW_ARROWS)
    _draw_flow_column(ax_r, _AFTER_FLOW, "After (87s)", _FLOW_ARROWS)
    fig.text(
        0.5,
        0.01,
        "3.2\u00d7 end-to-end speedup  |  Controllable overhead: 264s \u2192 16s (94% reduction)",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=C_BLUE,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


class _PDF(FPDF):
    """PDF with page numbering footer."""

    def footer(self) -> None:
        """Render page footer."""
        self.set_y(-12)
        self.set_font("Sans", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()} of 3", align="C")


def _build_pdf(fig_breakdown: str, fig_waterfall: str, fig_flow: str) -> None:
    """Compose 3-page PDF from text and figure PNGs."""
    pdf = _PDF(orientation="P", unit="mm", format="A4")
    pdf.add_font("Sans", "", f"{FONT_DIR}/LiberationSans-Regular.ttf")
    pdf.add_font("Sans", "B", f"{FONT_DIR}/LiberationSans-Bold.ttf")
    pdf.add_font("Sans", "I", f"{FONT_DIR}/LiberationSans-Italic.ttf")
    pdf.add_font("Mono", "", f"{FONT_DIR}/LiberationMono-Regular.ttf")
    pdf.set_auto_page_break(auto=True, margin=15)

    _page1(pdf, fig_breakdown)
    _page2(pdf, fig_waterfall)
    _page3(pdf, fig_flow)
    pdf.output("summary.pdf")


def _page1(pdf: FPDF, fig1: str) -> None:
    """Page 1: title and before profile."""
    pdf.add_page()
    pdf.ln(8)
    pdf.set_font("Sans", "B", 20)
    pdf.cell(0, 11, "NKI Gym Search Backend", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Sans", "B", 16)
    pdf.cell(0, 9, "Performance Optimization Summary", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(2)
    pdf.set_font("Sans", "I", 10)
    pdf.cell(
        0,
        7,
        "256\u00b3 matmul  \u00b7  2\u00d72\u00d72 tiling  \u00b7  100 target variants  \u00b7  February 2026",
        new_x="LMARGIN",
        new_y="NEXT",
        align="C",
    )
    pdf.ln(10)

    _section_heading(pdf, "Performance Profile: Before Optimization (274s)")
    pdf.set_font("Sans", "", 10)
    pdf.multi_cell(
        0,
        5,
        "The baseline search pipeline took 274 seconds end-to-end. Profiling revealed"
        " that 92% of runtime (264s) was controllable Python overhead, dominated by"
        " copy.deepcopy calls in loop rolling (180s, 117M invocations) and repeated"
        " ast.dump/format calls (64s, 371K dumps). Hardware execution and compilation"
        " accounted for only 10 seconds.",
    )
    pdf.ln(4)
    pdf.image(fig1, x=10, w=190)
    pdf.ln(5)
    _callout_box(
        pdf, "92% of runtime was controllable Python overhead." "  Only 10s (4%) was irreducible hardware time."
    )


def _page2(pdf: FPDF, fig2: str) -> None:
    """Page 2: optimizations table and waterfall chart."""
    pdf.add_page()
    _section_heading(pdf, "Five Optimizations Applied")
    pdf.ln(1)
    _optimizations_table(pdf)
    pdf.ln(6)
    _section_heading(pdf, "Cumulative Impact")
    pdf.ln(1)
    pdf.image(fig2, x=10, w=190)


def _page3(pdf: FPDF, fig3: str) -> None:
    """Page 3: before/after flow diagrams and summary."""
    pdf.add_page()
    _section_heading(pdf, "Pipeline Architecture: Before vs After")
    pdf.set_font("Sans", "", 10)
    pdf.multi_cell(
        0,
        5,
        "After optimization, the bottleneck shifted from Python overhead to neuronxcc"
        " compilation wait (47s, 54%). The 47s was previously hidden\u2014compilations"
        " ran in parallel while the slow search executed and finished first."
        " Now that search completes in 19s, wait_all() blocks for the remaining time.",
    )
    pdf.ln(3)
    pdf.image(fig3, x=10, w=190)
    pdf.ln(3)
    _callout_box(
        pdf,
        "Controllable overhead: 264s \u2192 16s (94% reduction)"
        "  |  E2E: 274s \u2192 87s (3.2\u00d7)"
        "  |  Remaining bottleneck: neuronxcc compilation (47s)",
    )


def _section_heading(pdf: FPDF, text: str) -> None:
    """Render a section heading with blue accent line."""
    pdf.set_font("Sans", "B", 13)
    pdf.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
    y = pdf.get_y()
    pdf.set_draw_color(25, 118, 210)
    pdf.set_line_width(0.6)
    pdf.line(10, y, 200, y)
    pdf.set_line_width(0.2)
    pdf.ln(3)


def _callout_box(pdf: FPDF, text: str) -> None:
    """Render a callout box with left accent bar."""
    x, y = pdf.get_x(), pdf.get_y()
    pdf.set_fill_color(232, 245, 233)
    pdf.rect(x, y, 190, 14, style="F")
    pdf.set_fill_color(46, 125, 50)
    pdf.rect(x, y, 3, 14, style="F")
    pdf.set_font("Sans", "B", 10)
    pdf.set_xy(x + 7, y + 2)
    pdf.multi_cell(179, 5, text)
    pdf.ln(3)


_OPT_ROWS = [
    ("1", "Regex on ast.dump strings\ninstead of copy.deepcopy", "loop_rolling.py", "180s \u2192 2s\n(90\u00d7)"),
    ("2", "Per-statement\nast.dump cache", "loop_rolling.py", "39s \u2192 3s\n(13\u00d7)"),
    ("3", "Direct IR interpreter\nvia GymOp.simulate", "interpret.py\n(new)", "13s \u2192 1.5s\n(8.7\u00d7)"),
    ("4", "Direct numpy\ncomparison", "search.py", "7s \u2192 0.1s\n(70\u00d7)"),
    ("5", "Early-exit\nNamedTuple reuse", "data_reuse.py", "~2s saved"),
]


def _render_table_row(pdf: FPDF, row: tuple, widths: tuple, mono_col: int) -> None:
    """Render one multi-line table row."""
    line_count = max(cell.count("\n") + 1 for cell in row)
    row_h = 6 * line_count
    y_start = pdf.get_y()
    x_start = pdf.get_x()
    for i, (w, cell) in enumerate(zip(widths, row)):
        x = x_start + sum(widths[:i])
        pdf.rect(x, y_start, w, row_h)
        if i == mono_col:
            pdf.set_font("Mono", "", 7.5)
        else:
            pdf.set_font("Sans", "", 8.5)
        align = "C" if i in (0, 3) else "L"
        pdf.set_xy(x + 1, y_start + 1)
        pdf.multi_cell(w - 2, 5, cell, align=align)
    pdf.set_y(y_start + row_h)


def _optimizations_table(pdf: FPDF) -> None:
    """Render the 5-row optimizations table."""
    widths = (12, 78, 48, 52)
    headers = ("#", "Optimization", "File", "Speedup")

    pdf.set_font("Sans", "B", 9)
    pdf.set_fill_color(55, 71, 79)
    pdf.set_text_color(255, 255, 255)
    for w, h in zip(widths, headers):
        pdf.cell(w, 7, h, border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    for row in _OPT_ROWS:
        _render_table_row(pdf, row, widths, 2)


def main() -> None:
    """Generate summary.pdf."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = {
            "breakdown": f"{tmp}/fig_breakdown.png",
            "waterfall": f"{tmp}/fig_waterfall.png",
            "flow": f"{tmp}/fig_flow.png",
        }
        _fig_before_breakdown(paths["breakdown"])
        _fig_waterfall(paths["waterfall"])
        _fig_flow_diagrams(paths["flow"])
        _build_pdf(paths["breakdown"], paths["waterfall"], paths["flow"])

    out = Path("summary.pdf")
    print(f"Generated {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
