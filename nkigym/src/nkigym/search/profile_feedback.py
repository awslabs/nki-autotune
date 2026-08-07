"""Neuron Explorer metric formatting for agentic search observations."""

from __future__ import annotations

import math

from nkigym.profile.types import ProfileMetrics
from nkigym.search.types import Evaluation, EvaluationMetric, SearchNode

_BYTES_PER_KIB = 1024.0
_BYTES_PER_MIB = 1024.0 * 1024.0
_PERCENT_DIAGNOSTICS = (
    ("mfu_max_achievable_estimated_percent", "mfu_ceiling_percent"),
    ("mbu_estimated_percent", "mbu_percent"),
    ("total_active_time_percent", "total_active_percent"),
    ("tensor_engine_active_time_percent", "tensor_engine_active_percent"),
    ("vector_engine_active_time_percent", "vector_engine_active_percent"),
    ("scalar_engine_active_time_percent", "scalar_engine_active_percent"),
    ("gpsimd_engine_active_time_percent", "gpsimd_engine_active_percent"),
    ("dma_active_time_percent", "dma_active_percent"),
    ("sync_engine_active_time_percent", "sync_engine_active_percent"),
)
_VALUE_DIAGNOSTICS = (
    ("mm_arithmetic_intensity", "matmul_arithmetic_intensity_flops_per_byte"),
    ("peak_flops_bandwidth_ratio", "peak_flops_bandwidth_ratio"),
)
_MIB_DIAGNOSTICS = (
    ("hbm_read_bytes", "hbm_read_mib"),
    ("hbm_write_bytes", "hbm_write_mib"),
    ("sbuf_read_bytes", "sbuf_read_mib"),
    ("sbuf_write_bytes", "sbuf_write_mib"),
    ("psum_read_bytes", "psum_read_mib"),
    ("psum_write_bytes", "psum_write_mib"),
    ("spill_save_bytes", "spill_save_mib"),
    ("spill_reload_bytes", "spill_reload_mib"),
)
_THROTTLE_PREFIX = "throttle_avg_util_limit_nc"
_THROTTLE_SUFFIX = "_percent"

_TRACE_METRICS = (
    "profile_succeeded",
    "mfu_percent",
    "latency_ms",
    "mbu_percent",
    "total_active_percent",
    "tensor_engine_active_percent",
    "vector_engine_active_percent",
    "dma_active_percent",
    "hbm_read_mib",
    "hbm_write_mib",
    "spill_save_mib",
    "spill_reload_mib",
    "average_throttle_limit_percent",
)


def evaluation_from_profile(profile: ProfileMetrics, node_id: int) -> Evaluation:
    """Select Neuron Explorer fields for one successful agent evaluation."""
    metrics: dict[str, EvaluationMetric] = {
        "profile_succeeded": True,
        "mfu_percent": profile.mfu_percent,
        "latency_ms": profile.latency_ms,
        **_diagnostic_metrics(profile.profiler_summary),
    }
    return Evaluation(
        score=profile.mfu_percent,
        metrics=metrics,
        message=(
            f"Neuron profile succeeded for N{node_id:03d}: "
            f"MFU={profile.mfu_percent:.2f}%, latency={profile.latency_ms:.4f} ms"
        ),
    )


def _diagnostic_metrics(summary: dict[str, object]) -> dict[str, float]:
    """Select utilization, engine, traffic, and roofline metrics for reasoning."""
    metrics: dict[str, float] = {}
    for source, destination in _PERCENT_DIAGNOSTICS:
        value = _summary_number(summary, source)
        if value is not None:
            metrics[destination] = value * 100.0
    for source, destination in _VALUE_DIAGNOSTICS:
        value = _summary_number(summary, source)
        if value is not None:
            metrics[destination] = value
    for source, destination in _MIB_DIAGNOSTICS:
        value = _summary_number(summary, source)
        if value is not None:
            metrics[destination] = value / _BYTES_PER_MIB
    dma_transfer_average_bytes = _summary_number(summary, "dma_transfer_average_bytes")
    if dma_transfer_average_bytes is not None:
        metrics["dma_transfer_average_kib"] = dma_transfer_average_bytes / _BYTES_PER_KIB
    throttle_limits = [
        value
        for name in sorted(summary)
        if name.startswith(_THROTTLE_PREFIX) and name.endswith(_THROTTLE_SUFFIX)
        for value in [_summary_number(summary, name)]
        if value is not None
    ]
    if throttle_limits:
        metrics["average_throttle_limit_percent"] = sum(throttle_limits) * 100.0 / len(throttle_limits)
    return metrics


def _summary_number(summary: dict[str, object], name: str) -> float | None:
    """Read one finite non-negative summary value, allowing negative sentinels."""
    raw_value = summary.get(name)
    value: float | None = None
    if raw_value is not None:
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise RuntimeError(f"profiler summary metric {name} must be numeric")
        numeric_value = float(raw_value)
        if not math.isfinite(numeric_value):
            raise RuntimeError(f"profiler summary metric {name} must be finite")
        if numeric_value >= 0:
            value = numeric_value
    return value


def format_profile_metrics(node: SearchNode) -> list[str]:
    """Format every agent-selected profiler metric with explicit units."""
    lines = [f"- {name}: {format_metric_value(name, value)}" for name, value in node.evaluation.metrics.items()]
    if not lines:
        lines.append("- no structured metrics")
    return lines


def format_metric_value(name: str, value: EvaluationMetric) -> str:
    """Format one metric with the unit encoded by its stable name."""
    result = str(value)
    if isinstance(value, bool):
        result = str(value).lower()
    elif isinstance(value, (int, float)):
        numeric = float(value)
        if name.endswith("_percent"):
            result = f"{numeric:.2f}%"
        elif name.endswith("_ms"):
            result = f"{numeric:.6f} ms"
        elif name.endswith("_mib"):
            result = f"{numeric:.2f} MiB"
        elif name.endswith("_kib"):
            result = f"{numeric:.2f} KiB"
        elif name.endswith("_flops_per_byte") or name == "peak_flops_bandwidth_ratio":
            result = f"{numeric:.2f} FLOP/byte"
        elif isinstance(value, int):
            result = str(value)
        else:
            result = f"{numeric:.6g}"
    return result


def format_trace_metrics(node: SearchNode) -> str:
    """Format a bounded diagnostic subset for one complete-trace line."""
    metrics = node.evaluation.metrics
    parts = [f"{name}={format_metric_value(name, metrics[name])}" for name in _TRACE_METRICS if name in metrics]
    return ", ".join(parts) or "none"


__all__ = ["evaluation_from_profile", "format_metric_value", "format_profile_metrics", "format_trace_metrics"]
