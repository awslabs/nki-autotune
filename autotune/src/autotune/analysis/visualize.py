# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os

import matplotlib.pyplot as plt
import numpy as np


def collect_metrics_data_with_stats(directory: str, metric_name: str) -> dict[str, dict[str, float]]:
    """Collect performance metrics for all workloads, calculating best and mean values.

    Scans subdirectories of the given directory for perf_metrics.json files.
    Each subdirectory represents a workload (input shapes combination).

    Args:
        directory: Directory containing workload subdirectories with perf_metrics.json files.
        metric_name: Metric to collect (e.g., 'min_ms', 'mfu_estimated_percent').

    Returns:
        Dictionary mapping workload labels to dicts with 'best' and 'mean' statistics.
    """
    metrics_data: dict[str, dict[str, float]] = {}

    if not os.path.exists(directory):
        print(f"Directory not found, skip plotting: {directory}")
        return metrics_data

    for dirname in os.listdir(directory):
        json_path = os.path.join(directory, dirname, "perf_metrics.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        main_metric = data["metadata"]["main_metric"]
        valid_results = [r for r in data.get("results", []) if "error" not in r]
        valid_results = [r for r in valid_results if metric_name in r]
        sorted_valid_results = sorted(valid_results, key=lambda result: result[main_metric])
        if not sorted_valid_results:
            continue

        sorted_metrics = [result[metric_name] for result in sorted_valid_results]
        best_metric = sorted_metrics[0]
        mean_metric = float(np.mean(sorted_metrics))
        metrics_data[dirname] = {"best": best_metric, "mean": mean_metric}

    return metrics_data


def plot_metric(cache_root_dir: str, metric_name: str, kernel_names: list[str]) -> None:
    """Create a line plot comparing a metric across kernel implementations.

    Plots the best values with error bars extending to the mean values for each
    workload across different kernels.

    Args:
        cache_root_dir: Root directory for the cache data.
        metric_name: Name of the metric to plot.
        kernel_names: List of kernel names to include in the plot.
    """
    plots_dir = f"{cache_root_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    all_kernels_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for kernel_name in kernel_names:
        metrics = collect_metrics_data_with_stats(f"{cache_root_dir}/{kernel_name}", metric_name)
        all_kernels_metrics[kernel_name] = metrics

    all_workload_labels: set[str] = set()
    for kernel_metrics in all_kernels_metrics.values():
        all_workload_labels.update(kernel_metrics.keys())

    all_workload_labels_sorted = sorted(all_workload_labels)

    plt.figure(figsize=(16, 8))

    x_positions = {label: idx for idx, label in enumerate(all_workload_labels_sorted)}

    colors = ["blue", "red", "green", "purple", "orange", "cyan", "magenta"]
    markers = ["o", "s", "d", "^", "X", "P"]

    for i, (kernel_name, metrics) in enumerate(all_kernels_metrics.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        x_values: list[int] = []
        best_values: list[float] = []
        yerr_lower: list[float] = []
        yerr_upper: list[float] = []

        for label in all_workload_labels_sorted:
            if label in metrics:
                if metrics[label]["best"] is not None and metrics[label]["mean"] is not None:
                    x_values.append(x_positions[label])
                    best_val = metrics[label]["best"]
                    mean_val = metrics[label]["mean"]
                    best_values.append(best_val)

                    if mean_val <= best_val:
                        yerr_lower.append(best_val - mean_val)
                        yerr_upper.append(0)
                    else:
                        yerr_lower.append(0)
                        yerr_upper.append(mean_val - best_val)

        if x_values:
            plt.errorbar(
                x_values,
                best_values,
                yerr=[yerr_lower, yerr_upper],
                fmt=marker + "-",
                color=color,
                ecolor=color,
                capsize=5,
                linewidth=2,
                markersize=8,
                label=kernel_name,
            )

    plt.xlabel("Input Shapes")
    plt.ylabel(f"{metric_name.replace('_', ' ')}")
    plt.title(f"{metric_name.replace('_', ' ')} (Best values with error bars to Mean)")
    plt.xticks(range(len(all_workload_labels_sorted)), all_workload_labels_sorted, rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    kernels_str = "_vs_".join(kernel_names)
    save_path = os.path.join(plots_dir, f"{kernels_str}_{metric_name}_best_with_error_bars.png")
    plt.savefig(save_path, dpi=400)
    plt.close()
