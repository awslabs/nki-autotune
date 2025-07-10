import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from autotune.cache.results import get_best_result


def numerical_key(dim_string: str) -> list:
    """
    Convert a dimension string to a list of integers for proper numerical sorting.
    For strings like '1x2048x1024_1024x2048', converts to [1, 2048, 1024, 1024, 2048]
    """
    # Replace '_' with 'x' to treat the entire string uniformly
    unified_string = dim_string.replace("_", "x")

    # Split by 'x' and convert each component to integer if possible
    components = []
    for part in unified_string.split("x"):
        try:
            components.append(int(part))
        except ValueError:
            # If not a number, add a string (this is just a fallback)
            components.append(part)

    return components


def collect_metrics_data_with_stats(directory: str, metric_name: str):
    """
    Collect performance metrics for all MNK combinations, calculating both best and mean values.
    Skips directories where the requested metric doesn't exist.

    Parameters:
    -----------
    directory : str
        Directory containing M-N-K subdirectories with perf_metrics.json files.
    metric_name : str
        Metric to collect (e.g., "pe_util" or "hfu_estimated_percent")

    Returns:
    --------
    dict
        Dictionary mapping dimension strings to dictionaries with 'best' and 'mean' statistics.
    """
    # Initialize metrics data
    metrics_data = {}

    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return metrics_data

    # Scan the directories to find all MNK combinations
    for dirname in os.listdir(directory):
        json_path = os.path.join(directory, dirname, "perf_metrics.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)

            # Get all results without errors
            valid_results = [r for r in data.get("results", []) if "error" not in r]

            if not valid_results:
                continue

            # Filter results that contain the metric_name
            valid_metric_results = [r for r in valid_results if metric_name in r]

            if not valid_metric_results:
                # Skip this directory if no results have the metric
                continue

            try:
                best_config = get_best_result(data)
                # Skip if the requested metric isn't in the best config
                if metric_name not in best_config:
                    continue

                best_metric = best_config[metric_name]

                # Calculate mean metric using only results that have this metric
                all_metrics = [r[metric_name] for r in valid_metric_results]
                mean_metric = np.mean(all_metrics) if all_metrics else None

                # Store metrics only if we successfully calculated them
                metrics_data[dirname] = {"best": best_metric, "mean": mean_metric}
            except:
                # If there's any issue getting the best result or metric, skip this directory
                continue

    return metrics_data


def plot_metric(cache_root_dir: str, metric_name: str, kernel_names: List[str]):
    """
    Create a single line plot showing the specified metric for all (M,N,K) combinations,
    comparing different kernel implementations. Plots the best values with error bars
    extending to the mean values, regardless of which is higher or lower.

    Parameters:
    -----------
    cache_root_dir : str
        Root directory for the cache data
    metric_name : str
        Name of the metric to plot
    kernel_names : List[str]
        List of kernel names to include in the plot
    """
    plots_dir = f"{cache_root_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    all_kernels_metrics = {}
    for kernel_name in kernel_names:
        metrics = collect_metrics_data_with_stats(f"{cache_root_dir}/{kernel_name}", metric_name)
        all_kernels_metrics[kernel_name] = metrics

    all_inputs_strings = set()
    for kernel_metrics in all_kernels_metrics.values():
        all_inputs_strings.update(kernel_metrics.keys())

    all_inputs_strings_sorted = sorted(list(all_inputs_strings), key=lambda x: numerical_key(x))

    # Create the plot
    plt.figure(figsize=(16, 8))

    # Generate x-axis positions
    x_positions = {dim: idx for idx, dim in enumerate(all_inputs_strings_sorted)}

    # Plot each kernel's data
    colors = ["blue", "red", "green", "purple", "orange", "cyan", "magenta"]
    markers = ["o", "s", "d", "^", "X", "P"]

    for i, (kernel_name, metrics) in enumerate(all_kernels_metrics.items()):
        # Choose color and marker
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Collect data points
        x_values = []
        best_values = []
        yerr_lower = []  # For error bars extending downward
        yerr_upper = []  # For error bars extending upward

        for dim_string in all_inputs_strings_sorted:
            if dim_string in metrics:
                if metrics[dim_string]["best"] is not None and metrics[dim_string]["mean"] is not None:
                    x_values.append(x_positions[dim_string])
                    best_val = metrics[dim_string]["best"]
                    mean_val = metrics[dim_string]["mean"]
                    best_values.append(best_val)

                    # Calculate error bars based on which value is higher
                    if mean_val <= best_val:
                        # Mean is lower than or equal to best, error bar goes down
                        yerr_lower.append(best_val - mean_val)
                        yerr_upper.append(0)  # No upward error
                    else:
                        # Mean is higher than best, error bar goes up
                        yerr_lower.append(0)  # No downward error
                        yerr_upper.append(mean_val - best_val)

        # Only plot if we have data points
        if x_values:
            plt.errorbar(
                x_values,
                best_values,
                yerr=[yerr_lower, yerr_upper],  # Asymmetric error bars
                fmt=marker + "-",  # Combine marker with line
                color=color,
                ecolor=color,
                capsize=5,
                linewidth=2,
                markersize=8,
                label=kernel_name,
            )

    # Set plot properties
    plt.xlabel("Input Shapes")
    plt.ylabel(f"{metric_name.replace('_', ' ')}")
    plt.title(f"{metric_name.replace('_', ' ')} (Best values with error bars to Mean)")
    plt.xticks(range(len(all_inputs_strings_sorted)), all_inputs_strings_sorted, rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    kernels_str = "_vs_".join(kernel_names)
    save_path = os.path.join(plots_dir, f"{kernels_str}_{metric_name}_best_with_error_bars.png")
    plt.savefig(save_path, dpi=400)
    plt.close()
