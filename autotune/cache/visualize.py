import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from autotune.cache.directories import CACHE_ROOT_DIR
from autotune.cache.results import ProfileResults


def collect_metrics_data_with_stats(directory: str, metric_name: str):
    """
    Collect multiple performance metrics (MFU and/or HFU) for all MNK combinations,
    calculating both best and mean performance metrics.

    Parameters:
    -----------
    directory : str
        Directory containing M-N-K subdirectories with perf_metrics.json files.
    metric_names : tuple
        Metrics to collect, options include "pe_util" and "hfu_estimated_percent"

    Returns:
    --------
    dict
        Dictionary mapping metric names to dictionaries of (M, N) pairs to dictionaries of K values
        and their respective metric data with 'best' and 'mean' statistics.
    """
    # Initialize metrics data with a cleaner structure
    metrics_data = {}

    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return metrics_data

    # Scan the directories to find all MNK combinations
    for dirname in os.listdir(directory):
        # Read the perf_metrics.json file
        json_path = os.path.join(directory, dirname, "perf_metrics.json")
        if os.path.exists(json_path):
            loaded_metrics = ProfileResults.load(json_path)

            # Get all results
            all_results = loaded_metrics.results

            if not all_results:
                continue

            # Find the best metric
            best_config = loaded_metrics.get_best_result()
            best_metric = getattr(best_config, metric_name)

            # Calculate mean metric across all configurations
            all_metrics = [getattr(r, metric_name) for r in all_results if "error" not in r.attributes]
            mean_metric = np.mean(all_metrics) if all_metrics else 0

            # Store both metrics
            metrics_data[dirname] = {"best": best_metric, "mean": mean_metric}

    return metrics_data


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


def plot_metric(metric_name: str, kernel_names: List[str]):
    """
    Create a single line plot showing MFU for all (M,N,K) combinations,
    comparing tuned vs baseline configurations.

    Parameters:
    -----------
    kernel_name : str
        Name of the workload (used to locate directories and save plots)
    """
    plots_dir = f"{CACHE_ROOT_DIR}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    all_kernels_metrics = {}
    for kernel_name in kernel_names:
        metrics = collect_metrics_data_with_stats(f"{CACHE_ROOT_DIR}/{kernel_name}", metric_name)
        all_kernels_metrics[kernel_name] = metrics

    all_inputs_strings = set()
    for kernel_metrics in all_kernels_metrics.values():
        all_inputs_strings.update(kernel_metrics.keys())

    all_inputs_strings_sorted = sorted(list(all_inputs_strings), key=lambda x: numerical_key(x))

    # Create the plot
    plt.figure(figsize=(16, 8))

    # Generate x-axis positions
    x = np.arange(len(all_inputs_strings_sorted))

    # Plot each kernel's data
    colors = ["blue", "red", "green", "purple", "orange", "cyan", "magenta"]
    markers = ["o", "s", "d", "^", "X", "P"]
    line_styles = ["-"]

    for i, (kernel_name, metrics) in enumerate(all_kernels_metrics.items()):
        # Extract best values for each dimension
        best_values = []
        mean_values = []
        for dim_string in all_inputs_strings_sorted:
            if dim_string in metrics:
                best_values.append(metrics[dim_string]["best"])
                mean_values.append(metrics[dim_string]["mean"])
            else:
                best_values.append(0)  # No data for this dimension
                mean_values.append(0)

        # Choose color, marker and line style
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        line_style = line_styles[i % len(line_styles)]

        # Plot best values
        plt.plot(
            x,
            best_values,
            marker=marker,
            linestyle=line_style,
            color=color,
            linewidth=2,
            markersize=8,
            label=f"{kernel_name}",
        )

    # Set plot properties
    plt.xlabel("Input Shapes")
    plt.ylabel(f"{metric_name.replace('_', ' ')}")
    plt.title(f"{metric_name.replace('_', ' ')}")
    plt.xticks(x, all_inputs_strings_sorted, rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    kernels_str = "_vs_".join(kernel_names)
    save_path = os.path.join(plots_dir, f"{kernels_str}_{metric_name}.png")
    plt.savefig(save_path, dpi=400)
    plt.close()
