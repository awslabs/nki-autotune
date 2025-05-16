import os

import matplotlib.pyplot as plt
import numpy as np

from autotune.cache.directories import extract_mnk_from_dirname, get_cache_dir, get_save_path
from autotune.cache.results import PerformanceMetrics, PerformanceResult


def collect_metrics_data_with_stats(directory: str, metric_names):
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
    for metric in metric_names:
        metrics_data[metric] = {}

    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return metrics_data

    # Scan the directories to find all MNK combinations
    for dirname in os.listdir(directory):
        m, n, k = extract_mnk_from_dirname(dirname)
        if m is not None and n is not None and k is not None:
            # Read the perf_metrics.json file
            json_path = os.path.join(directory, dirname, "perf_metrics.json")
            if os.path.exists(json_path):
                loaded_metrics = PerformanceMetrics.load(json_path)

                # Get all results
                all_results = loaded_metrics.results

                if not all_results:
                    continue

                # Find best configuration (minimum time)
                best_config = min(all_results, key=lambda r: r.min_ms)

                # Process each requested metric
                for metric in metric_names:
                    best_metric = parse_metric(best_config, metric)

                    # Calculate mean metric across all configurations
                    all_metrics = [parse_metric(r, metric) for r in all_results]
                    mean_metric = np.mean(all_metrics) if all_metrics else 0

                    # Initialize nested dictionaries if they don't exist
                    if (m, n) not in metrics_data[metric]:
                        metrics_data[metric][(m, n)] = {}

                    if k not in metrics_data[metric][(m, n)]:
                        metrics_data[metric][(m, n)][k] = {}

                    # Store both metrics
                    metrics_data[metric][(m, n)][k]["best"] = best_metric
                    metrics_data[metric][(m, n)][k]["mean"] = mean_metric

    return metrics_data


def parse_metric(result: PerformanceResult, metric_name: str):
    try:
        metric = result.metrics[metric_name]
    except:
        metric = 0
    return metric


def plot_single_metric_vs_k_comparison(m, n, metric_name, tuned_data, baseline_data, plots_dir):
    """
    Create a single plot comparing tuned vs baseline metrics vs K for a specific (M, N) pair,
    showing best and mean statistics for tuned data.

    Parameters:
    -----------
    m, n : int
        M and N dimensions.
    metric_name : str
        Name of the metric to plot ('pe_util' or 'hfu')
    tuned_data : dict
        Nested dict mapping K values to 'best' and 'mean' metric values for tuned configs.
    baseline_data : dict
        Dictionary mapping K values to their respective metric values for baseline configs.
    plots_dir : str
        Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))

    # Define colors and styles for the lines
    styles = [("blue", "o-", "Tuned Best"), ("green", "s-", "Tuned Mean"), ("red", "d--", "Baseline Best")]

    # Get all K values from both datasets for later use
    all_k_values = set()
    if tuned_data:
        all_k_values.update(tuned_data.keys())
    if baseline_data:
        all_k_values.update(baseline_data.keys())

    # Plot tuned data if available
    if tuned_data:
        # Get all K values
        k_values = sorted(tuned_data.keys())

        # Plot Best performance
        best_values = []
        best_k_values = []
        for k in k_values:
            if "best" in tuned_data[k]:
                best_values.append(tuned_data[k]["best"])
                best_k_values.append(k)

        if best_k_values:
            plt.plot(best_k_values, best_values, styles[0][1], linewidth=2, color=styles[0][0], label=styles[0][2])

        # Plot Mean performance
        mean_values = []
        mean_k_values = []
        for k in k_values:
            if "mean" in tuned_data[k]:
                mean_values.append(tuned_data[k]["mean"])
                mean_k_values.append(k)

        if mean_k_values:
            plt.plot(mean_k_values, mean_values, styles[1][1], linewidth=2, color=styles[1][0], label=styles[1][2])

    # Plot baseline data if available (best performance only)
    if baseline_data:
        k_values = sorted(baseline_data.keys())
        baseline_best_values = []
        baseline_k_values = []

        for k in k_values:
            if "best" in baseline_data[k]:
                baseline_best_values.append(baseline_data[k]["best"])
                baseline_k_values.append(k)

        if baseline_k_values:
            plt.plot(
                baseline_k_values,
                baseline_best_values,
                styles[2][1],
                linewidth=2,
                color=styles[2][0],
                label=styles[2][2],
            )

    # Set metric-specific properties
    # Add a horizontal line at 87.5% to represent Trn1 Max
    plt.axhline(y=0.875, color="purple", linestyle="--", linewidth=2, label="Trn1 Max")
    plt.text(
        min(all_k_values),  # X position - start of plot
        0.885,  # Y position - slightly above the line
        "87.5%",  # Text label
        color="purple",
        fontweight="bold",
    )
    if metric_name == "mfu_estimated_percent":
        plot_title = f"Model Flops Utilization vs. K for M={m}, N={n}"
        y_label = "MFU (%)"
        plot_type = "Model_Flops_Utilization"
    elif metric_name == "hfu_estimated_percent":
        plot_title = f"Hardware Flops Utilization vs. K for M={m}, N={n}"
        y_label = "HFU (%)"
        plot_type = "Hardware_Flops_Utilization"
    else:
        plot_title = f"{metric_name} vs. K for M={m}, N={n}"
        y_label = f"{metric_name} (%)"
        plot_type = f"{metric_name}_vs_k"

    plt.title(plot_title)
    plt.xlabel("K")
    plt.ylabel(y_label)
    plt.grid(True)

    # Use the already collected all_k_values for x-ticks
    if all_k_values:
        min_k, max_k = min(all_k_values), max(all_k_values)

        # Generate powers of 2 that lie within our range
        powers = []
        power = 1
        while power <= max_k:
            if power >= min_k:
                powers.append(power)
            power *= 2

        # If there are power of 2 values in our range, use them as ticks
        if powers:
            plt.xticks(powers, [str(x) for x in powers])

    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))

    # Add legend
    plt.legend()

    plt.tight_layout()

    # Save plot
    save_dir, filename = get_save_path(plots_dir, plot_type, m, n)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=400)
    plt.close()


def plot_metrics_vs_k_comparison(workload_name: str):
    """
    Create plots comparing selected metrics vs K for each unique (M, N) pair between
    tuned and baseline, showing best and mean statistics for tuned data.

    Parameters:
    -----------
    workload_name : str
        Name of the workload (used to locate directories and save plots)
    metric_names : tuple
        Metrics to plot, options include "pe_util" and "hfu_estimated_percent"
    """
    # Construct directory paths
    tuned_dir = get_cache_dir(workload_name, "tuned")
    baseline_dir = get_cache_dir(workload_name, "baseline")
    plots_dir = get_cache_dir(workload_name, "plots")

    # Make sure output directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Hardcode the metrics instead of using default args
    metrics_to_collect = ["mfu_estimated_percent", "hfu_estimated_percent"]

    # Collect data from tuned and baseline directories with statistics
    tuned_metrics = collect_metrics_data_with_stats(tuned_dir, metrics_to_collect)
    baseline_metrics = collect_metrics_data_with_stats(baseline_dir, metrics_to_collect)

    # Process each metric
    for metric in metrics_to_collect:
        tuned_data = tuned_metrics.get(metric, {})
        baseline_data = baseline_metrics.get(metric, {})

        # Get all unique (M, N) pairs from both datasets for this metric
        all_mn_pairs = set(tuned_data.keys()).union(baseline_data.keys())

        # Create plots for each (M, N) pair
        for m, n in all_mn_pairs:
            tuned_k_data = tuned_data.get((m, n), {})
            baseline_k_data = baseline_data.get((m, n), {})

            # Only create plot if we have data for either tuned or baseline
            if tuned_k_data or baseline_k_data:
                plot_single_metric_vs_k_comparison(m, n, metric, tuned_k_data, baseline_k_data, plots_dir)
