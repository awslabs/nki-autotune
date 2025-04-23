import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from autotune.cache.directories import extract_mnk_from_dirname, get_cache_dir, get_save_path
from autotune.cache.results import PerformanceMetrics
from autotune.tune.metrics import calculate_GEMM_pe_utilization


def collect_pe_data_with_stats(directory: str):
    """
    Collect PE utilization data for all MNK combinations in a directory,
    calculating both best and mean performance metrics.

    Parameters:
    -----------
    directory : str
        Directory containing M-N-K subdirectories with perf_metrics.json files.

    Returns:
    --------
    dict
        Dictionary mapping (M, N) pairs to dictionaries of K values and their PE utilization data
        with 'best' and 'mean' statistics.
    """
    pe_utilization_data = defaultdict(lambda: defaultdict(dict))

    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return pe_utilization_data

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

                # Calculate best (minimum time) configuration
                best_config = min(all_results, key=lambda r: r.min_ms)
                best_pe_util = calculate_GEMM_pe_utilization((k, m), (k, n), best_config.min_ms, "trn1")

                # Calculate mean PE utilization across all configurations
                all_pe_utils = [calculate_GEMM_pe_utilization((k, m), (k, n), r.min_ms, "trn1") for r in all_results]
                mean_pe_util = np.mean(all_pe_utils) if all_pe_utils else 0

                # Store both metrics
                pe_utilization_data[(m, n)][k]["best"] = best_pe_util
                pe_utilization_data[(m, n)][k]["mean"] = mean_pe_util

    return pe_utilization_data


def plot_single_pe_vs_k_comparison(m, n, tuned_data, baseline_data, plots_dir):
    """
    Create a single plot comparing tuned vs baseline PE utilization vs K for a specific (M, N) pair,
    showing best and mean statistics for tuned data.

    Parameters:
    -----------
    m, n : int
        M and N dimensions.
    tuned_data : dict
        Nested dict mapping K values to 'best' and 'mean' PE utilization for tuned configs.
    baseline_data : dict
        Dictionary mapping K values to their respective PE utilization for baseline configs.
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

    # Add a horizontal line at 87.5% to represent Trn1 Max
    plt.axhline(y=0.875, color="purple", linestyle="--", linewidth=2, label="Trn1 Max")

    # Add text annotation for the line
    if all_k_values:
        plt.text(
            min(all_k_values),  # X position - start of plot
            0.885,  # Y position - slightly above the line
            "87.5%",  # Text label
            color="purple",
            fontweight="bold",
        )

    plt.title(f"PE Utilization vs. K for M={m}, N={n}")
    plt.xlabel("K")
    plt.ylabel("PE Utilization (%)")
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

    # GEMM_PE_utilization plots go directly in the plots directory
    save_dir, filename = get_save_path(plots_dir, "GEMM_PE_utilization", m, n)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=400)
    plt.close()


def plot_pe_vs_k_comparison(workload_name: str):
    """
    Create plots comparing PE utilization vs K for each unique (M, N) pair between tuned and baseline,
    showing best and mean statistics for tuned data.

    Parameters:
    -----------
    workload_name : str
        Name of the workload (used to locate directories and save plots)
    """
    # Construct directory paths
    tuned_dir = get_cache_dir(workload_name, "tuned")
    baseline_dir = get_cache_dir(workload_name, "baseline")
    plots_dir = get_cache_dir(workload_name, "plots")

    # Make sure output directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Collect data from tuned and baseline directories with statistics
    tuned_data = collect_pe_data_with_stats(tuned_dir)
    baseline_data = collect_pe_data_with_stats(baseline_dir)

    # Get all unique (M, N) pairs from both datasets
    all_mn_pairs = set(tuned_data.keys()).union(baseline_data.keys())

    # Create plots for each (M, N) pair
    for m, n in all_mn_pairs:
        tuned_k_data = tuned_data.get((m, n), {})
        baseline_k_data = baseline_data.get((m, n), {})

        # Only create plot if we have data for either tuned or baseline
        if tuned_k_data or baseline_k_data:
            plot_single_pe_vs_k_comparison(m, n, tuned_k_data, baseline_k_data, plots_dir)
