import os

import matplotlib.pyplot as plt
import numpy as np

from autotune.cache.directories import extract_mnk_from_dirname, get_cache_dir
from autotune.cache.results import PerformanceMetrics


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

                # Find the best metric
                best_config = loaded_metrics.get_best_result()
                best_metric = getattr(best_config, metric_name)

                # Calculate mean metric across all configurations
                all_metrics = [getattr(r, metric_name) for r in all_results if "error" not in r.attributes]
                mean_metric = np.mean(all_metrics) if all_metrics else 0

                # Initialize nested dictionaries if they don't exist
                if (m, n) not in metrics_data:
                    metrics_data[(m, n)] = {}

                if k not in metrics_data[(m, n)]:
                    metrics_data[(m, n)][k] = {}

                # Store both metrics
                metrics_data[(m, n)][k]["best"] = best_metric
                metrics_data[(m, n)][k]["mean"] = mean_metric

    return metrics_data


def plot_metric(workload_name: str, metric_name: str):
    """
    Create a single line plot showing MFU for all (M,N,K) combinations,
    comparing tuned vs baseline configurations.

    Parameters:
    -----------
    workload_name : str
        Name of the workload (used to locate directories and save plots)
    """
    # Same setup as the bar chart version
    tuned_dir = get_cache_dir(workload_name, "tuned")
    baseline_dir = get_cache_dir(workload_name, "baseline")
    plots_dir = get_cache_dir(workload_name, "plots")

    os.makedirs(plots_dir, exist_ok=True)

    tuned_metrics = collect_metrics_data_with_stats(tuned_dir, metric_name)
    baseline_metrics = collect_metrics_data_with_stats(baseline_dir, metric_name)

    # Collect all MNK combinations
    mnk_labels = []
    mnk_tuples = []
    all_mn_pairs = set(tuned_metrics.keys()).union(baseline_metrics.keys())

    for m, n in all_mn_pairs:
        tuned_k_data = tuned_metrics.get((m, n), {})
        baseline_k_data = baseline_metrics.get((m, n), {})
        all_k_values = set(tuned_k_data.keys()).union(baseline_k_data.keys())

        for k in all_k_values:
            mnk_tuples.append((m, n, k))

    # Sort MNK tuples
    mnk_tuples.sort()

    # Extract data for plotting
    tuned_best_values = []
    tuned_mean_values = []
    baseline_best_values = []

    for m, n, k in mnk_tuples:
        mnk_labels.append(f"({m},{n},{k})")

        # Get tuned data
        if (m, n) in tuned_metrics and k in tuned_metrics[(m, n)]:
            tuned_best_values.append(tuned_metrics[(m, n)][k].get("best", 0))
            tuned_mean_values.append(tuned_metrics[(m, n)][k].get("mean", 0))
        else:
            tuned_best_values.append(0)
            tuned_mean_values.append(0)

        # Get baseline data
        if (m, n) in baseline_metrics and k in baseline_metrics[(m, n)]:
            baseline_best_values.append(baseline_metrics[(m, n)][k].get("best", 0))
        else:
            baseline_best_values.append(0)

    # Create the line plot
    plt.figure(figsize=(16, 8))

    x = np.arange(len(mnk_labels))

    plt.plot(x, tuned_best_values, "o-", color="blue", linewidth=2, label="Tuned Best")
    plt.plot(x, tuned_mean_values, "s-", color="green", linewidth=2, label="Tuned Mean")
    plt.plot(x, baseline_best_values, "d--", color="red", linewidth=2, label="Baseline Best")

    # Set plot properties
    plt.xlabel("(M,N,K) Configurations")
    plt.ylabel(f"{metric_name}")
    plt.title(f"{workload_name} {metric_name}")
    plt.xticks(x, mnk_labels, rotation=90)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save plot
    save_path = os.path.join(plots_dir, f"{workload_name}_{metric_name}.png")
    plt.savefig(save_path, dpi=400)
    plt.close()
