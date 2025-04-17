# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
import re
from collections import defaultdict

import matplotlib.pyplot as plt

from autotune.cache.directories import VISUALIZATION_DIR
from autotune.cache.results import PerformanceMetrics
from autotune.tune.metrics import calculate_pe_utilization


def sort_lists_by_first(*lists):
    if not lists:
        return []

    # Create list of indices [0, 1, 2, ...]
    indices = list(range(len(lists[0])))

    # Sort indices based on the first list
    sorted_indices = sorted(indices, key=lambda i: lists[0][i])

    # Apply the same ordering to all lists
    return [[lst[i] for i in sorted_indices] for lst in lists]


def sort_by_key(data, sort_key):
    keys = list(data.keys())
    keys.remove(sort_key)
    keys.insert(0, sort_key)

    lists = [data[key] for key in keys]
    sorted_lists = sort_lists_by_first(*lists)
    return {key: sorted_list for key, sorted_list in zip(keys, sorted_lists)}


def plot_single_pe_vs_k(m, n, k_data, kernel_dir):
    """
    Create a single plot of PE utilization vs K for a specific (M, N) pair.

    Parameters:
    -----------
    m, n : int
        M and N dimensions.
    k_data : dict
        Dictionary mapping K values to their respective PE utilization.
    kernel_dir : str
        Directory to save the plot.
    """
    # Sort the K values
    k_values = sorted(k_data.keys())
    pe_util_values = [k_data[k] for k in k_values]

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, pe_util_values, "o-", linewidth=2)
    plt.title(f"PE Utilization vs. K for M={m}, N={n}")
    plt.xlabel("K")
    plt.ylabel("PE Utilization (%)")
    plt.grid(True)

    # Set x-ticks to powers of 2 within the range of k_values
    min_k, max_k = min(k_values), max(k_values)

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

    plt.tight_layout()
    plt.savefig(f"{kernel_dir}/pe_utilization_M{m}_N{n}.png", dpi=400)
    plt.close()


def plot_pe_vs_k(kernel_dir: str):
    """
    Create plots of PE utilization vs K for each unique (M, N) pair.

    Parameters:
    -----------
    kernel : object
        The kernel object containing func_name attribute.
    """
    # Dictionary to hold PE utilization data for each (M, N, K) combination
    pe_utilization_data = defaultdict(dict)

    # Check if the kernel directory exists
    assert os.path.exists(kernel_dir), f"Directory not found: {kernel_dir}"

    # Scan the directories to find all MNK combinations
    for dirname in os.listdir(kernel_dir):
        # Extract M, N, K values from directory name
        match = re.match(r"M(\d+)-N(\d+)-K(\d+)", dirname)
        if match:
            m, n, k = map(int, match.groups())

            # Read the perf_metrics.json file
            json_path = os.path.join(kernel_dir, dirname, "perf_metrics.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
                    # Find the configuration with the lowest latency
                    best_config = min(data["results"], key=lambda x: x["latency"])
                    # Store the PE utilization of the best configuration
                    pe_utilization_data[(m, n)][k] = best_config["pe_utilization"]

    # Create plots for each (M, N) pair
    for (m, n), k_data in pe_utilization_data.items():
        plot_single_pe_vs_k(m, n, k_data, kernel_dir)


def collect_pe_data_with_random_samples(directory, sample_sizes=[1, 10, "max"]):
    """
    Collect PE utilization data for all MNK combinations in a directory,
    with random sampling of different sizes and selecting the best from each sample.

    Parameters:
    -----------
    directory : str
        Directory containing M-N-K subdirectories with perf_metrics.json files.
    sample_sizes : list
        List of sample sizes to collect. Use 'max' to use all available configs.

    Returns:
    --------
    dict
        Dictionary mapping (M, N) pairs to dictionaries of K values and their PE utilization data
        for different sample sizes.
    """
    pe_utilization_data = defaultdict(lambda: defaultdict(dict))

    # Set random seed for reproducibility
    random.seed(42)

    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return pe_utilization_data

    # Scan the directories to find all MNK combinations
    for dirname in os.listdir(directory):
        # Extract M, N, K values from directory name
        match = re.match(r"M(\d+)-N(\d+)-K(\d+)", dirname)
        if match:
            m, n, k = map(int, match.groups())

            # Read the perf_metrics.json file
            json_path = os.path.join(directory, dirname, "perf_metrics.json")
            if os.path.exists(json_path):
                loaded_metrics = PerformanceMetrics.load(json_path)

                # Get all results
                all_results = loaded_metrics.results

                # For each sample size
                for size in sample_sizes:
                    # Determine actual sample size
                    if size == "max":
                        actual_size = len(all_results)
                    else:
                        actual_size = min(size, len(all_results))

                    if actual_size == 0:
                        continue

                    # Randomly sample configurations
                    if actual_size < len(all_results):
                        # Randomly sample without replacement
                        sampled_results = random.sample(all_results, actual_size)
                    else:
                        # Use all results if asking for max or more than available
                        sampled_results = all_results

                    # Find the best configuration in this sample
                    best_config = min(sampled_results, key=lambda r: r.min_ms)

                    # Calculate PE utilization for the best config
                    pe_util = calculate_pe_utilization((k, m), (k, n), best_config.min_ms, "trn1")

                    # Store by actual sample size
                    pe_utilization_data[(m, n)][k][actual_size] = pe_util

    return pe_utilization_data


def plot_single_pe_vs_k_comparison(m, n, tuned_data, baseline_data):
    """
    Create a single plot comparing tuned vs baseline PE utilization vs K for a specific (M, N) pair,
    showing multiple sampling strategies for tuned data.

    Parameters:
    -----------
    m, n : int
        M and N dimensions.
    tuned_data : dict
        Nested dict mapping K values to sample sizes to PE utilization for tuned configs.
    baseline_data : dict
        Dictionary mapping K values to their respective PE utilization for baseline configs.
    """
    plt.figure(figsize=(10, 6))

    # Define colors and styles for tuned data lines
    tuned_styles = [
        ("blue", "o-", "1-Sample Best"),
        ("green", "s-", "10-Sample Best"),
        ("purple", "d-", "All-Sample Best"),
    ]

    # Plot tuned data if available
    if tuned_data:
        # Get all K values
        k_values = sorted(tuned_data.keys())

        # Get all sample sizes from the data
        all_sizes = set()
        for k in k_values:
            all_sizes.update(tuned_data[k].keys())
        all_sizes = sorted(all_sizes)

        # Plot each available sample size
        for i, size in enumerate(all_sizes):
            if i < len(tuned_styles):
                color, style, label_base = tuned_styles[i]
                label = f"{size}-Sample Best"

                # Collect data for this sample size
                pe_util_values = []
                valid_k_values = []

                for k in k_values:
                    if size in tuned_data[k]:
                        pe_util_values.append(tuned_data[k][size])
                        valid_k_values.append(k)

                # Only plot if we have data
                if valid_k_values:
                    plt.plot(valid_k_values, pe_util_values, style, linewidth=2, color=color, label=label)

    # Plot baseline data if available
    if baseline_data:
        k_values = sorted(baseline_data.keys())
        pe_util_values = []
        valid_k_values = []

        for k in k_values:
            # Get the max sample size (best we have for baseline)
            max_size = max(baseline_data[k].keys()) if baseline_data[k] else 0
            if max_size > 0:
                pe_util_values.append(baseline_data[k][max_size])
                valid_k_values.append(k)

        if valid_k_values:
            # FIXED: removed redundant color definition to avoid warning
            plt.plot(valid_k_values, pe_util_values, "s--", linewidth=2, color="red", label="Baseline")

    plt.title(f"PE Utilization vs. K for M={m}, N={n}")
    plt.xlabel("K")
    plt.ylabel("PE Utilization (%)")
    plt.grid(True)

    # Get all K values from both datasets for setting x-ticks
    all_k_values = set()
    if tuned_data:
        all_k_values.update(tuned_data.keys())
    if baseline_data:
        all_k_values.update(baseline_data.keys())

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
    plt.savefig(f"{VISUALIZATION_DIR}/GEMM_PE_utilization_M{m}_N{n}.png", dpi=400)
    plt.close()


def plot_pe_vs_k_comparison(tuned_dir, baseline_dir):
    """
    Create plots comparing PE utilization vs K for each unique (M, N) pair between tuned and baseline,
    with multiple random sampling strategies for tuned data.

    Parameters:
    -----------
    tuned_dir : str
        Base cache directory for tuned results.
    baseline_dir : str
        Base directory for baseline results.
    """
    # Make sure output directory exists
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    # Collect data from tuned and baseline directories with random sampling
    tuned_data = collect_pe_data_with_random_samples(tuned_dir, sample_sizes=[1, 10, "max"])
    baseline_data = collect_pe_data_with_random_samples(baseline_dir, sample_sizes=["max"])

    # Get all unique (M, N) pairs from both datasets
    all_mn_pairs = set(tuned_data.keys()).union(baseline_data.keys())

    # Create plots for each (M, N) pair
    for m, n in all_mn_pairs:
        tuned_k_data = tuned_data.get((m, n), {})
        baseline_k_data = baseline_data.get((m, n), {})

        # Only create plot if we have data for either tuned or baseline
        if tuned_k_data or baseline_k_data:
            plot_single_pe_vs_k_comparison(m, n, tuned_k_data, baseline_k_data)
