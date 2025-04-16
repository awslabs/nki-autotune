# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

from src.cache.directories import VISUALIZATION_DIR
from src.cache.results import PerformanceMetrics
from src.tune.metrics import calculate_pe_utilization


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


def collect_pe_data(directory):
    """
    Collect PE utilization data for all MNK combinations in a directory.

    Parameters:
    -----------
    directory : str
        Directory containing M-N-K subdirectories with perf_metrics.json files.

    Returns:
    --------
    dict
        Dictionary mapping (M, N) pairs to dictionaries of K values and their PE utilization.
    """
    pe_utilization_data = defaultdict(dict)

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
                best_config = loaded_metrics.get_best_result()
                pe_utilization = calculate_pe_utilization((k, m), (k, n), best_config.min_ms, "trn1")
                pe_utilization_data[(m, n)][k] = pe_utilization

    return pe_utilization_data


def plot_single_pe_vs_k_comparison(m, n, tuned_data, baseline_data):
    """
    Create a single plot comparing tuned vs baseline PE utilization vs K for a specific (M, N) pair.

    Parameters:
    -----------
    m, n : int
        M and N dimensions.
    tuned_data : dict
        Dictionary mapping K values to their respective PE utilization for tuned configs.
    baseline_data : dict
        Dictionary mapping K values to their respective PE utilization for baseline configs.
    """
    plt.figure(figsize=(10, 6))

    # Plot tuned data if available
    if tuned_data:
        k_values = sorted(tuned_data.keys())
        pe_util_values = [tuned_data[k] for k in k_values]
        plt.plot(k_values, pe_util_values, "o-", linewidth=2, color="blue", label="Tuned")

    # Plot baseline data if available
    if baseline_data:
        k_values = sorted(baseline_data.keys())
        pe_util_values = [baseline_data[k] for k in k_values]
        plt.plot(k_values, pe_util_values, "s--", linewidth=2, color="red", label="Baseline")

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
    Create plots comparing PE utilization vs K for each unique (M, N) pair between tuned and baseline.

    Parameters:
    -----------
    kernel : object
        The kernel object containing func_name attribute.
    tuned_cache_dir : str
        Base cache directory for tuned results (TUNED_NKI_CACHE_DIR).
    baseline_dir : str
        Base directory for baseline results, which contains GEMM directory.
    """

    # Make sure output directory exists
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    # Collect data from tuned and baseline directories
    tuned_data = collect_pe_data(tuned_dir)
    baseline_data = collect_pe_data(baseline_dir)

    # Get all unique (M, N) pairs from both datasets
    all_mn_pairs = set(tuned_data.keys()).union(baseline_data.keys())

    # Create plots for each (M, N) pair
    for m, n in all_mn_pairs:
        tuned_k_data = tuned_data.get((m, n), {})
        baseline_k_data = baseline_data.get((m, n), {})

        # Only create plot if we have data for either tuned or baseline
        if tuned_k_data or baseline_k_data:
            plot_single_pe_vs_k_comparison(m, n, tuned_k_data, baseline_k_data)
