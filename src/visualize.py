# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import pickle, re, os, math
from glob import glob
from pprint import pformat
import numpy as np
import seaborn as sns
from typing import Dict, List
from src.cache.directories import TORCH_CACHE_DIR, NKI_CACHE_DIR, TUNED_NKI_CACHE_DIR


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


def plot_tuning_results(tuning_results, fig_dir: str):
    timestamps = [x["time_elapsed"] for x in tuning_results]
    latencies = [x["latency"] for x in tuning_results]
    timestamps, latencies = sort_lists_by_first(timestamps, latencies)
    best_latencies = [min(latencies[: i + 1]) for i in range(len(latencies))]

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the data
    ax.plot(timestamps, best_latencies)
    ax.scatter(timestamps, best_latencies, marker="X", color="r")

    # Set labels and title
    ax.set_xlabel("Timestamp (s)")
    ax.set_ylabel("Best Latency (us)")
    ax.set_title("Latency vs Timestamp")

    # Add grid lines and tick labels
    # ax.grid(True)
    ax.set_xticks(timestamps)
    ax.set_xticklabels([f"{t:.2f}" for t in timestamps], rotation=45, ha="right")

    # Adjust the layout and display the plot
    fig.tight_layout()
    plt.savefig(f"{fig_dir}/tradeoff.pdf", dpi=600)
    plt.close()


def make_sweep_plot(data, save_path, nki_baseline, torch_baseline, subplot_width=5, subplot_height=4):
    # Sort data by M value
    data = dict(sorted(data.items(), key=lambda x: x[0]))
    n_plots = len(data)
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)

    fig = plt.figure(figsize=(subplot_width * n_cols + 2, subplot_height * n_rows))

    # Get unique kernel base names (without _best/_worst suffix)
    kernel_bases = set()
    for results in data.values():
        kernel_bases.update(k.split("_")[0] for k in results.keys() if k.endswith("_best"))
    kernel_bases = sorted(list(kernel_bases))

    # Create color map for kernel base names
    colors = sns.color_palette("husl", len(kernel_bases))
    color_dict = dict(zip(kernel_bases, colors))

    # Store lines for legend
    kernel_lines = []  # For kernel types
    style_lines = [None, None]  # For best/worst styles

    for subplot_idx, ((M, N), results) in enumerate(data.items(), 1):
        print(f"M {M} N {N}\n{pformat(results)}")
        ax = fig.add_subplot(n_rows, n_cols, subplot_idx)

        k_values = results["K"]

        nki_baseline_plot = [nki_baseline[(M, N, k)] for k in k_values]
        nki_line = ax.plot(k_values, nki_baseline_plot, color="black", linestyle="-", marker="o")

        torch_baseline_plot = [torch_baseline[(M, N, k)] for k in k_values]
        torch_line = ax.plot(k_values, torch_baseline_plot, color="black", linestyle="--", marker="o")

        for kernel_base in kernel_bases:
            best_key = f"{kernel_base}_best"
            worst_key = f"{kernel_base}_worst"
            color = color_dict[kernel_base]

            if best_key in results:
                line = ax.plot(k_values, results[best_key], color=color, linestyle="-", marker="o")
                if subplot_idx == 1:  # Store only from first subplot for legend
                    kernel_lines.append(line[0])
                if style_lines[0] is None:
                    style_lines[0] = line[0]

            if worst_key in results:
                line = ax.plot(k_values, results[worst_key], color=color, linestyle="--", marker="x")
                if style_lines[1] is None:
                    style_lines[1] = line[0]

        ax.set_xticks(k_values)
        ax.set_xticklabels(k_values)
        ax.set_xlabel("K")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"M={M}, N={N}")
        ax.grid(True)

        if max(k_values) / min(k_values) > 10:
            ax.set_xscale("log")
        ax.set_yscale("log")

    # Create two legends
    # First legend for kernel types
    first_legend = fig.legend(
        kernel_lines, kernel_bases, bbox_to_anchor=(1.02, 0.6), loc="center left", title="Kernel Types", borderaxespad=0
    )

    baseline_legend = fig.legend(
        [nki_line[0], torch_line[0]],
        ["NKI", "Torch"],
        bbox_to_anchor=(1.1, 0.6),
        loc="center left",
        title="Baselines",
        borderaxespad=0,
    )

    # Second legend for line styles
    fig.legend(
        style_lines,
        ["Best Meta Parameter", "Worst Meta Parameter"],
        bbox_to_anchor=(1.02, 0.3),
        loc="center left",
        title="Performance",
        borderaxespad=0,
    )

    # Add legends to figure
    fig.add_artist(first_legend)
    fig.add_artist(baseline_legend)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def find_latencies(profiling_results: List):
    latencies = {"best": {}, "worst": {}}
    for result in profiling_results:
        kernel_name = result["configs"]["loop_order"]
        latency = result["latency"]
        if latency == float("inf"):
            continue
        if kernel_name not in latencies["best"] or latency < latencies["best"][kernel_name]:
            latencies["best"][kernel_name] = latency
        if kernel_name not in latencies["worst"] or latency > latencies["worst"][kernel_name]:
            latencies["worst"][kernel_name] = latency
    return latencies


def read_tuning_data(prefix):
    pattern = r"M(\d+)-N(\d+)-K(\d+)"
    fig_data = {}
    for folder in glob(f"{prefix}/*"):
        match = re.search(pattern, folder)
        if match:
            M, N, K = map(int, match.groups())
        if os.path.exists(f"{folder}/tune.pkl"):
            tuning_result = pickle.load(open(f"{folder}/tune.pkl", "rb"))
        else:
            continue
        latencies = find_latencies(tuning_result)
        if (M, N) not in fig_data:
            fig_data[(M, N)] = {"K": []}
        fig_data[(M, N)]["K"].append(K)
        for category in latencies:
            for kernel_name in latencies[category]:
                kernel_latency = latencies[category][kernel_name]
                latency_type = f"{kernel_name}_{category}"
                if latency_type not in fig_data[(M, N)]:
                    fig_data[(M, N)][latency_type] = []
                fig_data[(M, N)][latency_type].append(kernel_latency)
    sorted_fig_data = {}
    for MN in fig_data:
        M, N = MN
        sorted_fig_data[(M, N)] = sort_by_key(fig_data[MN], "K")
    return sorted_fig_data


if __name__ == "__main__":
    nki_baseline = pickle.load(
        open(f"{NKI_CACHE_DIR}/nki_matmul_fully_optimized_/nki_matmul_fully_optimized_.pkl", "rb")
    )
    torch_baseline = pickle.load(open(f"{TORCH_CACHE_DIR}/matmul/matmul.pkl", "rb"))
    tuned_data = read_tuning_data(TUNED_NKI_CACHE_DIR)
    make_sweep_plot(tuned_data, "./matmul_sweep.pdf", nki_baseline, torch_baseline)
