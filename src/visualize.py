# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import pickle, re, os, math
from glob import glob
from pprint import pformat
import numpy as np
import seaborn as sns


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


def make_sweep_plot(data, save_path):
    # Calculate number of subfigures needed
    n_plots = len(data)
    # Calculate reasonable grid dimensions
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)

    # Create color map for kernels
    kernel_names = set()
    for results in data.values():
        kernel_names.update(k for k in results.keys() if k.startswith("matmul_"))
    kernel_names = sorted(list(kernel_names))
    colors = sns.color_palette("husl", len(kernel_names))
    color_dict = dict(zip(kernel_names, colors))

    # Create figure and subfigures
    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    # Store one line from each kernel type for legend
    legend_lines = []
    legend_labels = []

    # Plot each M,N pair in a separate subfigure
    for idx, ((M, N), results) in enumerate(data.items(), 1):
        ax = fig.add_subplot(n_rows, n_cols, idx)

        # Get K values for x-axis
        k_values = results["K"]

        for kernel in kernel_names:
            if kernel in results:
                line = ax.plot(
                    k_values, results[kernel], marker="o", color=color_dict[kernel]
                )
                # Store only the first instance of each kernel type
                if kernel not in legend_labels:
                    legend_lines.append(line[0])
                    legend_labels.append(kernel)

        ax.set_xticks(k_values)
        ax.set_xticklabels(k_values)
        ax.set_xlabel("K")
        ax.set_ylabel("Latency (μs)")
        ax.set_title(f"M={M}, N={N}")
        ax.grid(True)

        # Use log scale if K values span multiple orders of magnitude
        if max(k_values) / min(k_values) > 10:
            ax.set_xscale("log")

    fig.legend(
        legend_lines,
        legend_labels,
        bbox_to_anchor=(1.02, 0.5),  # Position legend to the right
        loc="center left",  # Anchor point on the legend
        borderaxespad=0,
    )
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.close()


def find_best_latencies(profiling_results):
    best_latencies = {}
    for result in profiling_results:
        kernel_name = result["configs"]["kernel_name"]
        latency = result["latency"]
        if kernel_name not in best_latencies or latency < best_latencies[kernel_name]:
            best_latencies[kernel_name] = latency
    return best_latencies


def plot_matmul_sweep(prefix):
    pattern = r"M(\d+)-N(\d+)-K(\d+)"
    fig_data = {}
    for folder in glob(f"{prefix}*"):
        match = re.search(pattern, folder)
        if match:
            M, N, K = map(int, match.groups())
        if os.path.exists(f"{folder}/tune.pkl"):
            tuning_result = pickle.load(open(f"{folder}/tune.pkl", "rb"))
        else:
            continue
        best_latencies = find_best_latencies(tuning_result)
        if (M, N) not in fig_data:
            fig_data[(M, N)] = {"K": []}
        fig_data[(M, N)]["K"].append(K)
        for kernel_name in best_latencies:
            kernel_latency = best_latencies[kernel_name]
            if kernel_name not in fig_data[(M, N)]:
                fig_data[(M, N)][kernel_name] = []
            fig_data[(M, N)][kernel_name].append(kernel_latency)
    sorted_fig_data = {}
    for MN in fig_data:
        M, N = MN
        sorted_fig_data[(M, N)] = sort_by_key(fig_data[MN], "K")
    print(f"{pformat(sorted_fig_data)}")
    make_sweep_plot(sorted_fig_data, "./matmul_sweep.pdf")


if __name__ == "__main__":
    plot_matmul_sweep("private/matmul-")
