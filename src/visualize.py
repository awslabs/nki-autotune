# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import pickle


def simultaneous_sort(primary, secondary):
    sorted_pairs = sorted(zip(primary, secondary), key=lambda x: x[0])
    primary, secondary = zip(*sorted_pairs)

    # Convert back to lists if needed
    primary = list(primary)
    secondary = list(secondary)
    return primary, secondary


def plot_tuning_results(tuning_results):
    timestamps = [x["time_elapsed"] for x in tuning_results]
    latencies = [x["latency"] for x in tuning_results]
    timestamps, latencies = simultaneous_sort(timestamps, latencies)
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
    plt.savefig("latency_vs_timestamp.pdf", dpi=600)


# if __name__ == "__main__":
#     tuning_results = pickle.load(open("perf_results.pkl", "rb"))
#     plot_tuning_results(tuning_results)
