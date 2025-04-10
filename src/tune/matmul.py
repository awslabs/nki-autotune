# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
import re
import shutil
from collections import defaultdict
from itertools import permutations, product
from typing import Dict, List

import matplotlib.pyplot as plt
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt

from src.cache.directories import TUNED_NKI_CACHE_DIR
from src.kernels.matmul import MatMulCompatibility, matmul_main
from src.tune.autotune_kernel import Autotune


def get_autotune_configs() -> List[Dict]:
    """
    Define a list of configuration dictionaries representing the specific design choices for autotuning.

    Returns:
        list: A list of dictionaries, each containing configuration parameters for NUM_BLOCK_M,
                NUM_BLOCK_N, and NUM_BLOCK_K.
    """
    NUM_BLOCK_M_options = [1, 2, 4, 8, 16, 32, 64]
    NUM_BLOCK_N_options = [1, 2, 4, 8, 16, 32, 64]
    NUM_BLOCK_K_options = [1, 2, 4, 8, 16, 32, 64]
    BUFFER_M_options = [1, 2, 4, 8]
    BUFFER_N_options = [1, 2, 4, 8]
    BUFFER_K_options = [1, 2, 4, 8]
    loop_orders = ["".join(p) for p in permutations("MNK")]
    params = list(
        product(
            NUM_BLOCK_M_options,
            NUM_BLOCK_N_options,
            NUM_BLOCK_K_options,
            BUFFER_M_options,
            BUFFER_N_options,
            BUFFER_K_options,
            loop_orders,
        )
    )
    configs = []
    for NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K, BUFFER_M, BUFFER_N, BUFFER_K, loop_order in params:
        config = {
            "NUM_BLOCK_M": NUM_BLOCK_M,
            "NUM_BLOCK_N": NUM_BLOCK_N,
            "NUM_BLOCK_K": NUM_BLOCK_K,
            "BUFFER_M": BUFFER_M,
            "BUFFER_N": BUFFER_N,
            "BUFFER_K": BUFFER_K,
            "loop_order": loop_order,
        }
        configs.append(config)
    random.shuffle(configs)
    return configs


def profile(kernel):
    cache_root = TUNED_NKI_CACHE_DIR
    if os.path.exists(cache_root):
        shutil.rmtree(cache_root)
    os.makedirs(cache_root)
    dtype = nl.bfloat16
    MNK = list(product([2048, 4096], [2048], [1024, 2048, 4096]))
    for M, N, K in MNK:
        lhsT = nt.tensor[[K, M], dtype]
        rhs = nt.tensor[[K, N], dtype]
        tuner = Autotune(
            kernel=kernel,
            kernel_args=(lhsT, rhs),
            configs=get_autotune_configs(),
            max_configs=100,
            pruning_func=MatMulCompatibility,
            cache_dir=f"{TUNED_NKI_CACHE_DIR}/{kernel.func_name}/M{M}-N{N}-K{K}",
            trace=False,
        )
        tuner()


def plot_pe_vs_k(kernel):
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
    kernel_dir = f"{TUNED_NKI_CACHE_DIR}/{kernel.func_name}"
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


if __name__ == "__main__":
    os.environ["NEURON_CC_FLAGS"] = "--framework=XLA --target=trn1 --auto-cast=none"
    kernel = matmul_main
    # profile(kernel)
    plot_pe_vs_k(kernel)
