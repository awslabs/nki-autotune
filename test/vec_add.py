"""
For NKI functions, you can refer to the NKI documentation at:
https://awsdocs-neuron-staging.readthedocs-hosted.com/en/nki_docs_2.21_beta_class/
"""

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt

import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from itertools import product
import random
import sys

sys.path.append("../")
from src.autotune_kernel import AutotuneKernel


# This Just Disables the Compiler Logging Because It Clogs the Output
import logging

logging.disable(logging.OFF)


def get_autotune_configs():
    PARTITION_DIM_options = [int(2**x) for x in range(0, 8)]
    FREE_DIM_options = [int(2**x) * 1000 for x in range(0, 4)]

    params = list(product(PARTITION_DIM_options, FREE_DIM_options))

    configs = []

    for PARTITION_DIM, FREE_DIM in params:
        config = {"PARTITION_DIM": PARTITION_DIM, "FREE_DIM": FREE_DIM}
        configs.append(config)

    random.shuffle(configs)
    return configs


"""
This is an extension of the tiled implementation of a vector add kernel
from the Stanford CS 149 assignment "Programming a Machine Learning 
Accelerator". The source code is available at: 
https://github.com/stanford-cs149/asst4-trainium/tree/main.

The input vectors are reshaped into (PARTITION_DIM, FREE_DIM) tiles in 
order to amortize DMA transfer overheads and load many more elements per 
DMA transfer. The input vectors are then added together and the result
tiles are stored in HBM.
"""


def vector_add_stream_auto(a_vec, b_vec, out, PARTITION_DIM=128, FREE_DIM=1000):

    # Get the total number of vector rows
    M = a_vec.shape[0]

    if M % (PARTITION_DIM * FREE_DIM) != 0:
        FREE_DIM = M / PARTITION_DIM

    # The total size of each tile
    TILE_M = PARTITION_DIM * FREE_DIM

    # Reshape the the input vectors
    a_vec_re = a_vec.reshape((M // TILE_M, PARTITION_DIM, FREE_DIM))
    b_vec_re = b_vec.reshape((M // TILE_M, PARTITION_DIM, FREE_DIM))

    out = out.reshape((a_vec_re.shape))

    # Loop over the total number of tiles
    for m in nl.affine_range((M // TILE_M)):

        # Allocate space for a reshaped tile
        a_tile = nl.ndarray(
            (PARTITION_DIM, FREE_DIM), dtype=a_vec.dtype, buffer=nl.sbuf
        )
        b_tile = nl.ndarray(
            (PARTITION_DIM, FREE_DIM), dtype=a_vec.dtype, buffer=nl.sbuf
        )

        # Load the input tiles
        a_tile = nl.load(a_vec_re[m])
        b_tile = nl.load(b_vec_re[m])

        # Add the tiles together
        res = nl.add(a_tile, b_tile)

        # Store the result tile into HBM
        nl.store(out[m], value=res)

    # Reshape the output vector into its original shape
    out = out.reshape((M,))


def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("--size", type=int, required=True)
    args = parser.parse_args()

    # Generate random input arrays
    a = np.random.rand(args.size).astype(np.float32)
    b = np.random.rand(args.size).astype(np.float32)

    print(f"\nRunning vector add with vector size = {args.size}")

    tuner = AutotuneKernel.trace(
        vector_add_stream_auto, configs=get_autotune_configs(), show_compiler_tb=True
    )
    # Allocate space for the reshaped output vector in HBM
    output = nt.tensor[[a.shape[0], 1], a.dtype]
    tuner(a, b, output)


if __name__ == "__main__":
    main()
