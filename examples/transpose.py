# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np
from neuronpy.core.language import bfloat16

from autotune.core.benchmark import Benchmark
from autotune.core.job import ProfileJobs
from autotune.core.metrics import check_correctness
from autotune.core.tensor import HBMTensor, SBUFTensor, TileCoordinates
from autotune.typing import INPUT_TENSORS_DTYPE, KERNEL_KWARGS_DTYPE, OUTPUT_TENSORS_DTYPE


@nki.jit
def nki_tile_transpose(input_tensor):
    hbm_input_tensor = HBMTensor(input_tensor, axes=("M", "K"))
    par_tile_size = nl.tile_size.gemm_stationary_fmax
    free_tile_size = nl.tile_size.pmax
    tile_coordinates = TileCoordinates()
    tile_coordinates.add_axis("M", 0, math.ceil(hbm_input_tensor.sizes["M"] / par_tile_size))
    tile_coordinates.add_axis("K", 0, math.ceil(hbm_input_tensor.sizes["K"] / free_tile_size))
    loaded_tensor = SBUFTensor(
        par_axis="M", tile_sizes={"M": par_tile_size, "K": free_tile_size}, tile_coordinates=tile_coordinates
    )
    loaded_tensor.load(hbm_input_tensor)
    padded = loaded_tensor.dump()
    loaded_tensor.tile_transpose()
    padded_tileT = loaded_tensor.dump()
    return padded, padded_tileT


def pad_to_128_aligned(matrix):
    # Get current dimensions
    rows, cols = matrix.shape

    # Calculate target dimensions (next multiples of 128)
    target_rows = ((rows + 127) // 128) * 128
    target_cols = ((cols + 127) // 128) * 128

    # Calculate padding needed
    pad_rows = target_rows - rows
    pad_cols = target_cols - cols

    # Pad the matrix with zeros
    padded_matrix = np.pad(matrix, ((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0)

    return padded_matrix


def tile_transpose(input_tensor):
    tile_size = 128
    rows, cols = input_tensor.shape
    golden = np.empty((rows, cols), dtype=input_tensor.dtype)
    for row_start in range(0, rows, tile_size):
        row_end = min(row_start + tile_size, rows)
        for col_start in range(0, cols, tile_size):
            col_end = min(col_start + tile_size, cols)
            tile = input_tensor[row_start:row_end, col_start:col_end]
            transposed_tile = tile.transpose()
            golden[row_start:row_end, col_start:col_end] = transposed_tile
    return golden


def transpose_correctness(
    input_tensors: INPUT_TENSORS_DTYPE, kernel_kwargs: KERNEL_KWARGS_DTYPE, nki_out_tensors: OUTPUT_TENSORS_DTYPE
):
    input_tensor = input_tensors[0]
    print(f"input_tensor = \n{input_tensor.astype(int)}, {input_tensor.shape}")
    nki_padded, nki_padded_tileT = nki_out_tensors
    nki_padded = nl.static_cast(nki_padded, np.float32)
    nki_padded_tileT = nl.static_cast(nki_padded_tileT, np.float32)

    padded = pad_to_128_aligned(input_tensor)
    padded_tileT = tile_transpose(padded)

    for golden, nki_out in zip([padded, padded_tileT], [nki_padded, nki_padded_tileT]):
        print(f"golden = \n{golden.astype(int)}, {golden.shape}")
        print(f"nki_out = \n{nki_out.astype(int)}, {nki_out.shape}")
        check_correctness(desired=golden, actual=nki_out, atol=1e-5, rtol=1e-2)


def add_jobs(jobs, M, K):
    data_type = "float32"
    if data_type == "float32":
        data_type = np.float32
    elif data_type == "bf16":
        data_type = bfloat16
    input_tensor = np.arange(1, M * K + 1).reshape(M, K).astype(data_type)
    jobs.add_job(
        kernel=("/home/ec2-user/workplace/nki-autotune/examples/transpose.py", "nki_tile_transpose"),
        input_tensors=(input_tensor,),
        kernel_kwargs={},
        compiler_flags="--target=trn1 --auto-cast=none --internal-tensorizer-opt-level=nki",
        preprocessing=None,
        postprocessing=transpose_correctness,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GEMM benchmarks with different matrix configurations")
    parser.add_argument(
        "--cache-dir", type=str, default="/mnt/efs/autotune-dev-cache", help="Root directory for the benchmark cache"
    )
    args = parser.parse_args()
    jobs = ProfileJobs()
    add_jobs(jobs, 129, 128)
    tuner = Benchmark(jobs=jobs, cache_root_dir=args.cache_dir)
    tuner()
