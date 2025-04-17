import os
import pickle
import shutil
from itertools import product
from typing import Tuple

import torch
from torch_xla.core import xla_model as xm

from autotune.baseline.torch_utils import benchmark
from autotune.cache.directories import TORCH_CACHE_DIR


def initialize_tensors(lhs_shape: Tuple, rhs_shape: Tuple, dtype):
    """Initialize tensors for matrix multiplication on XLA device."""
    device = xm.xla_device()
    lhs = torch.rand(lhs_shape, dtype=dtype, device=device)
    rhs = torch.rand(rhs_shape, dtype=dtype, device=device)
    xm.mark_step()
    return lhs, rhs


def profile():
    torch_func = torch.matmul
    cache_dir = f"{TORCH_CACHE_DIR}/{torch_func.__name__}"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)
    shapes = [4096, 8192]
    dtype = torch.bfloat16
    num_warmup = 10
    num_runs = 100
    MNK = list(product(shapes, shapes, shapes))
    perf_results = {}
    for M, N, K in MNK:
        lhs, rhs = initialize_tensors(lhs_shape=(M, K), rhs_shape=(K, N), dtype=dtype)
        p99 = benchmark(torch_func, num_warmup, num_runs, lhs, rhs)
        perf_results[(M, N, K)] = p99
        pickle.dump(perf_results, open(f"{cache_dir}/{torch_func.__name__}.pkl", "wb"))


if __name__ == "__main__":
    os.environ["NEURON_CC_FLAGS"] = "--framework=XLA --target=trn1 --auto-cast=none"
    profile()
