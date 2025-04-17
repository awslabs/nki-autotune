import os
import pickle
import shutil
from itertools import product

import torch

from autotune.baseline.torch_utils import benchmark, initialize_xla_tensor
from autotune.cache.directories import TORCH_CACHE_DIR


def silu(x):
    return x * torch.sigmoid(x)


def torch_fun(lhs, rhs, eps):
    rms = torch.sqrt(torch.mean(lhs.pow(2), dim=-1, keepdim=True) + eps)
    output = lhs * rms.reciprocal()
    output = torch.matmul(output, rhs)
    return output


def profile():
    cache_dir = f"{TORCH_CACHE_DIR}/rmsnorm_linear"
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
        lhs = initialize_xla_tensor(shape=(M, K), dtype=dtype)
        rhs = initialize_xla_tensor(shape=(K, N), dtype=dtype)
        p99 = benchmark(torch_fun, num_warmup, num_runs, lhs, rhs)
        perf_results[(M, N, K)] = p99
        pickle.dump(perf_results, open(f"{cache_dir}/latency.pkl", "wb"))


if __name__ == "__main__":
    os.environ["NEURON_CC_FLAGS"] = "--framework=XLA --target=trn1 --auto-cast=none"
    profile()
