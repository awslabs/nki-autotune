import os
import pickle
import shutil
from itertools import product

import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt

from src.cache.directories import NKI_CACHE_DIR, get_cache_dir
from src.cache.results import PerformanceMetrics
from src.kernels.rmsnorm_linear import stack_allocated_fused_rms_norm_qkv
from src.tune.benchmark import profile_kernel


def profile(kernel):
    cache_dir = f"{NKI_CACHE_DIR}/{kernel.func_name}"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)
    dtype = nl.bfloat16
    batch = 1
    MNK = list(product([8192], [512], [4096]))
    for M, N, K in MNK:
        lhs = nt.tensor[[batch, M, K], dtype]
        rhs = nt.tensor[[K, N], dtype]
        p99, _ = profile_kernel(kernel, (lhs, rhs))
        cache_dir = get_cache_dir(NKI_CACHE_DIR, kernel, (lhs, rhs))
        perf_results = PerformanceMetrics()
        perf_results.add_result(configs={}, latency=p99)
        perf_results.save(cache_dir=cache_dir)


if __name__ == "__main__":
    os.environ["NEURON_CC_FLAGS"] = "--framework=XLA --target=trn1 --auto-cast=none"
    profile(stack_allocated_fused_rms_norm_qkv)
