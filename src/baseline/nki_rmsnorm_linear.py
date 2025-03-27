import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import os, pickle, shutil
from itertools import product

from src.cache.directories import NKI_CACHE_DIR
from src.benchmark import profile_kernel
from src.kernels.rmsnorm_linear import stack_allocated_fused_rms_norm_qkv


def profile(kernel):
    cache_dir = f"{NKI_CACHE_DIR}/{kernel.func_name}"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)
    dtype = nl.bfloat16
    batch = 1
    MNK = list(product([8192], [512], [4096]))
    perf_results = {}
    for M, N, K in MNK:
        lhs = nt.tensor[[batch, M, K], dtype]
        rhs = nt.tensor[[K, N], dtype]
        p99, _ = profile_kernel(kernel, (lhs, rhs))
        perf_results[(M, N, K)] = p99
        pickle.dump(perf_results, open(f"{cache_dir}/{kernel.func_name}.pkl", "wb"))


if __name__ == "__main__":
    os.environ["NEURON_CC_FLAGS"] = "--framework=XLA --target=trn1 --auto-cast=none"
    profile(stack_allocated_fused_rms_norm_qkv)
