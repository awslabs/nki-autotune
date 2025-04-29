import numpy as np
from neuronpy.core.language import bfloat16

from autotune.cache.directories import get_cache_dir
from autotune.cache.loader import get_best_result
from autotune.tune.utils import run_kernel

if __name__ == "__main__":
    workload_name = "fused_rmsnorm_GEMM"
    tuned_dir = get_cache_dir(workload_name, "tuned")
    batch = 1
    M = 4096
    N = 4096
    K = 16384
    lhs = np.random.randn(batch, M, K).astype(bfloat16)
    rhs = np.random.randn(K, N).astype(bfloat16)
    eps = 1e-6
    rtol = 1e-3
    atol = 1e-3
    best_result = get_best_result(f"{tuned_dir}/M{M}-N{N}-K{K}/perf_metrics.json")
    print(best_result["config"])
    kernel_output, metrics = run_kernel("blocked_fused_rms_norm_linear", (lhs, rhs), best_result["config"])
    print(metrics)

    # np_output = rmsnorm_linear_op(lhs, rhs,eps)
    # golden = rmsnorm_linear_golden(lhs, None, None, rhs, eps)
    # comparison = allclose(kernel_output, np_output, rtol=rtol, atol=atol)
    # print(comparison)
    # comparison = allclose(kernel_output, golden, rtol=rtol, atol=atol)
    # print(comparison)
    # comparison = allclose(np_output, golden, rtol=rtol, atol=atol)
    # print(comparison)
