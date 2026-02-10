"""Generate comparison plots for all kernel workload pairs."""

from autotune.analysis.visualize import plot_metric

if __name__ == "__main__":
    cache_root_dir = "/mnt/efs/autotune-cache"
    for metric in ["min_ms", "mfu_estimated_percent", "hfu_estimated_percent"]:
        for workload_pair in [
            ["rmsnorm_matmul_golden", "online_rmsnorm_linear_MKN"],
            ["lhs_rhs_gemm_np", "lhs_rhs_gemm"],
            ["softmax_gemm_np", "online_softmax_linear_MKN"],
        ]:
            plot_metric(cache_root_dir, metric, workload_pair)
