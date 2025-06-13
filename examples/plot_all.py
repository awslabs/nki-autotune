from autotune.cache.visualize import plot_metric

if __name__ == "__main__":
    cache_root_dir = "/mnt/efs/autotune-cache"
    for metric in ["min_ms", "mfu_estimated_percent"]:
        for workload_pair in [
            ["lhsT_rhs_gemm_np", "lhsT_rhs_GEMM"],
            ["rmsnorm_matmul_golden", "online_rmsnorm_linear_MKN"],
            ["lhs_rhs_gemm_np", "lhs_rhs_gemm"],
        ]:
            plot_metric(cache_root_dir, metric, workload_pair)
