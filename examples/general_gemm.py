from autotune.generation.specialize import save_code_to_file, specialize_kernel
from autotune.modules.lhsT_rhs import check_template, lhsT_rhs_gemm_general

if __name__ == "__main__":
    config = {
        "loop_order": {"M": 0, "N": 1, "K": 2},
        "tensor_positions": {"rhs_block": 2, "lhsT_block": 2, "result_block": 1, "matmul": 2},
    }
    check_template(**config)
    kernel_code = specialize_kernel(lhsT_rhs_gemm_general, ["maybe_init", "maybe_compute", "maybe_save"], **config)
    save_code_to_file("generated_kernels/generated_lhsT_rhs.py", kernel_code, lhsT_rhs_gemm_general)
