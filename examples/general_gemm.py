import numpy as np

from autotune.generation.specialize import specialize_kernel
from autotune.modules.lhsT_rhs import lhsT_rhs_gemm_general


def save_code_to_file(filepath: str, kernel_code: str):
    with open(filepath, "w") as f:
        f.write(kernel_code)


if __name__ == "__main__":
    lhsT = np.random.normal(size=(1024, 2048))
    rhs = np.random.normal(size=(1024, 4096))
    config = {"tensor_positions": {"result_block": -1, "rhs_block": 1, "lhsT_block": 2}}

    kernel_code = specialize_kernel(lhsT_rhs_gemm_general, ["maybe_init"], **config)
    save_code_to_file("generated_kernels/generated_lhsT_rhs.py", kernel_code)
