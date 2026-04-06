"""Manually squeezed double_matmul kernel — no trivial dimensions.

Tests whether the compiler accepts <=5D access patterns.
"""

import inspect

import numpy as np

from autotune.runner.api import remote_profile
from autotune.runner.types import KernelJob

KERNEL_SRC = '''
import nki
import nki.language as nl
import nki.isa as nisa
from nki.backends.mlir_tracer.tensor import Tensor


@nki.jit
def double_matmul_nkigym_kernel(Q: Tensor, K: Tensor, V: Tensor):
    assert Q.shape == (2048, 128)
    assert Q.dtype == nl.bfloat16
    assert K.shape == (2048, 128)
    assert K.dtype == nl.bfloat16
    assert V.shape == (2048, 128)
    assert V.dtype == nl.bfloat16
    hbm_output = nl.ndarray((2048, 128), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    """nisa.nc_transpose -- Q(d0, d1) -> Q_t(d1, d0)"""
    psum_Q_t_tmp = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
    sbuf_Q = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_Q_t = nl.ndarray((128, 16, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_block_d0 in nl.affine_range(16):
        nisa.dma_copy(dst=sbuf_Q[0:128, 0:128], src=Q[i_block_d0*128:i_block_d0*128+128, 0:128])
        nisa.nc_transpose(dst=psum_Q_t_tmp[0:128, 0:128], data=sbuf_Q[0:128, 0:128])
        nisa.tensor_copy(dst=sbuf_Q_t[0:128, i_block_d0, 0:128], src=psum_Q_t_tmp[0:128, 0:128])

    """nisa.nc_transpose -- K(d2, d1) -> K_t(d1, d2)"""
    psum_K_t_tmp = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
    sbuf_K = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_K_t = nl.ndarray((128, 4, 4, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_block_d2 in nl.affine_range(4):
        for i_ig_d2 in nl.affine_range(4):
            nisa.dma_copy(dst=sbuf_K[0:128, 0:128], src=K[i_block_d2*512+i_ig_d2*128:i_block_d2*512+i_ig_d2*128+128, 0:128])
            nisa.nc_transpose(dst=psum_K_t_tmp[0:128, 0:128], data=sbuf_K[0:128, 0:128])
            nisa.tensor_copy(dst=sbuf_K_t[0:128, i_block_d2, i_ig_d2, 0:128], src=psum_K_t_tmp[0:128, 0:128])

    """nisa.nc_matmul -- Q_t(K=d1, M=d0) x K_t(K=d1, N=d2) -> S(d0, d2)"""
    psum_S = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    sbuf_S = nl.ndarray((128, 16, 4, 4, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_block_d0 in nl.affine_range(16):
        for i_block_d2 in nl.affine_range(4):
            nisa.memset(dst=psum_S[0:128, 0:512], value=0.0)
            nisa.nc_matmul(dst=psum_S[0:128, 0:512], stationary=sbuf_Q_t[0:128, i_block_d0, 0:128], moving=sbuf_K_t.reshape((128, 4, 512))[0:128, i_block_d2, 0:512])
            nisa.tensor_copy(dst=sbuf_S[0:128, i_block_d0, i_block_d2, 0:4, 0:128], src=psum_S[0:128, 0:512].reshape((128, 4, 128)))

    """nisa.nc_transpose -- S(d0, d2) -> S_t(d2, d0)"""
    psum_S_t_tmp = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
    sbuf_S_t = nl.ndarray((128, 4, 16, 4, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_block_d2 in nl.affine_range(4):
        for i_block_d0 in nl.affine_range(16):
            for i_ig_d2 in nl.affine_range(4):
                nisa.nc_transpose(dst=psum_S_t_tmp[0:128, 0:128], data=sbuf_S[0:128, i_block_d0, i_block_d2, i_ig_d2, 0:128])
                nisa.tensor_copy(dst=sbuf_S_t[0:128, i_block_d2, i_block_d0, i_ig_d2, 0:128], src=psum_S_t_tmp[0:128, 0:128])

    """nisa.nc_matmul -- S_t(K=d2, M=d0) x V(K=d2, N=d3) -> output(d0, d3)"""
    psum_output = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
    sbuf_V = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_output = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_block_d0 in nl.affine_range(16):
        nisa.memset(dst=psum_output[0:128, 0:128], value=0.0)
        for i_block_d2 in nl.affine_range(4):
            for i_ig_d2 in nl.affine_range(4):
                nisa.dma_copy(dst=sbuf_V[0:128, 0:128], src=V[i_block_d2*512+i_ig_d2*128:i_block_d2*512+i_ig_d2*128+128, 0:128])
                nisa.nc_matmul(dst=psum_output[0:128, 0:128], stationary=sbuf_S_t[0:128, i_block_d2, i_block_d0, i_ig_d2, 0:128], moving=sbuf_V[0:128, 0:128])
        nisa.tensor_copy(dst=sbuf_output[0:128, 0:128], src=psum_output[0:128, 0:128])
        nisa.dma_copy(dst=hbm_output[i_block_d0*128:i_block_d0*128+128, 0:128], src=sbuf_output[0:128, 0:128])

    return hbm_output
'''


def double_matmul_numpy(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Reference: (Q @ K.T) @ V.

    Args:
        Q: Shape (seq_q, d_k).
        K: Shape (seq_k, d_k).
        V: Shape (seq_k, d_v).

    Returns:
        Output of shape (seq_q, d_v).
    """
    return (Q @ K.T) @ V


if __name__ == "__main__":
    input_specs = {"Q": ((2048, 128), "bfloat16"), "K": ((2048, 128), "bfloat16"), "V": ((2048, 128), "bfloat16")}
    golden_source = inspect.getsource(double_matmul_numpy)
    kernels = {
        "squeezed_double_matmul": KernelJob(
            source=KERNEL_SRC,
            input_specs=input_specs,
            golden_source=golden_source,
            golden_func_name="double_matmul_numpy",
            atol=0.5,
            rtol=0.1,
        )
    }
    remote_profile(
        kernels=kernels,
        hosts=["gym-1", "gym-2", "gym-3", "gym-4", "gym-5", "gym-6"],
        cache_dir="/home/ubuntu/cache/squeezed_test",
        warmup=10,
        iters=100,
    )
