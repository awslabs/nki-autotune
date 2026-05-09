"""Shared test fixture for the rmsnorm+matmul kernel (first-class buffers form)."""

from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose

M, K, N = 2048, 2048, 2048
INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}

_F = K
_EPS = 1e-6


@nkigym_kernel
def f_nkigym(lhs, rhs):
    """Synthesised nkigym body for ``rmsnorm(lhs) @ rhs`` (first-class buffers form)."""
    lhs_sbuf = NKIAlloc(location="sbuf", shape=(M, K), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    sum_sq = NKIAlloc(location="sbuf", shape=(M,), dtype="float32")()
    ar_scratch = NKIAlloc(location="sbuf", shape=(M, K), dtype="float32")()
    rms_inv = NKIAlloc(location="sbuf", shape=(M,), dtype="float32")()
    normed = NKIAlloc(location="sbuf", shape=(M, K), dtype="bfloat16")()
    normed_T_psum = NKIAlloc(location="psum", shape=(K, M), dtype="float32")()
    normed_T = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(M, N), dtype="bfloat16")()

    NKILoad()(src=lhs, dst=lhs_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf, dst=ar_scratch, reduce_res=sum_sq)
    NKIActivation(op="rsqrt", scale=1.0 / _F, bias=_EPS)(data=sum_sq, dst=rms_inv)
    NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv, dst=normed)
    NKITranspose()(src=normed, dst=normed_T_psum)
    NKITensorCopy()(src=normed_T_psum, dst=normed_T)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=normed_T, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


def f_numpy(lhs, rhs):
    """Numpy reference for golden comparison in CPU-sim tests."""
    import numpy as np

    lhs_f32 = lhs.astype(np.float32)
    mean_sq = (lhs_f32 * lhs_f32).mean(axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(mean_sq + _EPS)
    normed = (lhs_f32 * inv_rms).astype(lhs.dtype)
    return normed @ rhs
