"""Shared test fixture for the rmsnorm+matmul kernel.

Tests cannot import these names from ``examples.rmsnorm_matmul`` because
that module only defines them inside ``if __name__ == "__main__":``.
Duplicating the kernel body here keeps the example script unchanged
while giving tests a stable import site for the parsed-graph fixtures.

Body copied verbatim from
``/home/ubuntu/cache/rmsnorm_matmul_compile/f_nkigym.py`` — regenerate
that cache and this file together when the synthesis output changes.
"""

from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose

M, K, N = 2048, 2048, 2048
INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}

_F = K
_EPS = 1e-6


@nkigym_kernel
def f_nkigym(lhs, rhs):
    """Synthesised nkigym body for ``rmsnorm(lhs) @ rhs``."""
    lhs_sbuf = NKILoad()(data=lhs)
    rhs_sbuf = NKILoad()(data=rhs)
    sum_sq = NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf)
    rms_inv = NKIActivation(op="rsqrt", scale=1.0 / _F, bias=_EPS)(data=sum_sq)
    normed = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
    normed_T = NKITranspose()(data=normed)
    matmul_out = NKIMatmul()(stationary=normed_T, moving=rhs_sbuf)
    out = NKIStore()(data=matmul_out)
    return out


def f_numpy(lhs, rhs):
    """Numpy reference for golden comparison in CPU-sim tests."""
    import numpy as np

    lhs_f32 = lhs.astype(np.float32)
    mean_sq = (lhs_f32 * lhs_f32).mean(axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(mean_sq + _EPS)
    normed = (lhs_f32 * inv_rms).astype(lhs.dtype)
    return normed @ rhs
