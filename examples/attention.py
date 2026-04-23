"""Causal attention: remote search over sampled KernelContext variants.

softmax(mask(scale * Q @ K^T)) @ V with lower-triangular causal mask.

Expressed as NKI ops (design doc section 1):
  Q_t      = nc_transpose(Q)
  K_t      = nc_transpose(K)
  S        = nc_matmul(Q_t, K_t)          Q @ K^T
  masked_S = affine_select(S, causal)
  scaled_S = tensor_scalar(masked_S * scale)
  neg_max  = tensor_reduce(scaled_S, max, negate)
  exp_S, sum_exp = activation_reduce(scaled_S, exp, add, bias=neg_max)
  inv_sum  = activation(sum_exp, reciprocal)
  exp_S_t  = nc_transpose(exp_S)
  attn     = nc_matmul(exp_S_t, V)        exp_S @ V
  output   = tensor_scalar(attn * inv_sum)

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/attention.py
"""

import inspect
import shutil
from pathlib import Path

import numpy as np

from autotune.runner.api import remote_profile
from autotune.runner.types import KernelJob
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.affine_select import NKIAffineSelect
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose
from nkigym.search import remote_search


def attention_cte_golden(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Causal attention golden on 3D (batch, seq_q, d) / (batch, d, seq_k) / (batch, seq_k, d) fp32 tensors."""
    scale = float(1.0 / np.sqrt(q.shape[-1]))
    s = (q @ k) * scale
    seqlen_q, seqlen_k = s.shape[-2], s.shape[-1]
    causal = np.triu(np.full((seqlen_q, seqlen_k), -np.inf, dtype=np.float32), k=1)
    s = s + causal
    s = s - s.max(axis=-1, keepdims=True)
    exp_s = np.exp(s)
    p = exp_s / exp_s.sum(axis=-1, keepdims=True)
    out = (p @ v).astype(np.float32)
    return out


def attention_nkigym(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Causal attention using nkigym NKIOp classes.

    Args:
        Q: Query tensor of shape (seq_q, d_k).
        K: Key tensor of shape (seq_k, d_k).
        V: Value tensor of shape (seq_k, d_v).

    Returns:
        Output tensor of shape (seq_q, d_v).
    """
    d_k = Q.shape[1]
    scale = 1.0 / np.sqrt(d_k)
    Q_t = NKITranspose()(data=Q)
    K_t = NKITranspose()(data=K)
    S = NKIMatmul()(stationary=Q_t, moving=K_t)
    masked_S = NKIAffineSelect()(
        on_true_tile=S, pattern=[[-1, K.shape[0]]], channel_multiplier=1, on_false_value=-np.inf, cmp_op="greater_equal"
    )
    scaled_S = NKITensorScalar()(data=masked_S, op0="multiply", operand0=scale)
    neg_max = NKITensorReduce()(data=scaled_S, op="maximum", axis=1, negate=True)
    exp_S, sum_exp = NKIActivationReduce()(data=scaled_S, op="exp", reduce_op="add", bias=neg_max)
    inv_sum = NKIActivation()(data=sum_exp, op="reciprocal")
    exp_S_t = NKITranspose()(data=exp_S)
    attn = NKIMatmul()(stationary=exp_S_t, moving=V)
    output = NKITensorScalar()(data=attn, op0="multiply", operand0=inv_sum)
    return output


if __name__ == "__main__":
    seq_len, d_k, d_v = 2048, 128, 128
    input_specs = {
        "Q": ((seq_len, d_k), "bfloat16"),
        "K": ((seq_len, d_k), "bfloat16"),
        "V": ((seq_len, d_v), "bfloat16"),
    }

    CACHE_DIR = Path("/home/ubuntu/cache/attention")
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True)

    output = remote_search(
        func=attention_nkigym,
        input_specs=input_specs,
        hosts=["gym-1", "gym-2", "gym-3"],
        cache_dir=str(CACHE_DIR),
        num_variants=100,
        atol=1e-2,
        rtol=1e-2,
        seed=0,
    )

    REF_CACHE_DIR = Path("/home/ubuntu/cache/attention_cte_ref")
    shutil.rmtree(REF_CACHE_DIR, ignore_errors=True)
    REF_CACHE_DIR.mkdir(parents=True)

    ref_input_specs: dict[str, tuple[tuple[int, ...], str]] = {
        "q": ((1, seq_len, d_k), "bfloat16"),
        "k": ((1, d_k, seq_len), "bfloat16"),
        "v": ((1, seq_len, d_v), "bfloat16"),
    }
    ref_scale = float(1.0 / np.sqrt(d_k))

    ref_kernel_source = (
        "import nki\n"
        "from nkilib.core.attention.attention_cte import attention_cte as _ref_attention_cte\n\n\n"
        "@nki.jit\n"
        "def attention_cte_ref(q, k, v):\n"
        f"    return _ref_attention_cte(q, k, v, scale={ref_scale!r}, causal_mask=True)\n"
    )

    ref_golden_source = "import numpy as np\n\n" + inspect.getsource(attention_cte_golden)

    ref_mac_count = 2 * seq_len * seq_len * d_k

    remote_profile(
        kernels={
            "attention_cte_ref.py": KernelJob(
                source=ref_kernel_source,
                func_name="attention_cte_ref",
                output_shape=(1, seq_len, d_v),
                input_specs=ref_input_specs,
                nkigym_source=ref_golden_source,
                nkigym_func_name=attention_cte_golden.__name__,
                mac_count=ref_mac_count,
                atol=1e-2,
                rtol=1e-2,
                neuronx_cc_args=("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false"),
            )
        },
        hosts=["gym-1", "gym-2", "gym-3"],
        cache_dir=str(REF_CACHE_DIR),
    )
