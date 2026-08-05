"""Ground-truth fixture adapter for contract-driven online fusion."""

from __future__ import annotations

import numpy as np

from examples.online_fusion_attention import f_nkigym
from nkigym.ir import KernelIR, build_initial_ir
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar

ATTENTION_INPUT_SPECS = {
    "query": ((128, 128), "bfloat16"),
    "key": ((256, 128), "bfloat16"),
    "value": ((256, 128), "bfloat16"),
}
MAPPED_ATTENTION_INPUT_SPECS = {
    "query": ((128, 256), "bfloat16"),
    "key": ((128, 256), "bfloat16"),
    "value": ((256, 128), "bfloat16"),
}


def attention_reference(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Compute row-major scaled dot-product attention in FP32."""
    scores = query.astype(np.float32) @ key.astype(np.float32).T / np.sqrt(128)
    scores -= np.max(scores, axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities @ value.astype(np.float32)


@nkigym_kernel
def f_naive_attention(query, key, value):
    """Compute materialized scaled dot-product attention in SSA form."""
    sbuf_query_t = NKIDMATranspose()(src=query)
    sbuf_key_t = NKIDMATranspose()(src=key)
    psum_scores = NKIMatmul()(stationary=sbuf_query_t, moving=sbuf_key_t)
    sbuf_scores = NKITensorCopy()(src=psum_scores)
    sbuf_scaled_scores = NKITensorScalar(op0="multiply")(data=sbuf_scores, operand0=128**-0.5)
    sbuf_row_max = NKITensorReduce(op="maximum", axis=1)(data=sbuf_scaled_scores)
    sbuf_centered = NKITensorScalar(op0="subtract")(data=sbuf_scaled_scores, operand0=sbuf_row_max)
    sbuf_exp = NKIActivation(op="exp")(data=sbuf_centered)
    sbuf_row_sum = NKITensorReduce(op="add", axis=1)(data=sbuf_exp)
    sbuf_inv_sum = NKIActivation(op="reciprocal")(data=sbuf_row_sum)
    sbuf_probability = NKITensorScalar(op0="multiply")(data=sbuf_exp, operand0=sbuf_inv_sum)
    sbuf_probability_t = NKIDMATranspose()(src=sbuf_probability)
    sbuf_value = NKILoad()(src=value)
    psum_output = NKIMatmul()(stationary=sbuf_probability_t, moving=sbuf_value)
    sbuf_output = NKITensorCopy()(src=psum_output)
    hbm_output = NKIStore()(src=sbuf_output)
    return hbm_output


@nkigym_kernel
def f_load_first_attention(query, key, value):
    """Compute attention after first loading every HBM parameter."""
    sbuf_query = NKILoad()(src=query)
    sbuf_key = NKILoad()(src=key)
    sbuf_value = NKILoad()(src=value)
    sbuf_query_t = NKIDMATranspose()(src=sbuf_query)
    sbuf_key_t = NKIDMATranspose()(src=sbuf_key)
    psum_scores = NKIMatmul()(stationary=sbuf_query_t, moving=sbuf_key_t)
    sbuf_scores = NKITensorCopy()(src=psum_scores)
    sbuf_scaled_scores = NKITensorScalar(op0="multiply")(data=sbuf_scores, operand0=128**-0.5)
    sbuf_row_max = NKITensorReduce(op="maximum", axis=1)(data=sbuf_scaled_scores)
    sbuf_centered = NKITensorScalar(op0="subtract")(data=sbuf_scaled_scores, operand0=sbuf_row_max)
    sbuf_exp = NKIActivation(op="exp")(data=sbuf_centered)
    sbuf_row_sum = NKITensorReduce(op="add", axis=1)(data=sbuf_exp)
    sbuf_inv_sum = NKIActivation(op="reciprocal")(data=sbuf_row_sum)
    sbuf_probability = NKITensorScalar(op0="multiply")(data=sbuf_exp, operand0=sbuf_inv_sum)
    sbuf_probability_t = NKIDMATranspose()(src=sbuf_probability)
    psum_output = NKIMatmul()(stationary=sbuf_probability_t, moving=sbuf_value)
    sbuf_output = NKITensorCopy()(src=psum_output)
    hbm_output = NKIStore()(src=sbuf_output)
    return hbm_output


def build_naive_attention_ir() -> KernelIR:
    """Build the canonical materialized-attention IR."""
    return build_initial_ir(f_naive_attention, ATTENTION_INPUT_SPECS)


def build_load_first_attention_ir() -> KernelIR:
    """Build attention with progress-dependent loads interleaved at root."""
    return build_initial_ir(f_load_first_attention, ATTENTION_INPUT_SPECS)


def build_mapped_attention_ir() -> KernelIR:
    """Build attention whose recurrence state spans two query tiles."""
    return build_initial_ir(f_nkigym, MAPPED_ATTENTION_INPUT_SPECS)
