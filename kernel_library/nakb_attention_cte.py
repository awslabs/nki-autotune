# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pytorch reference for attention_cte kernel

"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import torch

from kernel_library import TorchReference
from nkigym.profile import InputSpecs


def kernel_assert(condition: bool, error_text: str) -> None:
    """Raise NAKB's kernel validation assertion."""
    assert condition, (  # noqa: S101
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text} - "
        "Please check the validation message and adjust kernel inputs accordingly"
    )


def assert_shape(tensor: torch.Tensor, expected_shape: tuple[int, ...], tensor_name: str, error_text: str = "") -> None:
    """Assert that one tensor has the expected shape."""
    kernel_assert(
        tensor.shape == expected_shape,
        f"Received unexpected shape for {tensor_name}. "
        f"Expected {expected_shape}, received {tensor.shape}. {error_text}",
    )


def attention_cte_torch_ref(
    q: torch.tensor,
    k: torch.tensor,
    v: torch.tensor,
    scale: float = 1.0,
    causal_mask: bool = True,
    k_prior=None,
    v_prior=None,
    prior_used_len=None,
    sink=None,
    sliding_window=0,
    tp_q=True,
    tp_k=False,
    tp_out=False,
    cache_softmax=False,
    softmax_dtype=torch.float32,
    mm_out_dtype=torch.float32,
    cp_offset: torch.tensor = None,
    global_cp_deg: int = None,
    cp_strided_q_slicing: bool = False,
    cp_striped_input: bool = False,
    skip_output_normalization: bool = False,
    bound_min=None,
    bound_max=None,
    position_bias=None,
    bias_layout=None,
    bias_band_params=None,
):
    """PyTorch reference implementation for attention_cte NKI kernel.

    This function provides a CPU-based reference implementation with identical
    interface to the NKI kernel for validation and testing purposes.

    Summary:
        Computes multi-head attention with support for causal masking, sliding window,
        prefix caching, context parallelism, and GQA. All computations are performed
        in float32 for CPU compatibility.

    Args:
        q (torch.tensor): Query tensor
        k (torch.tensor): Key tensor
        v (torch.tensor): Value tensor
        scale (float, optional): Scaling factor for attention scores. Default: 1.0
        causal_mask (bool, optional): Whether to apply causal mask. Default: True
        k_prior (torch.tensor, optional): Prior key tensor for prefix caching. Default: None
        v_prior (torch.tensor, optional): Prior value tensor for prefix caching. Default: None
        prior_used_len (torch.tensor, optional): Length of prior to use. Default: None
        sink (torch.tensor, optional): Sink token tensor. Default: None
        sliding_window (int, optional): Sliding window size. Default: 0
        tp_q (bool, optional): Query transpose flag. Default: True
        tp_k (bool, optional): Key transpose flag. Default: False
        tp_out (bool, optional): Output transpose flag. Default: False
        cache_softmax (bool, optional): Whether to cache softmax statistics. Default: False
        softmax_dtype (torch.dtype, optional): Data type for softmax outputs. Default: torch.float32
        mm_out_dtype (torch.dtype, optional): Matmul output dtype (unused in ref). Default: torch.float32
        cp_offset (torch.tensor, optional): Context parallel offset. Default: None
        global_cp_deg (int, optional): Global context parallel degree. Default: None
        cp_strided_q_slicing (bool, optional): Whether Q is strided. Default: False
        bound_min (torch.tensor, optional): Per-query lower bound (inclusive) for
            sequence packing. Shape [bs, seqlen_q, 1]. Default: None
        bound_max (torch.tensor, optional): Per-query upper bound (exclusive) for
            sequence packing. Shape [bs, seqlen_q, 1]. Default: None
        position_bias (torch.tensor, optional): Position bias added to QK scores
            before scaling and masking. Shape ``(bs, seqlen_q, seqlen_kv)`` in the
            kernel's padded KV layout. Semantics: ``mask(scale * (QK + bias))``.
            Default: None
        bias_layout (str, optional): Kernel-only knob; accepted for signature
            compatibility with the kernel and ignored here. The reference always
            consumes a dense ``(bs, seqlen_q, seqlen_kv)`` bias; callers that pass
            ``"banded"`` to the kernel should pass the equivalent dense bias to this
            reference. Default: None
        bias_band_params (dict, optional): Kernel-only knob; accepted for signature
            compatibility and ignored here. Default: None

    Returns:
        dict[str, torch.tensor]: Dictionary with key "out" for the attention output tensor.
            If cache_softmax is True, also includes "out_cached_negative_max" and
            "out_cached_sum_reciprocal".

    Notes:
        - All inputs are converted to float32 for CPU compatibility
        - Supports GQA by replicating K/V tensors
        - Implements flash attention statistics when cache_softmax=True
    """
    # Resolve dtype arguments that may arrive as strings (e.g. "float32") from torch_ref_wrapper
    _dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if isinstance(softmax_dtype, str):
        softmax_dtype = _dtype_map.get(softmax_dtype, torch.float32)
    if isinstance(mm_out_dtype, str):
        mm_out_dtype = _dtype_map.get(mm_out_dtype, torch.float32)

    # Process shapes and configs
    is_prefix_caching = k_prior is not None

    # Convert to float32 because CPU doesn't support half precision
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)

    # Transpose inputs as per flag
    if not tp_q:
        q = q.transpose(1, 2)  # [bs, seqlen_q, d]
    if tp_k:
        k = k.transpose(1, 2)  # [bs, d, seqlen_k]
        if is_prefix_caching:
            k_prior = k_prior.transpose(1, 2)

    bs = q.shape[0]
    bs_kv, d = v.shape[0], v.shape[2]
    seqlen_q, seqlen_k = q.shape[1], k.shape[2]
    kernel_assert(d > 0, f"d must be postive, got {d}")

    # concatenate K and V with prior if prefix caching
    if is_prefix_caching:
        # Convert to float32 because CPU doesn't support half precision
        k_prior = k_prior.to(torch.float32)
        v_prior = v_prior.to(torch.float32)

        prior_used_len = prior_used_len.item()
        k_prior = k_prior[:, :, :prior_used_len]
        v_prior = v_prior[:, :prior_used_len, :]
        k = torch.cat([k_prior, k], dim=2)  # [bs, d, prior_used_len + seqlen_k]
        v = torch.cat([v_prior, v], dim=1)  # [bs, prior_used_len + seqlen_k, d]
    else:
        prior_used_len = 0

    # apply GQA replication if required
    if bs != bs_kv:
        kernel_assert(bs % bs_kv == 0, "Q batch size must be a multiple of KV batch size")
        k = torch.repeat_interleave(k, repeats=bs // bs_kv, dim=0)
        v = torch.repeat_interleave(v, repeats=bs // bs_kv, dim=0)

    # Build mask (causal, sliding window, prior used length for prefix caching)
    cp_offset_val = 0 if cp_offset is None else int(cp_offset.item())
    mask = q.new_full((seqlen_q, prior_used_len + seqlen_k), 0)

    # Define mask in terms of q/k position tensors
    if cp_offset is not None and cp_strided_q_slicing:
        q_positions = cp_offset_val + torch.arange(seqlen_q, dtype=torch.int32) * global_cp_deg
    else:
        q_positions = cp_offset_val + torch.arange(seqlen_q, dtype=torch.int32)
    kv_positions = torch.arange(prior_used_len + seqlen_k, dtype=torch.int32) - prior_used_len
    pos_diff = kv_positions[None, :] - q_positions[:, None]

    # Some torch versions require torch.where args to be tensors
    minus_inf_t = q.new_full((1,), -float("inf"))
    zero_t = q.new_full((1,), 0.0)
    # Generate mask
    if causal_mask:
        mask += torch.where(pos_diff > 0, minus_inf_t, zero_t)
    if sliding_window > 0:
        mask += torch.where(pos_diff <= -sliding_window, minus_inf_t, zero_t)

    # Compute QK, apply position bias (before scale and mask), then scale and mask
    qk = q @ k  # [bs, seqlen_q, prior_used_len + seqlen_k]

    # Add position bias before scaling and masking: final = mask(scale * (QK + bias))
    if position_bias is not None:
        position_bias = position_bias.to(torch.float32)
        kv_len_ref = prior_used_len + seqlen_k  # torch ref KV dim
        if position_bias.shape[2] != kv_len_ref:
            # position_bias is in kernel's padded layout: [prior_padded | active]
            # Extract [0:prior_used_len] from prior region + [prior_padded:prior_padded+seqlen_k] from active
            _K_TILE_SZ_REF = 512
            seqlen_prior_padded = (
                ((k_prior.shape[2] + _K_TILE_SZ_REF - 1) // _K_TILE_SZ_REF) * _K_TILE_SZ_REF if is_prefix_caching else 0
            )
            prior_bias = position_bias[:, :, :prior_used_len]
            active_bias = position_bias[:, :, seqlen_prior_padded : seqlen_prior_padded + seqlen_k]
            position_bias = torch.cat([prior_bias, active_bias], dim=2)
        qk += position_bias

    qk *= scale
    qk += mask[None, :, :]

    # Sequence packing mask: each query only attends to KV positions in [bound_min, bound_max)
    # bound_min/bound_max have shape [bs, seqlen_q, 1]
    if bound_min is not None and bound_max is not None:
        kernel_assert(
            bound_min.shape == (bs, seqlen_q, 1),
            f"bound_min shape mismatch: expected ({bs}, {seqlen_q}, 1), got {bound_min.shape}",
        )
        kernel_assert(
            bound_max.shape == (bs, seqlen_q, 1),
            f"bound_max shape mismatch: expected ({bs}, {seqlen_q}, 1), got {bound_max.shape}",
        )
        kv_len = prior_used_len + seqlen_k
        kv_idx = torch.arange(kv_len, dtype=torch.int32).reshape(1, 1, -1)
        seq_pack_mask = (kv_idx < bound_min) | (kv_idx >= bound_max)  # [bs, seqlen_q, kv_len]
        qk = qk.masked_fill(seq_pack_mask, float("-inf"))

    # Concat sink
    if sink is not None:
        assert_shape(sink, (bs, 1), "sink")
        sink = sink.reshape(bs, 1, 1).expand(-1, seqlen_q, -1)  # [bs, seqlen_q, 1]
        qk = torch.cat([qk, sink], dim=-1)  # [bs, seqlen_q, seqlen_k+1]

    # Softmax + PV matmul
    if cache_softmax:
        # in this case we need to return intermediate tensors
        tile_size = 128
        kernel_assert(
            seqlen_q % tile_size == 0,
            f"For cache softmax, kernel currently expects seqlen_q multiple of {tile_size}, got {seqlen_q=}",
        )
        # Compute softmax with caching of intermediate statistics
        max_value = torch.max(qk, dim=-1, keepdim=True).values  # [bs, seqlen_q, 1]
        qk_shifted = qk - max_value
        exp_values = torch.exp(qk_shifted)
        sum_exp = torch.sum(exp_values, dim=-1, keepdim=True)  # [bs, seqlen_q, 1]

        w = exp_values / sum_exp  # [bs, seqlen_q, seqlen_k+1] if sink else [bs, seqlen_q, seqlen_k]

        # Cache statistics for backward pass
        neg_max = -max_value  # [bs, seqlen_q, 1]
        recip = torch.reciprocal(sum_exp)  # [bs, seqlen_q, 1]

        if sink is not None:
            w = w[..., :-1]  # [bs, seqlen_q, seqlen_k]

        # Compute out, transpose if needed
        if skip_output_normalization:
            # Return unnormalized output: exp(scores - max) @ V
            exp_w = exp_values[..., :-1] if sink is not None else exp_values
            out = exp_w @ v
        else:
            out = w @ v  # [bs, seqlen_q, d]
        out = out.transpose(1, 2) if tp_out else out

        # Reshape neg_max and recip/sum to match [bs, 128, seq_grps]
        seq_grps = seqlen_q // tile_size
        neg_max = neg_max.reshape(bs, seq_grps, tile_size).transpose(1, 2).to(softmax_dtype)

        if skip_output_normalization:
            sum_stat = sum_exp.reshape(bs, seq_grps, tile_size).transpose(1, 2).to(softmax_dtype)
        else:
            sum_stat = recip.reshape(bs, seq_grps, tile_size).transpose(1, 2).to(softmax_dtype)

        return {"out": out, "out_cached_negative_max": neg_max, "out_cached_sum_reciprocal": sum_stat}
    else:
        w = torch.softmax(qk, dim=-1)
        if sink is not None:
            w = w[..., :-1]  # [bs, seqlen_q, seqlen_k]

        # Compute out, transpose if needed
        out = w @ v  # [bs, seqlen_q, d]
        out = out.transpose(1, 2) if tp_out else out
        return {"out": out}


def _input_generator(input_specs: InputSpecs, seed: int) -> dict[str, np.ndarray]:
    """Generate deterministic NAKB sequence-packing attention inputs."""
    rng = np.random.default_rng(seed)
    special_dtypes = {"bfloat16": np.dtype(ml_dtypes.bfloat16)}
    inputs: dict[str, np.ndarray] = {}
    for name, (shape, dtype_name) in input_specs.items():
        dtype = special_dtypes.get(dtype_name)
        if dtype is None:
            dtype = np.dtype(dtype_name)
        values = rng.standard_normal(shape, dtype=np.float32) * 0.1
        inputs[name] = values.astype(dtype)
    batch_size, sequence_length, _ = input_specs["bound_min"][0]
    bound_min = np.zeros((batch_size, sequence_length, 1), dtype=np.int32)
    bound_max = np.zeros((batch_size, sequence_length, 1), dtype=np.int32)
    for batch_index in range(batch_size):
        partitions = np.sort(rng.random(15))
        partitions = np.concatenate(([0.0], partitions, [1.0]))
        segment_lengths = np.round(np.diff(partitions) * sequence_length).astype(int)
        segment_lengths[-1] += sequence_length - segment_lengths.sum()
        segment_ends = np.cumsum(segment_lengths)
        segment_starts = np.concatenate(([0], segment_ends[:-1]))
        for segment_start, segment_end in zip(segment_starts, segment_ends, strict=True):
            bound_min[batch_index, segment_start:segment_end, 0] = segment_start
            bound_max[batch_index, segment_start:segment_end, 0] = segment_end
    inputs["bound_min"] = bound_min.astype(np.float32)
    inputs["bound_max"] = bound_max.astype(np.float32)
    return inputs


def _bind_attention_cte(*, tp_q: bool, tp_k: bool, tp_out: bool) -> TorchReference:
    """Bind one NAKB sequence-packing attention case."""
    return TorchReference(
        attention_cte_torch_ref,
        ("q", "k", "v", "bound_min", "bound_max"),
        bound_kwargs={
            "scale": 1.0,
            "causal_mask": False,
            "k_prior": None,
            "v_prior": None,
            "prior_used_len": None,
            "sink": None,
            "sliding_window": 0,
            "tp_q": tp_q,
            "tp_k": tp_k,
            "tp_out": tp_out,
            "cache_softmax": False,
            "softmax_dtype": torch.float32,
            "mm_out_dtype": torch.float32,
            "cp_offset": None,
            "global_cp_deg": None,
            "cp_strided_q_slicing": False,
            "cp_striped_input": False,
            "skip_output_normalization": False,
            "position_bias": None,
            "bias_layout": "dense",
            "bias_band_params": None,
        },
    )


_torch_ref_0 = _bind_attention_cte(tp_q=True, tp_k=False, tp_out=False)
_torch_ref_1 = _bind_attention_cte(tp_q=True, tp_k=False, tp_out=True)
_torch_ref_2 = _bind_attention_cte(tp_q=False, tp_k=False, tp_out=False)


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {
            "q": ((1, 2048, 128), "bfloat16"),
            "k": ((1, 128, 2048), "bfloat16"),
            "v": ((1, 2048, 128), "bfloat16"),
            "bound_min": ((1, 2048, 1), "float32"),
            "bound_max": ((1, 2048, 1), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.091467357,
        "best_historical_latency_ms": 0.091467357,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {
            "q": ((1, 4096, 128), "bfloat16"),
            "k": ((1, 128, 4096), "bfloat16"),
            "v": ((1, 4096, 128), "bfloat16"),
            "bound_min": ((1, 4096, 1), "float32"),
            "bound_max": ((1, 4096, 1), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.181100551,
        "best_historical_latency_ms": 0.177796389,
    },
    {
        "torch_ref": _torch_ref_2,
        "input_specs": {
            "q": ((2, 96, 2048), "bfloat16"),
            "k": ((2, 96, 2048), "bfloat16"),
            "v": ((2, 2048, 96), "bfloat16"),
            "bound_min": ((2, 2048, 1), "float32"),
            "bound_max": ((2, 2048, 1), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.09397152,
        "best_historical_latency_ms": 0.09397152,
    },
)
