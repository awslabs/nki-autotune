"""Measured NAKB ``router_topk`` workload targets."""

from __future__ import annotations

from enum import IntEnum

import ml_dtypes
import numpy as np
import torch
import torch.nn.functional as F

from kernel_library import TorchReference
from nkigym.profile import InputSpecs


class RouterActFnType(IntEnum):
    """NAKB router activation choices."""

    SIGMOID = 0
    SOFTMAX = 1


def router_topk_torch_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    w_bias: torch.Tensor | None,
    router_logits: torch.Tensor | None,
    expert_affinities: torch.Tensor | None,
    expert_index: torch.Tensor | None,
    act_fn: RouterActFnType,
    k: int,
    x_hbm_layout: int,
    x_sb_layout: int,
    router_pre_norm: bool = True,
    norm_topk_prob: bool = False,
    use_column_tiling: bool = False,
    use_indirect_dma_scatter: bool = False,
    return_eager_affi: bool = False,
    use_PE_broadcast_w_bias: bool = False,
    shard_on_tokens: bool = False,
    skip_store_expert_index: bool = False,
    skip_store_router_logits: bool = False,
    x_input_in_sbuf: bool = False,
) -> dict[str, torch.Tensor]:
    """PyTorch reference implementation for router top-K."""
    x_th_layout = x_input_in_sbuf or x_hbm_layout == 1
    x_work = x.T if x_th_layout else x
    router_logits_out = x_work.T @ w
    if w_bias is not None:
        router_logits_out = router_logits_out + w_bias
    tokens, experts = router_logits_out.shape
    indices = torch.argsort(-router_logits_out, dim=-1)
    expert_index_out = indices[..., :k]
    if router_pre_norm:
        if act_fn == RouterActFnType.SOFTMAX:
            expert_affinities_full = F.softmax(router_logits_out, dim=-1)
        elif act_fn == RouterActFnType.SIGMOID:
            expert_affinities_full = torch.sigmoid(router_logits_out)
        else:
            raise NotImplementedError(f"Unsupported activation function: {act_fn}")
        if norm_topk_prob:
            expert_affinities_select = torch.zeros((tokens, experts))
            for token_idx in range(tokens):
                for topk_idx in range(k):
                    expert_idx = expert_index_out[token_idx][topk_idx]
                    expert_affinities_select[token_idx][expert_idx] = expert_affinities_full[token_idx][expert_idx]
            expert_affinities_out = expert_affinities_select / torch.sum(expert_affinities_select, dim=1, keepdim=True)
        else:
            expert_affinities_out = expert_affinities_full
    else:
        top_k_values = torch.zeros((tokens, k))
        for token_idx in range(tokens):
            for topk_idx in range(k):
                top_k_values[token_idx][topk_idx] = router_logits_out[token_idx][expert_index_out[token_idx][topk_idx]]
        if act_fn == RouterActFnType.SOFTMAX:
            expert_affinities_topk = F.softmax(top_k_values, dim=-1)
        elif act_fn == RouterActFnType.SIGMOID:
            expert_affinities_topk = torch.sigmoid(top_k_values)
        else:
            raise NotImplementedError(f"Unsupported activation function: {act_fn}")
        expert_affinities_out = torch.zeros((tokens, experts))
        for token_idx in range(tokens):
            for topk_idx in range(k):
                expert_affinities_out[token_idx][expert_index_out[token_idx][topk_idx]] = expert_affinities_topk[
                    token_idx
                ][topk_idx]
    return {
        "router_logits": router_logits_out,
        "expert_index": expert_index_out,
        "expert_affinities": expert_affinities_out,
    }


def _input_generator(input_specs: InputSpecs, seed: int) -> dict[str, np.ndarray]:
    """Generate deterministic arrays matching one strict workload contract."""
    rng = np.random.default_rng(seed)
    special_dtypes = {"bfloat16": np.dtype(ml_dtypes.bfloat16), "float8_e4m3": np.dtype(ml_dtypes.float8_e4m3)}
    inputs: dict[str, np.ndarray] = {}
    for name, (shape, dtype_name) in input_specs.items():
        dtype = special_dtypes.get(dtype_name)
        if dtype is None:
            dtype = np.dtype(dtype_name)
        if np.issubdtype(dtype, np.bool_):
            values = np.ones(shape, dtype=np.bool_)
        elif np.issubdtype(dtype, np.integer):
            values = np.zeros(shape, dtype=dtype)
        elif name == "cos" or "weight" in name or "gamma" in name or "scale" in name:
            values = np.ones(shape, dtype=np.float32)
        elif name == "sin" or "bias" in name:
            values = np.zeros(shape, dtype=np.float32)
        elif name == "expert_affinities":
            values = rng.random(shape, dtype=np.float32)
            denominator = np.sum(values, axis=-1, keepdims=True)
            values = np.divide(values, denominator, out=np.zeros_like(values), where=denominator != 0)
        else:
            values = rng.standard_normal(shape, dtype=np.float32) * 0.1
        inputs[name] = values.astype(dtype)
    return inputs


def _bind_router_reference(
    parameters: tuple[str, ...],
    *,
    w_bias: torch.Tensor | None,
    act_fn: RouterActFnType,
    k: int,
    router_pre_norm: bool,
    norm_topk_prob: bool,
) -> TorchReference:
    """Bind one measured NAKB router configuration."""
    return TorchReference(
        router_topk_torch_ref,
        parameters,
        bound_kwargs={
            "w_bias": w_bias,
            "router_logits": None,
            "expert_affinities": None,
            "expert_index": None,
            "act_fn": act_fn,
            "k": k,
            "x_hbm_layout": 1,
            "x_sb_layout": 0,
            "router_pre_norm": router_pre_norm,
            "norm_topk_prob": norm_topk_prob,
            "use_column_tiling": False,
            "use_indirect_dma_scatter": False,
            "return_eager_affi": False,
            "use_PE_broadcast_w_bias": False,
            "shard_on_tokens": False,
            "skip_store_expert_index": False,
            "skip_store_router_logits": False,
            "x_input_in_sbuf": False,
        },
    )


_torch_ref_0 = _bind_router_reference(
    ("x", "w", "w_bias"), w_bias=None, act_fn=RouterActFnType.SIGMOID, k=4, router_pre_norm=True, norm_topk_prob=False
)
_torch_ref_1 = _bind_router_reference(
    ("x", "w", "w_bias"), w_bias=None, act_fn=RouterActFnType.SIGMOID, k=8, router_pre_norm=True, norm_topk_prob=True
)
_torch_ref_2 = _bind_router_reference(
    ("x", "w"), w_bias=None, act_fn=RouterActFnType.SIGMOID, k=8, router_pre_norm=False, norm_topk_prob=False
)
_torch_ref_3 = _bind_router_reference(
    ("x", "w"), w_bias=None, act_fn=RouterActFnType.SIGMOID, k=8, router_pre_norm=True, norm_topk_prob=False
)
_torch_ref_4 = _bind_router_reference(
    ("x", "w"), w_bias=None, act_fn=RouterActFnType.SIGMOID, k=1, router_pre_norm=False, norm_topk_prob=False
)
_torch_ref_5 = _bind_router_reference(
    ("x", "w"), w_bias=None, act_fn=RouterActFnType.SIGMOID, k=1, router_pre_norm=True, norm_topk_prob=False
)
_torch_ref_6 = _bind_router_reference(
    ("x", "w", "w_bias"), w_bias=None, act_fn=RouterActFnType.SOFTMAX, k=8, router_pre_norm=False, norm_topk_prob=False
)


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {"x": ((32, 3072), "bfloat16"), "w": ((3072, 128), "bfloat16"), "w_bias": ((128,), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.029343287,
        "best_historical_latency_ms": 0.029343287,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {"x": ((8, 8192), "bfloat16"), "w": ((8192, 128), "bfloat16"), "w_bias": ((128,), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.071924054,
        "best_historical_latency_ms": 0.071924054,
    },
    {
        "torch_ref": _torch_ref_2,
        "input_specs": {"x": ((2, 4096), "bfloat16"), "w": ((4096, 128), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.038488273,
        "best_historical_latency_ms": 0.038488273,
    },
    {
        "torch_ref": _torch_ref_3,
        "input_specs": {"x": ((2, 4096), "bfloat16"), "w": ((4096, 128), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.033045782,
        "best_historical_latency_ms": 0.033045782,
    },
    {
        "torch_ref": _torch_ref_4,
        "input_specs": {"x": ((2, 5120), "bfloat16"), "w": ((5120, 128), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.037751608,
        "best_historical_latency_ms": 0.037751608,
    },
    {
        "torch_ref": _torch_ref_4,
        "input_specs": {"x": ((2, 5120), "bfloat16"), "w": ((5120, 16), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.035512445,
        "best_historical_latency_ms": 0.035512445,
    },
    {
        "torch_ref": _torch_ref_5,
        "input_specs": {"x": ((2, 5120), "bfloat16"), "w": ((5120, 128), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.036743276,
        "best_historical_latency_ms": 0.036743276,
    },
    {
        "torch_ref": _torch_ref_5,
        "input_specs": {"x": ((2, 5120), "bfloat16"), "w": ((5120, 16), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.035074945,
        "best_historical_latency_ms": 0.035074945,
    },
    {
        "torch_ref": _torch_ref_6,
        "input_specs": {"x": ((8, 8192), "bfloat16"), "w": ((8192, 128), "bfloat16"), "w_bias": ((128,), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.069405725,
        "best_historical_latency_ms": 0.069405725,
    },
)
