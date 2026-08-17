"""Measured NAKB ``moe_block_tkg`` workload targets."""

from __future__ import annotations

from enum import IntEnum

import ml_dtypes
import nki.language as nl
import numpy as np
import torch
import torch.nn.functional as F

from kernel_library import TorchReference
from nkigym.profile import InputSpecs


class ActFnType(IntEnum):
    """NAKB activation choices."""

    SiLU = 0
    GELU = 1
    GELU_Tanh_Approx = 2
    Swish = 3
    ReLU = 4


class RouterActFnType(IntEnum):
    """NAKB router activation choices."""

    SIGMOID = 0
    SOFTMAX = 1


class ExpertAffinityScaleMode(IntEnum):
    """NAKB expert affinity scaling choices."""

    NO_SCALE = 0
    POST_SCALE = 1
    PRE_SCALE = 2
    PRE_SCALE_DELAYED = 3


def rms_norm_torch_ref(
    hidden: torch.Tensor, gamma: torch.Tensor | None, eps: float = 1e-6, hidden_actual: int | None = None, **_: object
) -> torch.Tensor:
    """PyTorch reference implementation of RMS normalization."""
    hidden = hidden.to(torch.float32)
    if hidden_actual is not None:
        sum_squares = hidden.square().sum(dim=-1, keepdim=True)
        rms = (sum_squares / hidden_actual + eps).sqrt()
    else:
        rms = (hidden.square().mean(dim=-1, keepdim=True) + eps).sqrt()
    norm = hidden * rms.reciprocal()
    if gamma is not None:
        norm *= gamma
    return norm


def router_topk_torch_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    w_bias: torch.Tensor | None,
    router_logits: torch.Tensor,
    expert_affinities: torch.Tensor,
    expert_index: torch.Tensor,
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


def _compute_expert_mlp(
    hidden_input: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_bias: torch.Tensor | None,
    down_bias: torch.Tensor | None,
    act_fn: str,
    gate_clamp_upper: float | None,
    up_clamp_upper: float | None,
    up_clamp_lower: float | None,
    gate_up_scale: torch.Tensor | None = None,
    down_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the copied NAKB MLP for one expert."""
    gate_weight = gate_up_weight[:, 0, :]
    up_weight = gate_up_weight[:, 1, :]
    if gate_up_scale is not None:
        gate_weight = gate_weight.float() * gate_up_scale[0:1, :]
        up_weight = up_weight.float() * gate_up_scale[1:2, :]
    gate_out = torch.matmul(hidden_input.float(), gate_weight.float())
    if gate_up_bias is not None:
        gate_out = gate_out + gate_up_bias[0, :]
    if gate_clamp_upper is not None:
        gate_out = torch.clamp(gate_out, max=gate_clamp_upper)
    if act_fn == "silu":
        gate_out = F.silu(gate_out)
    elif act_fn == "swish":
        gate_out = gate_out * torch.sigmoid(1.702 * gate_out)
    elif act_fn == "gelu":
        gate_out = F.gelu(gate_out)
    elif act_fn == "gelu_tanh":
        gate_out = F.gelu(gate_out, approximate="tanh")
    up_out = torch.matmul(hidden_input.float(), up_weight.float())
    if gate_up_bias is not None:
        up_out = up_out + gate_up_bias[1, :]
    if up_clamp_upper is not None or up_clamp_lower is not None:
        up_out = torch.clamp(
            up_out,
            min=up_clamp_lower if up_clamp_lower is not None else float("-inf"),
            max=up_clamp_upper if up_clamp_upper is not None else float("inf"),
        )
    intermediate = gate_out * up_out
    if down_scale is not None:
        down_weight = down_weight.float() * down_scale.unsqueeze(0)
    expert_out = torch.matmul(intermediate, down_weight.float())
    if down_bias is not None:
        expert_out = expert_out + down_bias
    return expert_out.to(hidden_input.dtype)


def moe_tkg_torch_ref(
    hidden_input: torch.Tensor,
    expert_gate_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
    expert_affinities: torch.Tensor,
    expert_index: torch.Tensor,
    is_all_expert: bool,
    rank_id: torch.Tensor | None = None,
    expert_gate_up_bias: torch.Tensor | None = None,
    expert_down_bias: torch.Tensor | None = None,
    expert_gate_up_weights_scale: torch.Tensor | None = None,
    expert_down_weights_scale: torch.Tensor | None = None,
    hidden_input_scale: torch.Tensor | None = None,
    gate_up_input_scale: torch.Tensor | None = None,
    down_input_scale: torch.Tensor | None = None,
    mask_unselected_experts: bool = False,
    expert_affinities_eager: torch.Tensor | None = None,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode | None = None,
    activation_fn: ActFnType | None = None,
    output_dtype: torch.dtype | None = None,
    gate_clamp_upper_limit: float | None = None,
    gate_clamp_lower_limit: float | None = None,
    up_clamp_upper_limit: float | None = None,
    up_clamp_lower_limit: float | None = None,
    output_in_sbuf: bool = False,
    is_all_expert_dynamic: bool = False,
    block_size: int | None = None,
    input_dequant_scale: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """PyTorch reference implementation of MoE token generation."""
    act_fn = "silu"
    if activation_fn is not None:
        act_fn = {0: "silu", 1: "gelu", 2: "gelu_tanh", 3: "swish"}.get(int(activation_fn.value), "silu")
    scale_mode = 0
    if expert_affinities_scaling_mode is not None:
        scale_mode = int(expert_affinities_scaling_mode.value)
    is_fp8_row = expert_gate_up_weights_scale is not None and len(expert_gate_up_weights_scale.shape) == 3
    tokens, hidden = hidden_input.shape
    experts = expert_gate_up_weights.shape[0]
    output = torch.zeros(tokens, hidden, dtype=hidden_input.dtype, device=hidden_input.device)
    if is_all_expert:
        for expert in range(experts):
            affinity = expert_affinities[:, expert : expert + 1]
            if affinity.sum() == 0:
                continue
            gate_up_scale = expert_gate_up_weights_scale[expert] if is_fp8_row else None
            down_scale = expert_down_weights_scale[expert] if is_fp8_row else None
            expert_out = _compute_expert_mlp(
                hidden_input,
                expert_gate_up_weights[expert],
                expert_down_weights[expert],
                expert_gate_up_bias[expert] if expert_gate_up_bias is not None else None,
                expert_down_bias[expert] if expert_down_bias is not None else None,
                act_fn,
                gate_clamp_upper_limit,
                up_clamp_upper_limit,
                up_clamp_lower_limit,
                gate_up_scale,
                down_scale,
            )
            if scale_mode == 1:
                expert_out = affinity * expert_out
            output = output + expert_out
    else:
        tokens, top_k = expert_index.shape
        for token in range(tokens):
            for selected in range(top_k):
                expert = int(expert_index[token, selected].item())
                affinity = expert_affinities[token, expert].unsqueeze(0).unsqueeze(0)
                gate_up_scale = expert_gate_up_weights_scale[expert] if is_fp8_row else None
                down_scale = expert_down_weights_scale[expert] if is_fp8_row else None
                token_input = hidden_input[token : token + 1]
                expert_out = _compute_expert_mlp(
                    token_input,
                    expert_gate_up_weights[expert],
                    expert_down_weights[expert],
                    expert_gate_up_bias[expert] if expert_gate_up_bias is not None else None,
                    expert_down_bias[expert] if expert_down_bias is not None else None,
                    act_fn,
                    gate_clamp_upper_limit,
                    up_clamp_upper_limit,
                    up_clamp_lower_limit,
                    gate_up_scale,
                    down_scale,
                )
                if scale_mode == 1:
                    expert_out = affinity * expert_out
                output[token] = output[token] + expert_out.squeeze(0)
    return {"out": output.to(hidden_input.dtype)}


def moe_block_tkg_torch_ref(
    inp: torch.Tensor,
    gamma: torch.Tensor,
    router_weights: torch.Tensor,
    expert_gate_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
    shared_expert_gate_w: torch.Tensor | None = None,
    shared_expert_up_w: torch.Tensor | None = None,
    shared_expert_down_w: torch.Tensor | None = None,
    expert_gate_up_weights_scale: torch.Tensor | None = None,
    expert_down_weights_scale: torch.Tensor | None = None,
    router_bias: torch.Tensor | None = None,
    expert_gate_up_bias: torch.Tensor | None = None,
    expert_down_bias: torch.Tensor | None = None,
    shared_expert_gate_bias: torch.Tensor | None = None,
    shared_expert_up_bias: torch.Tensor | None = None,
    shared_expert_down_bias: torch.Tensor | None = None,
    eps: float = 1e-6,
    top_k: int = 1,
    router_act_fn: RouterActFnType = RouterActFnType.SIGMOID,
    router_pre_norm: bool = True,
    norm_topk_prob: bool = False,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.NO_SCALE,
    hidden_act_fn: ActFnType | None = None,
    hidden_act_scale_factor: float | None = None,
    hidden_act_bias: float | None = None,
    gate_clamp_upper_limit: float | None = None,
    gate_clamp_lower_limit: float | None = None,
    up_clamp_upper_limit: float | None = None,
    up_clamp_lower_limit: float | None = None,
    router_mm_dtype: object | None = None,
    hidden_actual: int | None = None,
    skip_router_logits: bool = False,
    is_all_expert: bool = False,
    rank_id: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    gate_up_input_scale: torch.Tensor | None = None,
    down_input_scale: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Composite torch reference for the NAKB MoE block."""
    batch_size, sequence_length, hidden = inp.shape
    tokens = batch_size * sequence_length
    dtype = inp.dtype
    rmsnorm_out = rms_norm_torch_ref(inp, gamma, eps=eps, hidden_actual=hidden_actual)
    rmsnorm_out = rmsnorm_out.to(dtype).reshape(tokens, hidden)
    _, experts = router_weights.shape
    router_outputs = router_topk_torch_ref(
        x=rmsnorm_out,
        w=router_weights,
        w_bias=router_bias,
        router_logits=torch.zeros(tokens, experts, dtype=dtype),
        expert_affinities=torch.zeros(tokens, experts, dtype=dtype),
        expert_index=torch.zeros(tokens, top_k, dtype=torch.int32),
        act_fn=router_act_fn,
        k=top_k,
        x_hbm_layout=1,
        x_sb_layout=0,
        router_pre_norm=router_pre_norm,
        norm_topk_prob=norm_topk_prob,
    )
    expert_affinities_out = router_outputs["expert_affinities"]
    expert_index_out = router_outputs["expert_index"]
    if is_all_expert and rank_id is not None:
        local_experts = expert_gate_up_weights.shape[0]
        expert_offset = int(rank_id[0, 0].item()) * local_experts
        expert_affinities_out = expert_affinities_out[:, expert_offset : expert_offset + local_experts].clone()
        if router_pre_norm:
            for expert in range(local_experts):
                mask = (expert_index_out == expert_offset + expert).any(dim=1).to(expert_affinities_out.dtype)
                expert_affinities_out[:, expert] *= mask
    moe_outputs = moe_tkg_torch_ref(
        hidden_input=rmsnorm_out,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_affinities=expert_affinities_out,
        expert_index=expert_index_out,
        is_all_expert=is_all_expert,
        rank_id=rank_id,
        expert_gate_up_bias=expert_gate_up_bias,
        expert_down_bias=expert_down_bias,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        gate_up_input_scale=gate_up_input_scale,
        down_input_scale=down_input_scale,
        mask_unselected_experts=router_pre_norm,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        activation_fn=hidden_act_fn,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
    )
    result = {"out": moe_outputs["out"]}
    if not skip_router_logits:
        result["router_logits"] = router_outputs["router_logits"]
    return result


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


_torch_ref_0 = TorchReference(
    moe_block_tkg_torch_ref,
    ("inp", "gamma", "router_weights", "expert_gate_up_weights", "expert_down_weights"),
    bound_kwargs={
        "shared_expert_gate_w": None,
        "shared_expert_up_w": None,
        "shared_expert_down_w": None,
        "expert_gate_up_weights_scale": None,
        "expert_down_weights_scale": None,
        "router_bias": None,
        "expert_gate_up_bias": None,
        "expert_down_bias": None,
        "shared_expert_gate_bias": None,
        "shared_expert_up_bias": None,
        "shared_expert_down_bias": None,
        "eps": 1e-6,
        "top_k": 8,
        "router_act_fn": RouterActFnType.SOFTMAX,
        "router_pre_norm": False,
        "norm_topk_prob": False,
        "expert_affinities_scaling_mode": ExpertAffinityScaleMode.POST_SCALE,
        "hidden_act_fn": ActFnType.Swish,
        "hidden_act_scale_factor": None,
        "hidden_act_bias": None,
        "gate_clamp_upper_limit": None,
        "gate_clamp_lower_limit": None,
        "up_clamp_upper_limit": None,
        "up_clamp_lower_limit": None,
        "router_mm_dtype": nl.bfloat16,
        "hidden_actual": None,
        "skip_router_logits": False,
        "is_all_expert": False,
        "rank_id": None,
        "residual": None,
        "gate_up_input_scale": None,
        "down_input_scale": None,
    },
)


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {
            "inp": ((1, 1, 4096), "bfloat16"),
            "gamma": ((1, 4096), "bfloat16"),
            "router_weights": ((4096, 128), "bfloat16"),
            "expert_gate_up_weights": ((128, 4096, 2, 384), "bfloat16"),
            "expert_down_weights": ((128, 384, 4096), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 1e-05,
        "rtol": 1e-05,
        "nakb_latency_ms": 0.231192972,
        "best_historical_latency_ms": 0.231192972,
    },
)
