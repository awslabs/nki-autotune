"""Measured NAKB ``qkv_tkg`` workload targets."""

from __future__ import annotations

from enum import Enum

import ml_dtypes
import numpy as np
import torch

from kernel_library import TorchReference
from nkigym.profile import InputSpecs

FP8_E4M3_CLIP_VALUE = 240.0


class QKVOutputLayout(Enum):
    """Output layouts copied from NAKB."""

    BSD = 0
    NBSd = 1
    NBdS = 2


class NormType(Enum):
    """Normalization modes copied from NAKB."""

    NO_NORM = 0
    RMS_NORM = 1
    LAYER_NORM = 2
    RMS_NORM_SKIP_GAMMA = 3


class QuantizationType(Enum):
    """Quantization modes copied from NAKB."""

    NONE = 0
    STATIC = 1
    ROW = 2
    MX = 3
    STATIC_MX = 4
    ROW_MX = 5


def rms_norm_torch_ref(
    hidden: torch.Tensor, gamma: torch.Tensor | None, eps: float = 1e-6, hidden_actual: int | None = None, **_: object
) -> torch.Tensor:
    """Apply NAKB's PyTorch RMS normalization."""
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


def layer_norm_torch_ref(
    hidden: torch.Tensor, gamma: torch.Tensor | None, norm_b: torch.Tensor | None = None, eps: float = 1e-6, **_: object
) -> torch.Tensor:
    """Apply NAKB's PyTorch layer normalization."""
    hidden = hidden.to(torch.float32)
    mean = hidden.mean(dim=-1, keepdim=True)
    variance = hidden.var(dim=-1, correction=0, keepdim=True)
    norm = (hidden - mean) * (variance + eps).sqrt().reciprocal().to(hidden.dtype)
    if gamma is not None:
        norm *= gamma
    if norm_b is not None:
        norm += norm_b
    return norm


def _no_norm_torch_ref(hidden: torch.Tensor, *_: object, **__: object) -> torch.Tensor:
    """Return the unnormalized hidden tensor."""
    return hidden


NORM_NAME_TO_TORCH_REF = {
    NormType.NO_NORM: _no_norm_torch_ref,
    NormType.RMS_NORM: rms_norm_torch_ref,
    NormType.LAYER_NORM: layer_norm_torch_ref,
    NormType.RMS_NORM_SKIP_GAMMA: rms_norm_torch_ref,
}


def qkv_tkg_torch_ref(
    hidden: torch.Tensor,
    qkv_w: torch.Tensor,
    norm_w: torch.Tensor | None = None,
    fused_add: bool = False,
    mlp_prev: torch.Tensor | None = None,
    attn_prev: torch.Tensor | None = None,
    d_head: int | None = None,
    num_kv_heads: int | None = None,
    num_q_heads: int | None = None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    eps: float = 1e-6,
    norm_type: NormType = NormType.RMS_NORM,
    quantization_type: QuantizationType = QuantizationType.NONE,
    is_h_dim_4h_transposed: bool = False,
    qkv_w_scale: torch.Tensor | None = None,
    qkv_in_scale: torch.Tensor | None = None,
    output_in_sbuf: bool = False,
    qkv_bias: torch.Tensor | None = None,
    norm_bias: torch.Tensor | None = None,
    hidden_actual: int | None = None,
    transposed_in: bool = False,
) -> dict[str, torch.Tensor]:
    """PyTorch reference implementation copied from NAKB's ``qkv_tkg``."""
    del output_in_sbuf
    hidden = hidden.to(torch.float32)
    if transposed_in:
        h0, n_prgs, h1_shard, batch_sequence = hidden.shape
        hidden_dim = h0 * n_prgs * h1_shard
        hidden = hidden.permute(3, 1, 0, 2).reshape(batch_sequence, 1, hidden_dim)
    mlp_prev = mlp_prev.to(torch.float32) if mlp_prev is not None else None
    attn_prev = attn_prev.to(torch.float32) if attn_prev is not None else None
    norm_w = norm_w.to(torch.float32) if norm_w is not None else None
    norm_bias = norm_bias.to(torch.float32) if norm_bias is not None else None
    qkv_bias = qkv_bias.to(torch.float32) if qkv_bias is not None else None
    if quantization_type not in (QuantizationType.NONE, QuantizationType.STATIC, QuantizationType.ROW):
        raise ValueError(f"unsupported copied qkv_tkg configuration: {quantization_type}")
    if is_h_dim_4h_transposed:
        raise ValueError("is_h_dim_4h_transposed is only valid for NAKB MX configurations")
    qkv_w = qkv_w.to(torch.float32)
    is_static = quantization_type == QuantizationType.STATIC
    if is_static:
        qkv_in_scale = (
            float(np.asarray(qkv_in_scale).flat[0])
            if not isinstance(qkv_in_scale, (int, float))
            else float(qkv_in_scale)
        )
        qkv_w_scale = torch.from_numpy(np.asarray(qkv_w_scale).reshape(-1).astype(np.float32))
    elif quantization_type == QuantizationType.ROW:
        qkv_w_scale = (
            qkv_w_scale[0, :].to(torch.float32)
            if isinstance(qkv_w_scale, torch.Tensor)
            else torch.from_numpy(np.asarray(qkv_w_scale).reshape(-1).astype(np.float32))
        )

    batch, sequence, _ = hidden.shape
    fused_hidden = None
    if fused_add:
        if mlp_prev is None:
            raise ValueError("mlp_prev required when fused_add is True")
        if attn_prev is None:
            raise ValueError("attn_prev required when fused_add is True")
        hidden = hidden + mlp_prev + attn_prev
        fused_hidden = hidden

    if norm_type == NormType.RMS_NORM:
        hidden = NORM_NAME_TO_TORCH_REF[norm_type](
            hidden, norm_w, eps=eps, norm_b=norm_bias, hidden_actual=hidden_actual
        )
    else:
        hidden = NORM_NAME_TO_TORCH_REF[norm_type](hidden, norm_w, eps=eps, norm_b=norm_bias)

    if is_static:
        if d_head is None:
            raise ValueError("d_head required for STATIC quantization")
        if num_q_heads is None:
            raise ValueError("num_q_heads required for STATIC quantization")
        if num_kv_heads is None:
            raise ValueError("num_kv_heads required for STATIC quantization")
        hidden = (hidden / qkv_in_scale).clamp(-FP8_E4M3_CLIP_VALUE, FP8_E4M3_CLIP_VALUE)

    qkv_out = hidden @ qkv_w
    if is_static:
        combined_scale = qkv_in_scale * qkv_w_scale
        q_end_index = num_q_heads * d_head
        k_end_index = (num_q_heads + num_kv_heads) * d_head
        v_end_index = (num_q_heads + 2 * num_kv_heads) * d_head
        qkv_out[:, :, :q_end_index] *= combined_scale[0]
        qkv_out[:, :, q_end_index:k_end_index] *= combined_scale[1]
        qkv_out[:, :, k_end_index:v_end_index] *= combined_scale[2]
    elif quantization_type == QuantizationType.ROW:
        qkv_out = qkv_out * qkv_w_scale

    if qkv_bias is not None:
        qkv_out += qkv_bias

    _, _, projected_dim = qkv_out.shape
    if output_layout in (QKVOutputLayout.NBSd, QKVOutputLayout.NBdS):
        if d_head is None:
            raise ValueError(f"d_head required for {output_layout} output layout")
        num_heads = projected_dim // d_head
    if output_layout == QKVOutputLayout.NBdS:
        qkv_out = torch.reshape(qkv_out, (batch, sequence, num_heads, d_head))
        qkv_out = torch.permute(qkv_out, (2, 0, 3, 1))
    elif output_layout == QKVOutputLayout.NBSd:
        qkv_out = torch.reshape(qkv_out, (batch, sequence, num_heads, d_head))
        qkv_out = torch.permute(qkv_out, (2, 0, 1, 3))
    if transposed_in:
        qkv_out = qkv_out.squeeze(1)
    if fused_add:
        return {"out": qkv_out, "fused_hidden": fused_hidden}
    return {"out": qkv_out}


def _input_generator(input_specs: InputSpecs, seed: int) -> dict[str, np.ndarray]:
    """Generate deterministic arrays matching one NAKB workload contract."""
    rng = np.random.default_rng(seed)
    special_dtypes = {"bfloat16": np.dtype(ml_dtypes.bfloat16), "float8_e4m3": np.dtype(ml_dtypes.float8_e4m3)}
    inputs: dict[str, np.ndarray] = {}
    for name, (shape, dtype_name) in input_specs.items():
        dtype = special_dtypes.get(dtype_name)
        if dtype is None:
            dtype = np.dtype(dtype_name)
        if name == "norm_w" or name.endswith("_scale"):
            values = np.ones(shape, dtype=np.float32)
        elif name.endswith("_bias"):
            values = np.zeros(shape, dtype=np.float32)
        else:
            values = rng.standard_normal(shape, dtype=np.float32) * 0.1
        inputs[name] = values.astype(dtype)
    return inputs


def _bind_qkv(
    parameters: tuple[str, ...],
    *,
    d_head: int,
    num_kv_heads: int,
    num_q_heads: int,
    output_layout: QKVOutputLayout,
    eps: float,
    norm_type: NormType,
    quantization_type: QuantizationType,
    fused_add: bool = False,
) -> TorchReference:
    """Bind one measured NAKB QKV configuration."""
    return TorchReference(
        qkv_tkg_torch_ref,
        parameters,
        bound_kwargs={
            "fused_add": fused_add,
            "d_head": d_head,
            "num_kv_heads": num_kv_heads,
            "num_q_heads": num_q_heads,
            "output_layout": output_layout,
            "eps": eps,
            "norm_type": norm_type,
            "quantization_type": quantization_type,
            "is_h_dim_4h_transposed": False,
            "output_in_sbuf": False,
            "hidden_actual": None,
            "transposed_in": False,
        },
    )


_torch_ref_0 = _bind_qkv(
    ("hidden", "qkv_w", "norm_w", "qkv_bias"),
    d_head=64,
    num_kv_heads=1,
    num_q_heads=8,
    output_layout=QKVOutputLayout.BSD,
    eps=1e-6,
    norm_type=NormType.RMS_NORM,
    quantization_type=QuantizationType.NONE,
)
_torch_ref_1 = _bind_qkv(
    ("hidden", "qkv_w"),
    d_head=128,
    num_kv_heads=1,
    num_q_heads=2,
    output_layout=QKVOutputLayout.NBSd,
    eps=1e-6,
    norm_type=NormType.NO_NORM,
    quantization_type=QuantizationType.NONE,
)
_torch_ref_2 = _bind_qkv(
    ("hidden", "qkv_w"),
    d_head=128,
    num_kv_heads=2,
    num_q_heads=2,
    output_layout=QKVOutputLayout.NBSd,
    eps=1e-6,
    norm_type=NormType.NO_NORM,
    quantization_type=QuantizationType.NONE,
)
_torch_ref_3 = _bind_qkv(
    ("hidden", "qkv_w", "norm_w"),
    d_head=128,
    num_kv_heads=8,
    num_q_heads=64,
    output_layout=QKVOutputLayout.BSD,
    eps=1e-6,
    norm_type=NormType.RMS_NORM,
    quantization_type=QuantizationType.NONE,
)
_torch_ref_4 = _bind_qkv(
    ("hidden", "qkv_w", "norm_w", "norm_bias"),
    d_head=128,
    num_kv_heads=1,
    num_q_heads=1,
    output_layout=QKVOutputLayout.BSD,
    eps=1e-6,
    norm_type=NormType.LAYER_NORM,
    quantization_type=QuantizationType.NONE,
)
_torch_ref_5 = _bind_qkv(
    ("hidden", "qkv_w", "norm_w", "qkv_w_scale"),
    d_head=128,
    num_kv_heads=1,
    num_q_heads=8,
    output_layout=QKVOutputLayout.BSD,
    eps=1e-6,
    norm_type=NormType.RMS_NORM,
    quantization_type=QuantizationType.ROW,
)
_torch_ref_6 = _bind_qkv(
    ("hidden", "qkv_w", "norm_w", "qkv_w_scale", "qkv_in_scale"),
    d_head=128,
    num_kv_heads=1,
    num_q_heads=8,
    output_layout=QKVOutputLayout.BSD,
    eps=1e-6,
    norm_type=NormType.RMS_NORM,
    quantization_type=QuantizationType.STATIC,
)
_torch_ref_7 = _bind_qkv(
    ("hidden", "qkv_w", "norm_w", "mlp_prev", "attn_prev"),
    d_head=128,
    num_kv_heads=1,
    num_q_heads=1,
    output_layout=QKVOutputLayout.BSD,
    eps=1e-6,
    norm_type=NormType.RMS_NORM,
    quantization_type=QuantizationType.NONE,
    fused_add=True,
)
_torch_ref_8 = _bind_qkv(
    ("hidden", "qkv_w", "mlp_prev", "attn_prev"),
    d_head=128,
    num_kv_heads=1,
    num_q_heads=1,
    output_layout=QKVOutputLayout.BSD,
    eps=1e-6,
    norm_type=NormType.NO_NORM,
    quantization_type=QuantizationType.NONE,
    fused_add=True,
)
_torch_ref_9 = _bind_qkv(
    ("hidden", "qkv_w", "norm_w", "mlp_prev", "attn_prev"),
    d_head=128,
    num_kv_heads=1,
    num_q_heads=2,
    output_layout=QKVOutputLayout.BSD,
    eps=77.0,
    norm_type=NormType.RMS_NORM,
    quantization_type=QuantizationType.NONE,
    fused_add=True,
)


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {
            "hidden": ((8, 5, 3072), "bfloat16"),
            "qkv_w": ((3072, 640), "bfloat16"),
            "norm_w": ((1, 3072), "bfloat16"),
            "qkv_bias": ((1, 640), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.044084931,
        "best_historical_latency_ms": 0.044084931,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {"hidden": ((1, 1, 16384), "bfloat16"), "qkv_w": ((16384, 512), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.053449917,
        "best_historical_latency_ms": 0.053449917,
    },
    {
        "torch_ref": _torch_ref_2,
        "input_specs": {"hidden": ((1, 1, 8192), "bfloat16"), "qkv_w": ((8192, 768), "bfloat16")},
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.049799089,
        "best_historical_latency_ms": 0.049799089,
    },
    {
        "torch_ref": _torch_ref_3,
        "input_specs": {
            "hidden": ((1, 1, 8192), "bfloat16"),
            "qkv_w": ((8192, 10240), "bfloat16"),
            "norm_w": ((1, 8192), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.348356955,
        "best_historical_latency_ms": 0.348356955,
    },
    {
        "torch_ref": _torch_ref_4,
        "input_specs": {
            "hidden": ((1, 5, 8192), "bfloat16"),
            "qkv_w": ((8192, 384), "bfloat16"),
            "norm_w": ((1, 8192), "bfloat16"),
            "norm_bias": ((1, 8192), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.035203278,
        "best_historical_latency_ms": 0.035203278,
    },
    {
        "torch_ref": _torch_ref_5,
        "input_specs": {
            "hidden": ((1, 5, 8192), "bfloat16"),
            "qkv_w": ((8192, 1280), "float8_e4m3"),
            "norm_w": ((1, 8192), "bfloat16"),
            "qkv_w_scale": ((128, 1280), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.053429083,
        "best_historical_latency_ms": 0.053429083,
    },
    {
        "torch_ref": _torch_ref_6,
        "input_specs": {
            "hidden": ((1, 5, 8192), "bfloat16"),
            "qkv_w": ((8192, 1280), "float8_e4m3"),
            "norm_w": ((1, 8192), "bfloat16"),
            "qkv_w_scale": ((128, 3), "float32"),
            "qkv_in_scale": ((128, 1), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.051194087,
        "best_historical_latency_ms": 0.051194087,
    },
    {
        "torch_ref": _torch_ref_7,
        "input_specs": {
            "hidden": ((1, 4, 16384), "bfloat16"),
            "qkv_w": ((16384, 384), "bfloat16"),
            "norm_w": ((1, 16384), "bfloat16"),
            "mlp_prev": ((1, 4, 16384), "bfloat16"),
            "attn_prev": ((1, 4, 16384), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.053172416,
        "best_historical_latency_ms": 0.053172416,
    },
    {
        "torch_ref": _torch_ref_8,
        "input_specs": {
            "hidden": ((4, 1, 32768), "bfloat16"),
            "qkv_w": ((32768, 384), "bfloat16"),
            "mlp_prev": ((4, 1, 32768), "bfloat16"),
            "attn_prev": ((4, 1, 32768), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.083299037,
        "best_historical_latency_ms": 0.083299037,
    },
    {
        "torch_ref": _torch_ref_9,
        "input_specs": {
            "hidden": ((1, 1, 8192), "bfloat16"),
            "qkv_w": ((8192, 512), "bfloat16"),
            "norm_w": ((1, 8192), "bfloat16"),
            "mlp_prev": ((1, 1, 8192), "bfloat16"),
            "attn_prev": ((1, 1, 8192), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.042026601,
        "best_historical_latency_ms": 0.042026601,
    },
)
