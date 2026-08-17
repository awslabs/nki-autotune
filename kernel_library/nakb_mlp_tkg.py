"""Measured NAKB ``mlp_tkg`` workload targets."""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import Enum

import ml_dtypes
import nki.language as nl
import numpy as np
import torch

from kernel_library import TorchReference
from nkigym.profile import InputSpecs

TKG_BS_SEQLEN_THRESHOLD = 96
FP8_E4M3_MAX = 240.0


class NormType(Enum):
    """Normalization modes copied from NAKB."""

    NO_NORM = 0
    RMS_NORM = 1
    LAYER_NORM = 2
    RMS_NORM_SKIP_GAMMA = 3


class ActFnType(Enum):
    """Activation modes copied from NAKB."""

    SiLU = 0
    GELU = 1
    GELU_Tanh_Approx = 2
    Swish = 3
    ReLU = 4


class QuantizationType(Enum):
    """Quantization modes copied from NAKB."""

    NONE = 0
    STATIC = 1
    ROW = 2
    MX = 3
    STATIC_MX = 4
    ROW_MX = 5


class ComputationMode(Enum):
    """Execution modes copied from NAKB."""

    AUTO = 0
    PREFILL = 1
    DECODE = 2


class LncSubscriptable:
    """Expose NAKB's ``reference[lnc](...)`` calling convention."""

    def __init__(self, function: Callable[..., dict[str, torch.Tensor]]) -> None:
        self._function = function
        self._lnc = 0

    def __getitem__(self, lnc: int) -> Callable[..., dict[str, torch.Tensor]]:
        """Return a callable bound to one LNC value."""

        def wrapper(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
            """Call the copied reference with the selected LNC."""
            self._lnc = lnc
            return self._function(*args, **kwargs)

        return wrapper

    def __call__(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        """Reject calls that omit the required LNC subscript."""
        del args, kwargs
        raise TypeError("mlp_torch_ref must be subscripted with an LNC value")

    @property
    def lnc(self) -> int:
        """Return the current LNC value."""
        return self._lnc


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


def _apply_activation(x: torch.Tensor, act_fn: ActFnType) -> torch.Tensor:
    """Apply the activation selected by NAKB."""
    if act_fn == ActFnType.SiLU:
        result = torch.nn.functional.silu(x)
    elif act_fn == ActFnType.GELU:
        result = torch.nn.functional.gelu(x)
    elif act_fn == ActFnType.GELU_Tanh_Approx:
        result = torch.nn.functional.gelu(x, approximate="tanh")
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")
    return result


def _apply_clamp(x: torch.Tensor, upper: float | None = None, lower: float | None = None) -> torch.Tensor:
    """Clamp a tensor to NAKB's optional projection bounds."""
    if upper is not None:
        x = torch.clamp(x, max=upper)
    if lower is not None:
        x = torch.clamp(x, min=lower)
    return x


def _scale_with_broadcast(
    tensor: torch.Tensor, scale: torch.Tensor, mode: ComputationMode = ComputationMode.AUTO
) -> torch.Tensor:
    """Multiply by a NAKB weight or activation scale."""
    if tensor.dim() == 3 and scale.dim() == 2:
        batch, sequence, intermediate = tensor.shape
        is_tkg = mode == ComputationMode.DECODE or (
            mode != ComputationMode.PREFILL and batch * sequence <= TKG_BS_SEQLEN_THRESHOLD
        )
        if is_tkg:
            result = tensor * scale[:sequence, :]
        else:
            tile_sequence = math.ceil(sequence / scale.shape[0])
            tile_intermediate = math.ceil(intermediate / scale.shape[1])
            tiled = scale.repeat(tile_sequence, tile_intermediate)[:sequence, :intermediate]
            result = tensor * tiled.unsqueeze(0)
    elif tensor.dim() == 3 and scale.dim() == 3:
        result = tensor * scale
    else:
        tile_0 = math.ceil(tensor.shape[0] / scale.shape[0])
        tile_1 = math.ceil(tensor.shape[1] / scale.shape[1])
        tiled = scale.repeat(tile_0, tile_1)[: tensor.shape[0], : tensor.shape[1]]
        result = tensor * tiled
    return result


def _row_quantize(x: torch.Tensor, clip_bound: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply NAKB's per-row FP8 scaling."""
    absolute_maximum = x.abs().max(dim=-1, keepdim=True).values
    if clip_bound > 0:
        absolute_maximum = absolute_maximum.clamp(max=clip_bound)
        x = x.clamp(-clip_bound, clip_bound)
    minimum_scale = torch.tensor(1e-5, dtype=x.dtype)
    quantization_scale = torch.max(absolute_maximum / FP8_E4M3_MAX, minimum_scale)
    return x / quantization_scale, quantization_scale


def _fp8_round_trip(tensor: torch.Tensor) -> torch.Tensor:
    """Apply the FP8 E4M3 round trip used by NAKB's golden."""
    return torch.from_numpy(tensor.numpy().astype(nl.float8_e4m3).astype(np.float32))


def _static_quantize(x: torch.Tensor, quantization_scale: torch.Tensor) -> torch.Tensor:
    """Apply NAKB's static FP8 scaling and clipping."""
    scaled = _scale_with_broadcast(x, 1.0 / quantization_scale)
    return torch.clamp(scaled, -FP8_E4M3_MAX, FP8_E4M3_MAX)


def _mlp_ref_standard(
    hidden: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    quantization_type: QuantizationType,
    gate_w_scale: torch.Tensor | None,
    up_w_scale: torch.Tensor | None,
    down_w_scale: torch.Tensor | None,
    gate_up_in_scale: torch.Tensor | None,
    down_in_scale: torch.Tensor | None,
    quant_clipping_bound: float,
    gate_proj_bias_tensor: torch.Tensor | None,
    up_proj_bias_tensor: torch.Tensor | None,
    down_proj_bias_tensor: torch.Tensor | None,
    skip_gate_proj: bool,
    activation_fn: ActFnType,
    gate_clamp_upper_limit: float | None,
    gate_clamp_lower_limit: float | None,
    up_clamp_upper_limit: float | None,
    up_clamp_lower_limit: float | None,
    mode: ComputationMode = ComputationMode.AUTO,
) -> torch.Tensor:
    """Run NAKB's standard NONE, ROW, or STATIC MLP projection path."""
    is_static_quantization = quantization_type == QuantizationType.STATIC
    batch, sequence = hidden.shape[0], hidden.shape[1]
    is_tkg = mode == ComputationMode.DECODE or (
        mode != ComputationMode.PREFILL and batch * sequence <= TKG_BS_SEQLEN_THRESHOLD
    )
    quantize_activations = not (quantization_type == QuantizationType.NONE or (is_static_quantization and is_tkg))
    gate_w_scale_t = gate_w_scale.to(torch.float32) if gate_w_scale is not None else None
    up_w_scale_t = up_w_scale.to(torch.float32) if up_w_scale is not None else None
    down_w_scale_t = down_w_scale.to(torch.float32) if down_w_scale is not None else None
    gate_up_in_scale_t = gate_up_in_scale.to(torch.float32) if gate_up_in_scale is not None else None
    down_in_scale_t = down_in_scale.to(torch.float32) if down_in_scale is not None else None
    if quantization_type == QuantizationType.ROW:
        projection_input, row_input_scale = _row_quantize(hidden, quant_clipping_bound)
    else:
        projection_input = hidden
        row_input_scale = None

    def project(
        projection_value: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor | None,
        input_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply one NAKB projection with its dequantization scales."""
        if weight_scale is None and input_scale is None:
            result = projection_value @ weight
        elif input_scale is None:
            if weight_scale is None:
                raise ValueError("weight scale is required")
            result = _scale_with_broadcast(projection_value @ weight, weight_scale, mode)
        elif quantization_type == QuantizationType.ROW:
            if weight_scale is None:
                raise ValueError("weight scale is required")
            result = _scale_with_broadcast(
                _scale_with_broadcast(projection_value @ weight, weight_scale, mode), input_scale, mode
            )
        else:
            if weight_scale is None:
                raise ValueError("weight scale is required")
            result = _scale_with_broadcast(projection_value @ weight, weight_scale * input_scale, mode)
        return result

    if not quantize_activations:
        gate_up_input_scale = None
    elif quantization_type == QuantizationType.ROW:
        gate_up_input_scale = row_input_scale
    elif is_static_quantization:
        gate_up_input_scale = gate_up_in_scale_t
    else:
        gate_up_input_scale = None

    if not skip_gate_proj:
        gate_out = project(projection_input, gate_w, gate_w_scale_t, gate_up_input_scale)
        if gate_proj_bias_tensor is not None:
            gate_out = gate_out + gate_proj_bias_tensor.to(torch.float32)
        gate_out = _apply_clamp(gate_out, gate_clamp_upper_limit, gate_clamp_lower_limit)
        up_out = project(projection_input, up_w, up_w_scale_t, gate_up_input_scale)
        if up_proj_bias_tensor is not None:
            up_out = up_out + up_proj_bias_tensor.to(torch.float32)
        up_out = _apply_clamp(up_out, up_clamp_upper_limit, up_clamp_lower_limit)
        intermediate = _apply_activation(gate_out, activation_fn) * up_out
    else:
        up_out = project(projection_input, up_w, up_w_scale_t, gate_up_input_scale)
        if up_proj_bias_tensor is not None:
            up_out = up_out + up_proj_bias_tensor.to(torch.float32)
        up_out = _apply_clamp(up_out, up_clamp_upper_limit, up_clamp_lower_limit)
        intermediate = _apply_activation(up_out, activation_fn)

    if not quantize_activations:
        output = project(intermediate, down_w, down_w_scale_t, None)
    elif quantization_type == QuantizationType.ROW:
        quantized_intermediate, intermediate_scale = _row_quantize(intermediate, quant_clipping_bound)
        quantized_intermediate = _fp8_round_trip(quantized_intermediate)
        output = project(quantized_intermediate, down_w, down_w_scale_t, intermediate_scale)
    else:
        if down_in_scale_t is None:
            raise ValueError("down_in_scale required for static activation quantization")
        quantized_intermediate = _fp8_round_trip(_static_quantize(intermediate, down_in_scale_t))
        output = project(quantized_intermediate, down_w, down_w_scale_t, down_in_scale_t)
    if down_proj_bias_tensor is not None:
        output = output + down_proj_bias_tensor.to(torch.float32)
    return output


def _mlp_torch_ref_impl(
    hidden_tensor: torch.Tensor,
    gate_proj_weights_tensor: torch.Tensor,
    up_proj_weights_tensor: torch.Tensor,
    down_proj_weights_tensor: torch.Tensor,
    normalization_weights_tensor: torch.Tensor | None = None,
    gate_proj_bias_tensor: torch.Tensor | None = None,
    up_proj_bias_tensor: torch.Tensor | None = None,
    down_proj_bias_tensor: torch.Tensor | None = None,
    normalization_bias_tensor: torch.Tensor | None = None,
    fused_add_tensor: torch.Tensor | None = None,
    store_fused_add_result: bool = False,
    activation_fn: ActFnType = ActFnType.SiLU,
    normalization_type: NormType = NormType.NO_NORM,
    quantization_type: QuantizationType = QuantizationType.NONE,
    gate_w_scale: torch.Tensor | None = None,
    up_w_scale: torch.Tensor | None = None,
    down_w_scale: torch.Tensor | None = None,
    gate_up_in_scale: torch.Tensor | None = None,
    down_in_scale: torch.Tensor | None = None,
    quant_clipping_bound: float = 0.0,
    output_dtype: object | None = None,
    store_output_in_sbuf: bool = False,
    eps: float = 1e-6,
    skip_gate_proj: bool = False,
    use_tkg_gate_up_proj_column_tiling: bool = True,
    use_tkg_down_proj_column_tiling: bool = True,
    use_tkg_down_proj_optimized_layout: bool = False,
    gate_clamp_upper_limit: float | None = None,
    gate_clamp_lower_limit: float | None = None,
    up_clamp_upper_limit: float | None = None,
    up_clamp_lower_limit: float | None = None,
    force_cte_mode: bool = False,
    mode: ComputationMode = ComputationMode.AUTO,
    sbm: object | None = None,
    transposed_in: bool = False,
    transposed_out: bool = False,
    **kwargs: object,
) -> dict[str, torch.Tensor]:
    """PyTorch reference implementation copied from NAKB's ``mlp_tkg``."""
    del output_dtype, store_output_in_sbuf, force_cte_mode, sbm, transposed_out, kwargs
    del use_tkg_gate_up_proj_column_tiling, use_tkg_down_proj_column_tiling
    if normalization_type in (NormType.RMS_NORM, NormType.LAYER_NORM):
        if normalization_weights_tensor is None:
            raise ValueError(f"normalization_weights_tensor required when normalization_type is {normalization_type}")
    if quantization_type in (QuantizationType.ROW, QuantizationType.STATIC):
        if gate_w_scale is None:
            raise ValueError(f"gate_w_scale required for {quantization_type} quantization")
        if up_w_scale is None:
            raise ValueError(f"up_w_scale required for {quantization_type} quantization")
        if down_w_scale is None:
            raise ValueError(f"down_w_scale required for {quantization_type} quantization")
    if quantization_type not in (QuantizationType.NONE, QuantizationType.ROW, QuantizationType.STATIC):
        raise ValueError(f"unsupported copied mlp_tkg configuration: {quantization_type}")
    if transposed_in:
        h0, n_prgs, h1_shard, batch_sequence = hidden_tensor.shape
        hidden = (
            hidden_tensor.to(torch.float32)
            .permute(3, 1, 0, 2)
            .reshape(batch_sequence, n_prgs * h0 * h1_shard)
            .unsqueeze(0)
        )
    else:
        hidden = hidden_tensor.to(torch.float32)
    gate_w = gate_proj_weights_tensor.to(torch.float32)
    up_w = up_proj_weights_tensor.to(torch.float32)
    down_w = down_proj_weights_tensor.to(torch.float32)
    gamma = normalization_weights_tensor.to(torch.float32) if normalization_weights_tensor is not None else None
    norm_bias = normalization_bias_tensor.to(torch.float32) if normalization_bias_tensor is not None else None
    if use_tkg_down_proj_optimized_layout:
        intermediate, hidden_dim = down_w.shape
        lnc = mlp_torch_ref.lnc
        down_w = (
            down_w.reshape((intermediate, lnc, hidden_dim // 128 // lnc, 128))
            .permute(0, 1, 3, 2)
            .reshape((intermediate, hidden_dim))
        )
    add_out = None
    if fused_add_tensor is not None:
        hidden = hidden + fused_add_tensor.to(torch.float32)
        if store_fused_add_result:
            add_out = hidden.clone()
    normalization_gamma = None if normalization_type == NormType.RMS_NORM_SKIP_GAMMA else gamma
    hidden = NORM_NAME_TO_TORCH_REF[normalization_type](hidden, normalization_gamma, eps=eps, norm_b=norm_bias)
    output = _mlp_ref_standard(
        hidden=hidden,
        gate_w=gate_w,
        up_w=up_w,
        down_w=down_w,
        quantization_type=quantization_type,
        gate_w_scale=gate_w_scale,
        up_w_scale=up_w_scale,
        down_w_scale=down_w_scale,
        gate_up_in_scale=gate_up_in_scale,
        down_in_scale=down_in_scale,
        quant_clipping_bound=quant_clipping_bound,
        gate_proj_bias_tensor=gate_proj_bias_tensor,
        up_proj_bias_tensor=up_proj_bias_tensor,
        down_proj_bias_tensor=down_proj_bias_tensor,
        skip_gate_proj=skip_gate_proj,
        activation_fn=activation_fn,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        mode=mode,
    )
    result = {"out": output}
    if fused_add_tensor is not None and store_fused_add_result:
        if add_out is None:
            raise RuntimeError("fused add output was not retained")
        result["add_out"] = add_out
    return result


mlp_torch_ref = LncSubscriptable(_mlp_torch_ref_impl)


def _input_generator(input_specs: InputSpecs, seed: int) -> dict[str, np.ndarray]:
    """Generate deterministic arrays matching one NAKB workload contract."""
    rng = np.random.default_rng(seed)
    special_dtypes = {"bfloat16": np.dtype(ml_dtypes.bfloat16), "float8_e4m3": np.dtype(ml_dtypes.float8_e4m3)}
    inputs: dict[str, np.ndarray] = {}
    for name, (shape, dtype_name) in input_specs.items():
        dtype = special_dtypes.get(dtype_name)
        if dtype is None:
            dtype = np.dtype(dtype_name)
        if name == "normalization_weights_tensor" or name.endswith("_scale"):
            values = np.ones(shape, dtype=np.float32)
        else:
            values = rng.standard_normal(shape, dtype=np.float32) * 0.1
        inputs[name] = values.astype(dtype)
    return inputs


def _bind_mlp(
    parameters: tuple[str, ...],
    *,
    lnc: int,
    normalization_type: NormType,
    quantization_type: QuantizationType,
    skip_gate_proj: bool,
) -> TorchReference:
    """Bind one measured NAKB MLP configuration."""
    return TorchReference(
        mlp_torch_ref,
        parameters,
        subscript=lnc,
        bound_kwargs={
            "gate_proj_bias_tensor": None,
            "up_proj_bias_tensor": None,
            "down_proj_bias_tensor": None,
            "normalization_bias_tensor": None,
            "fused_add_tensor": None,
            "store_fused_add_result": False,
            "activation_fn": ActFnType.SiLU,
            "normalization_type": normalization_type,
            "quantization_type": quantization_type,
            "quant_clipping_bound": 0.0,
            "output_dtype": nl.bfloat16,
            "store_output_in_sbuf": False,
            "eps": 1e-6,
            "skip_gate_proj": skip_gate_proj,
            "use_tkg_gate_up_proj_column_tiling": True,
            "use_tkg_down_proj_column_tiling": True,
            "use_tkg_down_proj_optimized_layout": False,
            "gate_clamp_upper_limit": None,
            "gate_clamp_lower_limit": None,
            "up_clamp_upper_limit": None,
            "up_clamp_lower_limit": None,
            "force_cte_mode": False,
            "mode": ComputationMode.DECODE,
            "sbm": None,
            "transposed_in": False,
            "transposed_out": False,
        },
    )


_torch_ref_0 = _bind_mlp(
    (
        "hidden_tensor",
        "gate_proj_weights_tensor",
        "up_proj_weights_tensor",
        "down_proj_weights_tensor",
        "normalization_weights_tensor",
    ),
    lnc=2,
    normalization_type=NormType.LAYER_NORM,
    quantization_type=QuantizationType.NONE,
    skip_gate_proj=False,
)
_torch_ref_1 = _bind_mlp(
    (
        "hidden_tensor",
        "gate_proj_weights_tensor",
        "up_proj_weights_tensor",
        "down_proj_weights_tensor",
        "normalization_weights_tensor",
        "gate_w_scale",
        "up_w_scale",
        "down_w_scale",
        "gate_up_in_scale",
        "down_in_scale",
    ),
    lnc=2,
    normalization_type=NormType.RMS_NORM,
    quantization_type=QuantizationType.STATIC,
    skip_gate_proj=False,
)
_torch_ref_2 = _bind_mlp(
    (
        "hidden_tensor",
        "gate_proj_weights_tensor",
        "up_proj_weights_tensor",
        "down_proj_weights_tensor",
        "normalization_weights_tensor",
        "gate_w_scale",
        "up_w_scale",
        "down_w_scale",
    ),
    lnc=2,
    normalization_type=NormType.RMS_NORM,
    quantization_type=QuantizationType.ROW,
    skip_gate_proj=False,
)
_torch_ref_3 = _bind_mlp(
    ("hidden_tensor", "gate_proj_weights_tensor", "up_proj_weights_tensor", "down_proj_weights_tensor"),
    lnc=2,
    normalization_type=NormType.NO_NORM,
    quantization_type=QuantizationType.NONE,
    skip_gate_proj=True,
)
_torch_ref_4 = _bind_mlp(
    (
        "hidden_tensor",
        "gate_proj_weights_tensor",
        "up_proj_weights_tensor",
        "down_proj_weights_tensor",
        "gate_w_scale",
        "up_w_scale",
        "down_w_scale",
    ),
    lnc=2,
    normalization_type=NormType.NO_NORM,
    quantization_type=QuantizationType.ROW,
    skip_gate_proj=False,
)
_torch_ref_5 = _bind_mlp(
    (
        "hidden_tensor",
        "gate_proj_weights_tensor",
        "up_proj_weights_tensor",
        "down_proj_weights_tensor",
        "gate_w_scale",
        "up_w_scale",
        "down_w_scale",
        "gate_up_in_scale",
        "down_in_scale",
    ),
    lnc=2,
    normalization_type=NormType.NO_NORM,
    quantization_type=QuantizationType.STATIC,
    skip_gate_proj=False,
)
_torch_ref_0_lnc1 = _bind_mlp(
    (
        "hidden_tensor",
        "gate_proj_weights_tensor",
        "up_proj_weights_tensor",
        "down_proj_weights_tensor",
        "normalization_weights_tensor",
    ),
    lnc=1,
    normalization_type=NormType.LAYER_NORM,
    quantization_type=QuantizationType.NONE,
    skip_gate_proj=False,
)


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {
            "hidden_tensor": ((2, 4, 8448), "bfloat16"),
            "gate_proj_weights_tensor": ((8448, 1408), "bfloat16"),
            "up_proj_weights_tensor": ((8448, 1408), "bfloat16"),
            "down_proj_weights_tensor": ((1408, 8448), "bfloat16"),
            "normalization_weights_tensor": ((1, 8448), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.143633942,
        "best_historical_latency_ms": 0.143633942,
    },
    {
        "torch_ref": _torch_ref_0_lnc1,
        "input_specs": {
            "hidden_tensor": ((3, 8, 8192), "bfloat16"),
            "gate_proj_weights_tensor": ((8192, 832), "bfloat16"),
            "up_proj_weights_tensor": ((8192, 832), "bfloat16"),
            "down_proj_weights_tensor": ((832, 8192), "bfloat16"),
            "normalization_weights_tensor": ((1, 8192), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.124893971,
        "best_historical_latency_ms": 0.124893971,
    },
    {
        "torch_ref": _torch_ref_1,
        "input_specs": {
            "hidden_tensor": ((256, 1, 8192), "bfloat16"),
            "gate_proj_weights_tensor": ((8192, 3584), "float8_e4m3"),
            "up_proj_weights_tensor": ((8192, 3584), "float8_e4m3"),
            "down_proj_weights_tensor": ((3584, 8192), "float8_e4m3"),
            "normalization_weights_tensor": ((1, 8192), "bfloat16"),
            "gate_w_scale": ((128, 1), "float32"),
            "up_w_scale": ((128, 1), "float32"),
            "down_w_scale": ((128, 1), "float32"),
            "gate_up_in_scale": ((128, 1), "float32"),
            "down_in_scale": ((128, 1), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.313052011,
        "best_historical_latency_ms": 0.313052011,
    },
    {
        "torch_ref": _torch_ref_2,
        "input_specs": {
            "hidden_tensor": ((4, 5, 8192), "bfloat16"),
            "gate_proj_weights_tensor": ((8192, 512), "float8_e4m3"),
            "up_proj_weights_tensor": ((8192, 512), "float8_e4m3"),
            "down_proj_weights_tensor": ((512, 8192), "float8_e4m3"),
            "normalization_weights_tensor": ((1, 8192), "bfloat16"),
            "gate_w_scale": ((128, 512), "float32"),
            "up_w_scale": ((128, 512), "float32"),
            "down_w_scale": ((128, 8192), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.064311566,
        "best_historical_latency_ms": 0.064311566,
    },
    {
        "torch_ref": _torch_ref_3,
        "input_specs": {
            "hidden_tensor": ((4, 1, 16384), "bfloat16"),
            "gate_proj_weights_tensor": ((16384, 832), "bfloat16"),
            "up_proj_weights_tensor": ((16384, 832), "bfloat16"),
            "down_proj_weights_tensor": ((832, 16384), "bfloat16"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.10905233,
        "best_historical_latency_ms": 0.10905233,
    },
    {
        "torch_ref": _torch_ref_4,
        "input_specs": {
            "hidden_tensor": ((1, 1, 16384), "bfloat16"),
            "gate_proj_weights_tensor": ((16384, 896), "float8_e4m3"),
            "up_proj_weights_tensor": ((16384, 896), "float8_e4m3"),
            "down_proj_weights_tensor": ((896, 16384), "float8_e4m3"),
            "gate_w_scale": ((128, 896), "float32"),
            "up_w_scale": ((128, 896), "float32"),
            "down_w_scale": ((128, 16384), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.106305668,
        "best_historical_latency_ms": 0.106305668,
    },
    {
        "torch_ref": _torch_ref_5,
        "input_specs": {
            "hidden_tensor": ((1, 1, 16384), "bfloat16"),
            "gate_proj_weights_tensor": ((16384, 896), "float8_e4m3"),
            "up_proj_weights_tensor": ((16384, 896), "float8_e4m3"),
            "down_proj_weights_tensor": ((896, 16384), "float8_e4m3"),
            "gate_w_scale": ((128, 1), "float32"),
            "up_w_scale": ((128, 1), "float32"),
            "down_w_scale": ((128, 1), "float32"),
            "gate_up_in_scale": ((128, 1), "float32"),
            "down_in_scale": ((128, 1), "float32"),
        },
        "input_generator": _input_generator,
        "atol": 0.05,
        "rtol": 0.05,
        "nakb_latency_ms": 0.101869841,
        "best_historical_latency_ms": 0.101869841,
    },
)
