"""Measured NAKB ``attention_tkg`` workload target."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import ml_dtypes
import numpy as np
import torch

from kernel_library import TorchReference
from nkigym.profile import InputSpecs


@dataclass
class AttnTKGConfig:
    """Configuration copied from NAKB's token-generation attention kernel."""

    bs: int = 0
    q_head: int = 0
    s_active: int = 0
    curr_sprior: int = 0
    full_sprior: int = 0
    d_head: int = 0
    block_len: int = 0
    tp_k_prior: bool = False
    strided_mm1: bool = True
    use_pos_id: bool = False
    fuse_rope: bool = False
    use_gpsimd_sb2sb: bool = True
    qk_in_sb: bool = False
    k_out_in_sb: bool = False
    out_in_sb: bool = False
    enable_fa_s_prior_tiling: bool = True


class LncSubscriptable:
    """Apply NAKB's LNC subscript before calling a torch reference."""

    def __init__(self, function: Callable[..., object]) -> None:
        """Store the copied reference implementation."""
        self._function = function
        self._lnc = 0

    def __getitem__(self, lnc: int) -> Callable[..., object]:
        """Return a callable configured for one logical Neuron core count."""

        def wrapper(*args: object, **kwargs: object) -> object:
            """Call the copied reference with the selected LNC."""
            self._lnc = lnc
            return self._function(*args, **kwargs)

        return wrapper

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Reject calls that omit the required LNC subscript."""
        del args, kwargs
        raise TypeError("attention_tkg_torch_ref must be subscripted with an LNC value")

    @property
    def lnc(self) -> int:
        """Return the configured logical Neuron core count."""
        return self._lnc


def kernel_assert(condition: bool, error_text: str) -> None:
    """Raise NAKB's kernel validation assertion."""
    assert condition, (  # noqa: S101
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text} - "
        "Please check the validation message and adjust kernel inputs accordingly"
    )


def _reshape_q_and_k_active(
    q: torch.Tensor, k_active: torch.Tensor, cfg: AttnTKGConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply NAKB's active Q/K layout conversion."""
    d_head, batch, q_head, s_active = cfg.d_head, cfg.bs, cfg.q_head, cfg.s_active
    if cfg.qk_in_sb:
        q = q.reshape(d_head, batch, q_head, s_active).permute((1, 2, 0, 3))
        k_active = k_active.reshape(d_head, 1, batch, s_active).permute((2, 1, 0, 3))
    else:
        q = q.permute(0, 1, 3, 2)
        k_active = k_active.permute(0, 1, 3, 2)
    return q, k_active


def _slice_and_reshape_kv_prior(
    k_prior: torch.Tensor, v_prior: torch.Tensor, cfg: AttnTKGConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply NAKB's prior-cache slicing and layout conversion."""
    batch, s_prior = cfg.bs, cfg.curr_sprior
    k_prior = k_prior[:batch, ...]
    v_prior = v_prior[:batch, ...]
    if cfg.tp_k_prior:
        k_prior = k_prior[..., :s_prior, :].permute(0, 1, 3, 2)
    else:
        k_prior = k_prior[..., :s_prior]
    v_prior = v_prior[..., :s_prior, :]
    return k_prior, v_prior


def _attention_tkg_fwd_ref(
    q: torch.Tensor,
    k_active: torch.Tensor,
    v_active: torch.Tensor,
    k_prior: torch.Tensor,
    v_prior: torch.Tensor,
    active_mask: torch.Tensor,
    cfg: AttnTKGConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the copied NAKB forward path used by the selected workload."""
    s_active = cfg.s_active
    k_prior = k_prior.clone()
    v_prior = v_prior.clone()
    k_prior[..., -s_active:] = k_active
    v_prior[..., -s_active:, :] = v_active
    score = k_prior.permute(0, 1, 3, 2) @ q
    score[active_mask == 0] = -torch.inf
    score_max = torch.max(score, dim=2, keepdim=True).values
    score -= score_max
    score = torch.exp(score)
    score_sum = torch.sum(score, dim=2, keepdim=True)
    score = score / score_sum
    out = score.permute(0, 1, 3, 2) @ v_prior
    out = out.permute(0, 1, 3, 2)
    return out, k_active


def _attention_tkg_torch_ref_impl(
    q: torch.Tensor,
    k_active: torch.Tensor,
    v_active: torch.Tensor,
    k_prior: torch.Tensor,
    v_prior: torch.Tensor,
    mask: torch.Tensor,
    out: torch.Tensor,
    cfg: AttnTKGConfig,
    sbm: object | None,
    inv_freqs: torch.Tensor | None = None,
    rope_pos_ids: torch.Tensor | None = None,
    start_pos_ids: torch.Tensor | None = None,
    sink: torch.Tensor | None = None,
    active_blocks_table: torch.Tensor | None = None,
    k_out: torch.Tensor | None = None,
    DBG_TENSORS: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """PyTorch reference for the selected NAKB attention_tkg path."""
    kernel_assert(attention_tkg_torch_ref.lnc == 1, "the selected NAKB workload uses LNC1")
    kernel_assert(cfg.block_len == 0, "the selected NAKB workload uses flat KV cache")
    kernel_assert(not cfg.use_pos_id and not cfg.fuse_rope, "the selected NAKB workload uses a supplied mask")
    kernel_assert(
        sink is None and active_blocks_table is None and DBG_TENSORS is None,
        "the selected NAKB workload has no sink, block table, or debug tensors",
    )
    q = q.to(torch.float32)
    k_active = k_active.to(torch.float32)
    v_active = v_active.to(torch.float32)
    k_prior = k_prior.to(torch.float32)
    v_prior = v_prior.to(torch.float32)
    active_mask = mask.to(torch.uint8).permute(1, 2, 0, 3)
    q, k_active = _reshape_q_and_k_active(q, k_active, cfg)
    k_prior, v_prior = _slice_and_reshape_kv_prior(k_prior, v_prior, cfg)
    attn_out, attn_k_out = _attention_tkg_fwd_ref(
        q=q, k_active=k_active, v_active=v_active, k_prior=k_prior, v_prior=v_prior, active_mask=active_mask, cfg=cfg
    )
    kernel_assert(
        out.shape == attn_out.shape, f"Output shape mismatch: out.shape={out.shape}, attn_out.shape={attn_out.shape}"
    )
    out.copy_(attn_out)
    if k_out is not None:
        kernel_assert(
            k_out.shape == attn_k_out.shape,
            f"Output shape mismatch: k_out.shape={k_out.shape}, attn_k_out.shape={attn_k_out.shape}",
        )
        k_out.copy_(attn_k_out)
    return out, k_out


attention_tkg_torch_ref = LncSubscriptable(_attention_tkg_torch_ref_impl)

_CONFIG = AttnTKGConfig(
    bs=4,
    q_head=1,
    s_active=5,
    curr_sprior=8192,
    full_sprior=8192,
    d_head=128,
    block_len=0,
    tp_k_prior=False,
    strided_mm1=False,
    use_pos_id=False,
    fuse_rope=False,
    qk_in_sb=True,
    k_out_in_sb=False,
    out_in_sb=False,
)


def _prepare_reference_arguments(arguments: dict[str, object]) -> dict[str, object]:
    """Allocate NAKB's output placeholder for the copied torch reference."""
    prepared = dict(arguments)
    prepared["out"] = torch.zeros((4, 1, 128, 5), dtype=torch.float32)
    return prepared


def _input_generator(input_specs: InputSpecs, seed: int) -> dict[str, np.ndarray]:
    """Generate the selected NAKB FP8 attention inputs and cache mask."""
    rng = np.random.default_rng(seed)
    fp8_dtype = np.dtype(ml_dtypes.float8_e4m3)
    inputs: dict[str, np.ndarray] = {}
    for name, (shape, dtype_name) in input_specs.items():
        if name == "mask":
            continue
        values = rng.random(shape, dtype=np.float32) * 480.0 - 240.0
        inputs[name] = values.astype(fp8_dtype)
    maximum_position = _CONFIG.curr_sprior - _CONFIG.s_active
    cache_lengths = np.round(rng.normal(maximum_position * 0.5, maximum_position * 0.1, size=(_CONFIG.bs, 1))).astype(
        int
    )
    cache_lengths = np.clip(cache_lengths, 1, maximum_position)
    key_positions = np.arange(_CONFIG.curr_sprior).reshape(-1, 1, 1, 1)
    prior_mask = key_positions < cache_lengths.reshape(1, _CONFIG.bs, 1, 1)
    mask = np.broadcast_to(prior_mask, (_CONFIG.curr_sprior, _CONFIG.bs, _CONFIG.q_head, _CONFIG.s_active)).copy()
    active_mask = np.tril(np.ones((_CONFIG.s_active, _CONFIG.s_active), dtype=np.bool_))
    mask[-_CONFIG.s_active :, :, :, :] = active_mask.T.reshape(_CONFIG.s_active, 1, 1, _CONFIG.s_active)
    inputs["mask"] = mask.astype(np.uint8)
    return inputs


_torch_ref_0 = TorchReference(
    attention_tkg_torch_ref,
    ("q", "k_active", "v_active", "k_prior", "v_prior", "mask"),
    bound_kwargs={
        "cfg": _CONFIG,
        "sbm": None,
        "inv_freqs": None,
        "rope_pos_ids": None,
        "start_pos_ids": None,
        "sink": None,
        "active_blocks_table": None,
        "k_out": None,
        "DBG_TENSORS": None,
    },
    subscript=1,
    argument_adapter=_prepare_reference_arguments,
)


WORKLOADS = (
    {
        "torch_ref": _torch_ref_0,
        "input_specs": {
            "q": ((128, 20), "float8_e4m3"),
            "k_active": ((128, 20), "float8_e4m3"),
            "v_active": ((4, 1, 5, 128), "float8_e4m3"),
            "k_prior": ((4, 1, 128, 8192), "float8_e4m3"),
            "v_prior": ((4, 1, 8192, 128), "float8_e4m3"),
            "mask": ((8192, 4, 1, 5), "uint8"),
        },
        "input_generator": _input_generator,
        "atol": 0.03,
        "rtol": 0.03,
        "nakb_latency_ms": 0.05941574,
        "best_historical_latency_ms": 0.05941574,
    },
)
