"""Tests for affine pointwise fusion and common-subexpression elimination."""

from __future__ import annotations

from examples.online_fusion_attention import f_nkigym
from nkigym.codegen import render
from nkigym.ir import KernelIR, build_initial_ir
from nkigym.transforms import (
    CommonSubexpressionElimination,
    FuseBroadcastActivation,
    FusePointwiseActivation,
    FusePointwiseReduction,
    OnlineFusion,
)


def _fuse_online_corrections() -> tuple[KernelIR, int]:
    """Build attention and fuse both correction chains into activations."""
    sequence_length = 512
    specs = {
        "query": ((128, sequence_length), "bfloat16"),
        "key": ((128, sequence_length), "bfloat16"),
        "value": ((sequence_length, 128), "bfloat16"),
    }
    ir = build_initial_ir(f_nkigym, specs)
    online = OnlineFusion()
    for _stage in range(2):
        option = next(candidate for candidate in online.analyze(ir) if candidate.chunk_size == 256)
        ir = online.apply(ir, option)

    reduction = FusePointwiseReduction()
    while options := reduction.analyze(ir):
        ir = reduction.apply(ir, options[0])
    broadcast = FuseBroadcastActivation()
    while options := broadcast.analyze(ir):
        ir = broadcast.apply(ir, options[0])

    transform = FusePointwiseActivation()
    option_count = len(transform.analyze(ir))
    while options := transform.analyze(ir):
        ir = transform.apply(ir, options[0])
    return ir, option_count


def test_online_corrections_use_native_activation_bias() -> None:
    """Each ``exp(-(current-old))`` chain becomes one activation call."""
    ir, option_count = _fuse_online_corrections()
    source = render(ir)
    assert option_count == 2
    assert "correction_difference" not in source
    assert source.count("bias=sbuf_row_max[") == 2
    assert source.count("op=nl.exp, scale=-1.0") == 2
    assert FusePointwiseActivation().analyze(ir) == []


def test_common_subexpression_elimination_reuses_online_correction() -> None:
    """Identical correction activations share one result."""
    ir, _option_count = _fuse_online_corrections()
    elimination = CommonSubexpressionElimination()
    options = elimination.analyze(ir)
    assert len(options) == 1
    ir = elimination.apply(ir, options[0])
    source = render(ir)
    assert source.count("bias=sbuf_row_max[") == 1
    assert "online_stage2_correction" not in source
    assert elimination.analyze(ir) == []
