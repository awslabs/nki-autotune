"""Tests for broadcast pointwise fusion into activation bias."""

from __future__ import annotations

from examples.online_fusion_attention import f_nkigym
from nkigym.codegen import render
from nkigym.ir import KernelIR, build_initial_ir
from nkigym.ir.tree import ISANode
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.load import NKILoad
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.transforms import FuseBroadcastActivation, FusePointwiseReduction, OnlineFusion


def _attention_before_bias_fusion() -> KernelIR:
    """Build online attention after native pointwise-reduction fusion."""
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
    pointwise_reduction = FusePointwiseReduction()
    while options := pointwise_reduction.analyze(ir):
        ir = pointwise_reduction.apply(ir, options[0])
    return ir


@nkigym_kernel
def f_scaled_broadcast_activation(data, bias):
    """Apply a scaled activation to a broadcast-shifted matrix."""
    sbuf_data = NKILoad()(src=data)
    sbuf_bias = NKILoad()(src=bias)
    shifted = NKITensorScalar(op0="add")(data=sbuf_data, operand0=sbuf_bias)
    reduced = NKIActivationReduce(op="exp", reduce_op="add", scale=2.0)(data=shifted)
    output = NKIStore()(src=reduced)
    return output


def test_fuse_broadcast_activation_materializes_only_row_negation() -> None:
    """A matrix subtraction becomes one row negation and an activation bias."""
    ir = _attention_before_bias_fusion()
    transform = FuseBroadcastActivation()
    options = transform.analyze(ir)
    assert len(options) == 1
    transformed = transform.apply(ir, options[0])
    source = render(transformed)
    assert "sbuf_centered" not in source
    assert "sbuf_row_max_online_current_negative" in source
    assert "op=nl.copy, scale=-1.0" in source
    assert "bias=sbuf_row_max_online_current_negative" in source

    leaves = [
        transformed.tree.isa(nid)
        for nid in transformed.tree.preorder()
        if isinstance(transformed.tree.data(nid), ISANode)
    ]
    activation_reduce = next(leaf for leaf in leaves if leaf.op_cls.NAME == "activation_reduce")
    assert activation_reduce.operand_bindings["data"].tensor == "sbuf_scaled_scores"
    assert activation_reduce.operand_bindings["bias"].tensor == "sbuf_row_max_online_current_negative"
    negation = transformed.tree.block(options[0].pointwise_block_nid)
    assert negation.axis_map == {"P": "d1"}
    assert tuple(iter_var.axis for iter_var in negation.iter_vars) == ("d1",)
    assert len(negation.iter_values) == 1
    assert transform.analyze(transformed) == []


def test_fuse_broadcast_activation_rejects_nonunit_activation_scale() -> None:
    """Moving an unscaled broadcast behind a scaled activation is not legal."""
    specs = {"data": ((128, 128), "bfloat16"), "bias": ((128,), "bfloat16")}
    ir = build_initial_ir(f_scaled_broadcast_activation, specs)
    assert FuseBroadcastActivation().analyze(ir) == []
