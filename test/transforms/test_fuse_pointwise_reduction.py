"""Tests for contract-driven pointwise-reduction fusion."""

from __future__ import annotations

import numpy as np

from examples.online_fusion_attention import f_nkigym
from nkigym.ir import KernelIR, build_initial_ir
from nkigym.ir.tree import ISANode
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.load import NKILoad
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.transforms import FusePointwiseReduction, OnlineFusion


def _online_attention() -> KernelIR:
    """Build one online attention state with both supported fusion patterns."""
    sequence_length = 512
    input_specs = {
        "query": ((128, sequence_length), "bfloat16"),
        "key": ((128, sequence_length), "bfloat16"),
        "value": ((sequence_length, 128), "bfloat16"),
    }
    ir = build_initial_ir(f_nkigym, input_specs)
    transform = OnlineFusion()
    for _stage in range(2):
        option = next(option for option in transform.analyze(ir) if option.chunk_size == 256)
        ir = transform.apply(ir, option)
    return ir


@nkigym_kernel
def f_biased_activation_reduce(data, bias):
    """Apply a tensor-biased activation before a free-axis reduction."""
    sbuf_data = NKILoad()(src=data)
    sbuf_bias = NKILoad()(src=bias)
    activated = NKIActivation(op="exp")(data=sbuf_data, bias=sbuf_bias)
    reduced = NKITensorReduce(op="add", axis=1)(data=activated)
    output = NKIStore()(src=reduced)
    return output


def test_fuse_pointwise_reduction_contract() -> None:
    """Both contract-compatible map-reduce pairs become dual-output native instructions."""
    ir = _online_attention()
    transform = FusePointwiseReduction()
    options = transform.analyze(ir)
    assert len(options) == 2

    while options := transform.analyze(ir):
        ir = transform.apply(ir, options[0])

    leaves = [ir.tree.isa(nid) for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ISANode)]
    fused = [leaf for leaf in leaves if leaf.op_cls.NAME in {"tensor_scalar_reduce", "activation_reduce"}]
    assert [leaf.op_cls.NAME for leaf in fused] == ["tensor_scalar_reduce", "activation_reduce"]
    assert all({"dst", "reduce_res"} <= set(leaf.operand_bindings) for leaf in fused)
    assert not any(leaf.op_cls.NAME == "tensor_reduce" for leaf in leaves)
    assert fused[0].operand_bindings["dst"].tensor == "sbuf_scaled_scores"
    assert fused[0].operand_bindings["reduce_res"].tensor == "sbuf_row_max_online_chunk"
    assert fused[1].operand_bindings["dst"].tensor == "sbuf_exp"
    assert fused[1].operand_bindings["reduce_res"].tensor == "sbuf_row_sum_online_chunk"
    assert transform.analyze(ir) == []


def test_activation_reduce_fusion_preserves_tensor_bias() -> None:
    """The native fused instruction retains a tensor-bound activation bias."""
    specs = {"data": ((128, 128), "bfloat16"), "bias": ((128,), "bfloat16")}
    ir = build_initial_ir(f_biased_activation_reduce, specs)
    transform = FusePointwiseReduction()
    options = transform.analyze(ir)
    assert len(options) == 1
    transformed = transform.apply(ir, options[0])
    fused = next(
        transformed.tree.isa(nid)
        for nid in transformed.tree.preorder()
        if isinstance(transformed.tree.data(nid), ISANode)
        and transformed.tree.isa(nid).op_cls.NAME == "activation_reduce"
    )
    assert fused.operand_bindings["bias"].tensor == "sbuf_bias"

    rng = np.random.default_rng(53)
    data = rng.standard_normal((128, 128)).astype(np.float32)
    bias = rng.standard_normal((128,)).astype(np.float32)
    actual = fused.op_cls()._run(data=data, bias=bias, **fused.kwargs)
    expected = np.sum(np.exp(data + bias[:, None]), axis=1)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
