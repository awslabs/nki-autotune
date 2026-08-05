"""Tests for slot-style free-axis RFactor."""

from __future__ import annotations

from pathlib import Path
from test._simulation import _load_source

import numpy as np
import pytest

from examples.online_fusion_attention import f_nkigym, f_numpy
from nkigym.codegen import render
from nkigym.ir import KernelIR, build_initial_ir
from nkigym.ir.tree import ISANode
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.synthesis.simulate_nki import simulate_fp32
from nkigym.transforms import FusePointwiseReduction, OnlineFusion, RFactor, Split, SplitOption
from nkigym.transforms._canonical_rewrite import owning_block
from nkigym.transforms.base import TransformLegalityError


@nkigym_kernel
def f_wide_reduction(data):
    """Apply an activation and reduce a wide free axis."""
    loaded = NKILoad()(src=data)
    activated = NKIActivation(op="exp")(data=loaded)
    reduced = NKITensorReduce(op="add", axis=1)(data=activated)
    output = NKIStore()(src=reduced)
    return output


def _split_fused_attention() -> KernelIR:
    """Build online attention with both fused reductions atomically factored."""
    sequence_length = 512
    input_specs = {
        "query": ((128, sequence_length), "bfloat16"),
        "key": ((128, sequence_length), "bfloat16"),
        "value": ((sequence_length, 128), "bfloat16"),
    }
    ir = build_initial_ir(f_nkigym, input_specs)
    transform = OnlineFusion()
    for _stage in range(2):
        online = next(option for option in transform.analyze(ir) if option.chunk_size == 256)
        ir = transform.apply(ir, online)
    fusion = FusePointwiseReduction()
    while options := fusion.analyze(ir):
        ir = fusion.apply(ir, options[0])

    fused_leaves = [
        nid
        for nid in ir.tree.preorder()
        if isinstance((node := ir.tree.data(nid)), ISANode) and node.op_cls.RFACTOR_RECIPE == "slot"
    ]
    for leaf_nid in fused_leaves:
        block_nid = owning_block(ir.tree, leaf_nid)
        free_axis = ir.tree.block(block_nid).axis_map["F"]
        ir = Split().apply(ir, SplitOption(target_nid=leaf_nid, factors=(2, 128), target_axis=free_axis))
    return ir


def _rfactor_all(ir: KernelIR) -> KernelIR:
    """Apply every remaining non-slot RFactor option."""
    transform = RFactor()
    while options := transform.analyze(ir):
        ir = transform.apply(ir, options[0])
    return ir


def test_slot_rfactor_contract() -> None:
    """Split atomically creates FP32 slots and final reduction axes."""
    ir = _split_fused_attention()
    assert RFactor().analyze(ir) == []
    partials = [buffer for name, buffer in ir.all_buffers().items() if name.endswith("_rfactor")]
    assert len(partials) == 2
    assert all(buffer.shape == (512, 2) for buffer in partials)
    assert all(buffer.physical_dtype() == "float32" for buffer in partials)

    fused_blocks = []
    final_blocks = []
    for block_nid in ir.tree.blocks():
        leaves = [
            ir.tree.isa(nid)
            for nid in ir.tree.preorder(block_nid)
            if isinstance(ir.tree.data(nid), ISANode) and owning_block(ir.tree, nid) == block_nid
        ]
        if len(leaves) != 1:
            continue
        if leaves[0].op_cls.RFACTOR_RECIPE == "slot" and leaves[0].op_cls is not NKITensorReduce:
            fused_blocks.append(ir.tree.block(block_nid))
        if leaves[0].op_cls is not None and leaves[0].op_cls.NAME == "tensor_reduce":
            final_blocks.append(ir.tree.block(block_nid))
    assert len(fused_blocks) == 2
    assert len(final_blocks) == 2
    assert all(
        next(iter_var for iter_var in block.iter_vars if iter_var.axis == block.axis_map["F"]).role == AxisRole.PARALLEL
        for block in fused_blocks
    )
    assert all(
        next(iter_var for iter_var in block.iter_vars if iter_var.axis == block.axis_map["F"]).role
        == AxisRole.ACCUMULATION
        for block in final_blocks
    )
    assert all(block.axis_map["F"] not in {"d0", "d1", "d2", "d3"} for block in final_blocks)


def test_factored_slot_axis_cannot_be_split_again() -> None:
    """A second split cannot make several partial reductions overwrite one slot."""
    ir = build_initial_ir(f_wide_reduction, {"data": ((128, 1024), "bfloat16")})
    fusion = FusePointwiseReduction()
    ir = fusion.apply(ir, fusion.analyze(ir)[0])
    leaf_nid = next(
        nid
        for nid in ir.tree.preorder()
        if isinstance((node := ir.tree.data(nid)), ISANode) and node.op_cls.RFACTOR_RECIPE == "slot"
    )
    free_axis = ir.tree.block(owning_block(ir.tree, leaf_nid)).axis_map["F"]
    ir = Split().apply(ir, SplitOption(target_nid=leaf_nid, factors=(2, 512), target_axis=free_axis))

    remaining = [
        option for option in Split().analyze(ir) if option.target_nid == leaf_nid and option.target_axis == free_axis
    ]
    assert remaining == []
    with pytest.raises(TransformLegalityError, match="cannot be split again"):
        Split().apply(ir, SplitOption(target_nid=leaf_nid, factors=(4, 128), target_axis=free_axis))


def test_standalone_tensor_reduce_split_is_atomically_factored(tmp_path: Path) -> None:
    """Splitting a standalone reduction writes partial slots before one final fold."""
    ir = build_initial_ir(f_wide_reduction, {"data": ((128, 256), "bfloat16")})
    leaf_nid = next(
        nid
        for nid in ir.tree.preorder()
        if isinstance((node := ir.tree.data(nid)), ISANode) and node.op_cls is NKITensorReduce
    )
    free_axis = ir.tree.block(owning_block(ir.tree, leaf_nid)).axis_map["F"]
    option = SplitOption(target_nid=leaf_nid, factors=(2, 128), target_axis=free_axis)
    assert option in Split().analyze(ir)
    transformed = Split().apply(ir, option)

    partials = [buffer for name, buffer in transformed.all_buffers().items() if name.endswith("_rfactor")]
    assert len(partials) == 1
    assert partials[0].shape == (128, 2)
    source = render(transformed)
    assert source.count("nisa.tensor_reduce(") == 2

    module = _load_source(source, tmp_path, "standalone_slot_rfactor")
    rng = np.random.default_rng(59)
    inputs = {"data": rng.standard_normal((128, 256)).astype(np.float32)}
    actual = np.asarray(simulate_fp32(module.nki_f_wide_reduction)(**inputs))
    expected = np.sum(np.exp(inputs["data"]), axis=1)
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_factored_attention_matches_numpy(tmp_path: Path) -> None:
    """The two-stage reductions preserve online attention numerically."""
    ir = _rfactor_all(_split_fused_attention())
    source = render(ir)
    assert source.count("nisa.tensor_scalar_reduce(") == 1
    assert source.count("nisa.activation_reduce(") == 1
    assert source.count("nisa.tensor_reduce(") == 2
    module = _load_source(source, tmp_path, "slot_rfactor_attention")
    rng = np.random.default_rng(43)
    inputs = {
        "query": rng.standard_normal((128, 512)).astype(np.float32),
        "key": rng.standard_normal((128, 512)).astype(np.float32),
        "value": rng.standard_normal((512, 128)).astype(np.float32),
    }
    actual = np.asarray(simulate_fp32(module.nki_f_nkigym)(**inputs))
    expected = f_numpy(**inputs)
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
