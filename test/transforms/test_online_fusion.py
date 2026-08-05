"""End-to-end tests for contract-driven online fusion."""

from __future__ import annotations

from pathlib import Path
from test._simulation import _load_source
from test.transforms._online_fusion_fixtures import (
    ATTENTION_INPUT_SPECS,
    MAPPED_ATTENTION_INPUT_SPECS,
    attention_reference,
    build_load_first_attention_ir,
    build_mapped_attention_ir,
    build_naive_attention_ir,
)

import numpy as np
import pytest

from examples import online_fusion_attention as demo
from nkigym.codegen import render
from nkigym.ir.tree import BlockNode, ForNode, ISANode
from nkigym.ops.base import AxisRole
from nkigym.synthesis.simulate_nki import simulate_fp32
from nkigym.transforms import CodeMotion, InsertTransposePair, OnlineFusion, TransformLegalityError


def _online_fusion_states(original):
    """Return the independently selectable prefix and completed fusion states."""
    transform = OnlineFusion()
    prefix = transform.apply(original, transform.analyze(original)[0])
    completed = transform.apply(prefix, transform.analyze(prefix)[0])
    return prefix, completed


def test_analyze_exposes_contract_proven_chunk() -> None:
    """Attention exposes the ``(m,l)`` prefix before the value-matmul extension."""
    transform = OnlineFusion()
    original = build_naive_attention_ir()
    options = transform.analyze(original)
    assert len(options) == 1
    assert options[0].match_id == ("d2", (15, 21))
    assert options[0].chunk_size == 128
    prefix = transform.apply(original, options[0])
    completions = transform.analyze(prefix)
    assert completions == [type(options[0])(match_id=("d2", (15, 21, 36)), chunk_size=128)]


def test_completion_preserves_intervening_node_allocations() -> None:
    """Completion retains unrelated nodes allocated after the prefix."""
    online = OnlineFusion()
    original = build_naive_attention_ir()
    prefix = online.apply(original, online.analyze(original)[0])
    insertion = InsertTransposePair()
    option = next(candidate for candidate in insertion.analyze(prefix) if candidate.source == "sbuf_query_t")
    with_pair = insertion.apply(prefix, option)
    added_nodes = set(with_pair.tree.graph) - set(prefix.tree.graph)
    added_buffers = set(with_pair.all_buffers()) - set(prefix.all_buffers())
    assert added_nodes
    assert added_buffers

    completions = online.analyze(with_pair)
    assert len(completions) == 1
    completed = online.apply(with_pair, completions[0])
    assert added_nodes <= set(completed.tree.graph)
    assert added_buffers <= set(completed.all_buffers())


def test_modified_incremental_prefix_has_no_completion() -> None:
    """Moving a retained prefix root invalidates only its completion action."""
    online = OnlineFusion()
    original = build_naive_attention_ir()
    prefix = online.apply(original, online.analyze(original)[0])
    completion = online.analyze(prefix)[0]
    added_roots = set(prefix.tree.children(prefix.tree.root)) - set(original.tree.children(original.tree.root))
    motion = CodeMotion()
    option = next(candidate for candidate in motion.analyze(prefix) if candidate.block_nid in added_roots)
    moved = motion.apply(prefix, option)

    assert online.analyze(moved) == []
    with pytest.raises(TransformLegalityError, match="illegal OnlineFusion completion"):
        online.apply(moved, completion)


def test_apply_emits_shared_sequential_loop_and_native_ops() -> None:
    """Lowering uses ordinary blocks and one shared sequential progress loop."""
    original = build_naive_attention_ir()
    _prefix, transformed = _online_fusion_states(original)
    sequential_blocks = [
        transformed.tree.block(nid)
        for nid in transformed.tree.blocks()
        if any(iter_var.role == AxisRole.SEQUENTIAL for iter_var in transformed.tree.block(nid).iter_vars)
    ]
    assert len(sequential_blocks) == 1
    carrier_nid = next(
        nid
        for nid in transformed.tree.blocks()
        if any(iter_var.role == AxisRole.SEQUENTIAL for iter_var in transformed.tree.block(nid).iter_vars)
    )
    loops = [
        payload
        for nid in transformed.tree.descendants(carrier_nid)
        if isinstance(payload := transformed.tree.data(nid), ForNode)
    ]
    assert any(loop.extent == 2 and loop.loop_var == "i_d2_online" for loop in loops)
    leaves = [
        payload
        for nid in transformed.tree.descendants(carrier_nid)
        if isinstance(payload := transformed.tree.data(nid), ISANode)
    ]
    assert any(leaf.op_cls.NAME == "scalar_tensor_tensor" for leaf in leaves)
    assert all(
        isinstance(transformed.tree.data(nid), (BlockNode, ForNode, ISANode)) for nid in transformed.tree.preorder()
    )
    source = render(transformed)
    assert "nisa.scalar_tensor_tensor(" in source
    assert "online_fusion_chain" not in source
    assert "value=float('-inf')" in source
    assert source.count("op=nl.reciprocal") == 1
    assert "sbuf_probability =" not in source
    assert "nisa.tensor_copy(src=psum_output" in source
    assert source.index("op=nl.reciprocal") > source.index("for i_d2_online in range(2):")


def test_transformed_attention_matches_numpy(tmp_path: Path) -> None:
    """Both independently selectable online-fusion states preserve attention."""
    original = build_naive_attention_ir()
    states = _online_fusion_states(original)
    rng = np.random.default_rng(13)
    inputs = {
        name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in ATTENTION_INPUT_SPECS.items()
    }
    expected = attention_reference(**inputs)
    for index, transformed in enumerate(states):
        module = _load_source(render(transformed), tmp_path, f"flash_attention_{index}")
        actual = np.asarray(simulate_fp32(module.nki_f_naive_attention)(**inputs))
        np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_load_first_attention_transforms_across_interleaved_root_blocks(tmp_path: Path) -> None:
    """Dependency-valid placement handles a noncontiguous matched chain."""
    original = build_load_first_attention_ir()
    options = OnlineFusion().analyze(original)
    assert len(options) == 1
    _prefix, transformed = _online_fusion_states(original)
    module = _load_source(render(transformed), tmp_path, "load_first_flash_attention")
    rng = np.random.default_rng(17)
    inputs = {
        name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in ATTENTION_INPUT_SPECS.items()
    }
    actual = np.asarray(simulate_fp32(module.nki_f_load_first_attention)(**inputs))
    expected = attention_reference(**inputs)
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_mapped_state_uses_fp32_hbm_carry_and_preserves_output_dtype(tmp_path: Path) -> None:
    """Multi-tile state is carried through FP32 HBM and stored to the typed output."""
    original = build_mapped_attention_ir()
    options = OnlineFusion().analyze(original)
    assert [option.chunk_size for option in options] == [128]
    _prefix, transformed = _online_fusion_states(original)
    assert transformed.buffer("sbuf_row_max").physical_dtype() == "float32"
    assert transformed.buffer("sbuf_row_sum").physical_dtype() == "float32"
    assert transformed.buffer("sbuf_output").physical_dtype() == "float32"
    assert transformed.buffer("hbm_output").physical_dtype() == "bfloat16"
    carry = transformed.buffer("hbm_output_online_carry")
    assert carry.location == "shared_hbm"
    assert carry.physical_dtype() == "float32"

    source = render(transformed)
    assert "hbm_output_online_carry = nl.ndarray((256, 128), dtype=nl.float32" in source
    assert source.count("dst=hbm_output_online_carry[") == 2
    assert source.count("dst=hbm_output[") == 1
    assert source.count("op=nl.reciprocal") == 1
    module = _load_source(source, tmp_path, "mapped_flash_attention")
    rng = np.random.default_rng(29)
    inputs = {
        name: rng.standard_normal(shape).astype(np.float32)
        for name, (shape, _dtype) in MAPPED_ATTENTION_INPUT_SPECS.items()
    }
    actual = np.asarray(simulate_fp32(module.nki_f_nkigym)(**inputs))
    expected = attention_reference(inputs["query"].T, inputs["key"].T, inputs["value"])
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_hardcoded_ladder_final_state_matches_numpy(tmp_path: Path) -> None:
    """The final literal ladder state preserves attention numerically."""
    input_specs = demo._input_specs(demo.VALIDATION_QUERY_LENGTH)
    state = demo._build_ladder(input_specs)[-1]
    module = _load_source(render(state), tmp_path, "hardcoded_attention_ladder")
    rng = np.random.default_rng(47)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in input_specs.items()}
    actual = np.asarray(simulate_fp32(module.nki_f_nkigym)(**inputs))
    expected = demo.f_numpy(**inputs)
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
