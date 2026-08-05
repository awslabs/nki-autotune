"""Tests for nkigym.transforms.SoftwarePipeline (Tier B)."""

from __future__ import annotations

from test._simulation import assert_matmul_ir_simulates
from test.transforms._fixtures import build_canonical_ir
from test.transforms._pipeline_fixtures import INPUT_SPECS, TRACE, f_nkigym, m_loop_and_children, tuned_ir

import pytest

from nkigym.codegen import render
from nkigym.environment import Action, KernelMDP
from nkigym.ir import KernelIR
from nkigym.ir.tree import Buffer, ISANode
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.transforms import (
    BufferCompaction,
    BufferLayout,
    BufferLayoutOption,
    CodeMotion,
    CodeMotionOption,
    Fuse,
    FuseOption,
    Reorder,
    ReorderOption,
    SoftwarePipeline,
    SoftwarePipelineOption,
    Split,
    SplitOption,
    TransformLegalityError,
)
from nkigym.transforms._canonical_rewrite import append_block, append_root_buffers, finalize_rewrite, required_spec


def _apply_action(ir: KernelIR, action: Action) -> KernelIR:
    """Apply one heterogeneous transform action."""
    transform, option = action
    return transform.apply(ir, option)


def _check_analyze_enumerates_nondecreasing_labelings():
    """The tuned M loop yields exactly the contiguous non-decreasing stage labelings."""
    ir = tuned_ir()
    opts = SoftwarePipeline().analyze(ir)
    m_loop, children = m_loop_and_children(ir)
    stage_sets = {o.stages for o in opts if o.loop_nid == m_loop}
    assert (0, 0, 1) in stage_sets
    assert (0, 1, 1) in stage_sets
    assert (0, 1, 2) in stage_sets
    assert (0, 0, 0) not in stage_sets
    assert all(max(s) <= len(children) - 1 for s in stage_sets)


def _check_apply_derives_versions_and_annotates():
    """apply((0,0,1)) versions PSUM and renders fill, steady, and drain phases."""
    ir = tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    n_nodes_before = ir.tree.graph.number_of_nodes()
    new_ir = SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 2)))
    assert new_ir.buffer("psum_prod").versions == 2
    assert new_ir.tree.graph.number_of_nodes() == n_nodes_before
    anns = [new_ir.tree.block(nid).annotations.get("software_pipeline") for nid in new_ir.tree.blocks()]
    assert any(a and a["stages"] == (0, 0, 1) for a in anns)
    assert any(a and a["versioned_buffers"] == ("psum_prod",) for a in anns)
    source = render(new_ir)
    assert "for i_d1_0 in range(15):" in source
    assert "dst=psum_prod[0][0:128, (i_d1_0 + 1) % 2, 0:0 + 2048]" in source
    assert "src=psum_prod[0][0:128, i_d1_0 % 2, 0:0 + 2048]" in source
    assert "src=psum_prod[0][0:128, 1, 0:0 + 2048], dst=sbuf_prod[0][0:128, 15, 0:0 + 2048]" in source


def _check_apply_rejects_consumer_before_producer_stage():
    """A stage assignment putting a consumer earlier than its producer raises."""
    ir = tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    with pytest.raises(TransformLegalityError):
        SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=m_loop, stages=(1, 0, 1), order=(0, 1, 2)))


def _check_apply_rejects_duplicate_order():
    """An order array that is not a permutation raises."""
    ir = tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    with pytest.raises(TransformLegalityError):
        SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 1)))


def _check_apply_rejects_malformed_stages():
    """Stage labels must start at zero, be contiguous, and contain work in two stages."""
    ir = tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    for stages in ((0, 0, 0), (0, 0, 2), (1, 1, 2)):
        with pytest.raises(TransformLegalityError):
            SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=m_loop, stages=stages, order=(0, 1, 2)))


def _wide_tuned_ir() -> KernelIR:
    """Return the tuned fixture with sixteen logical accumulator tiles."""
    ir = tuned_ir()
    object.__setattr__(ir.buffer("psum_prod"), "shape", (2048, 2048))
    return ir


def _check_analyze_composes_pipeline_and_list_layout():
    """Each transform still offers its option after the other transform applies."""
    ir = _wide_tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    listed_ir = BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=16))
    option = SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 2))
    assert option in SoftwarePipeline().analyze(listed_ir)

    ir = _wide_tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    pipelined_ir = SoftwarePipeline().apply(
        ir, SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 2))
    )
    layouts = [candidate for candidate in BufferLayout().analyze(pipelined_ir) if candidate.tensor == "psum_prod"]
    assert BufferLayoutOption(tensor="psum_prod", list_len=16) in layouts


def _check_apply_composes_in_both_orders():
    """List layout and software pipelining commute on buffer geometry and rendering."""
    ir = _wide_tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    listed_ir = BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=16))
    option = SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 2))
    listed_then_pipelined = SoftwarePipeline().apply(listed_ir, option)

    ir = _wide_tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    option = SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 2))
    pipelined_ir = SoftwarePipeline().apply(ir, option)
    pipelined_then_listed = BufferLayout().apply(pipelined_ir, BufferLayoutOption(tensor="psum_prod", list_len=16))

    for composed in (listed_then_pipelined, pipelined_then_listed):
        buffer = composed.buffer("psum_prod")
        assert (buffer.list_len, buffer.versions) == (16, 2)
        assert buffer.per_tile_physical_shape() == (128, 2, 2048)
    assert render(listed_then_pipelined) == render(pipelined_then_listed)


def _check_large_body_option_generation_is_bounded():
    """A seventeen-child body offers every contiguous two- and three-stage partition."""
    labelings = SoftwarePipeline()._nondecreasing_labelings(17)
    assert len(labelings) == 136
    assert (0,) * 16 + (1,) in labelings
    assert (0,) * 4 + (1,) * 8 + (2,) * 5 in labelings
    assert tuple(range(17)) not in labelings
    assert all(max(stages) in {1, 2} for stages in labelings)


def _check_pipeline_rejects_partial_version_write_with_wider_read():
    """A pipeline version cannot be read beyond the slice written in that iteration."""
    ir = build_canonical_ir()
    trace: tuple[Action, ...] = (
        (Split(), SplitOption(target_nid=8, factors=(8, 2), target_axis=None)),
        (Split(), SplitOption(target_nid=5, factors=(2, 8), target_axis=None)),
        (Split(), SplitOption(target_nid=17, factors=(4, 512), target_axis="d2")),
        (Split(), SplitOption(target_nid=21, factors=(4, 2), target_axis=None)),
        (BufferLayout(), BufferLayoutOption(tensor="psum_prod", list_len=4)),
        (CodeMotion(), CodeMotionOption(block_nid=18, target_loop_nid=25, index=1)),
        (Split(), SplitOption(target_nid=20, factors=(4, 512), target_axis="d2")),
        (CodeMotion(), CodeMotionOption(block_nid=4, target_loop_nid=26, index=1)),
        (Split(), SplitOption(target_nid=12, factors=(2, 2, 4), target_axis=None)),
        (BufferLayout(), BufferLayoutOption(tensor="sbuf_rhs", list_len=8)),
        (Split(), SplitOption(target_nid=24, factors=(2, 4), target_axis=None)),
        (Split(), SplitOption(target_nid=3, factors=(2, 8, 128), target_axis="d1")),
        (Reorder(), ReorderOption(outer_nid=29, inner_nid=30)),
        (Split(), SplitOption(target_nid=2, factors=(2, 2, 4), target_axis=None)),
        (Fuse(), FuseOption(target_nids=(23, 32, 33), target_axis=None)),
        (Fuse(), FuseOption(target_nids=(30, 31), target_axis=None)),
    )
    for action in trace:
        ir = _apply_action(ir, action)
    option = SoftwarePipelineOption(loop_nid=25, stages=(0, 1), order=(0, 1))
    with pytest.raises(TransformLegalityError):
        SoftwarePipeline().apply(ir, option)
    assert option not in SoftwarePipeline().analyze(ir)


def _check_pipeline_rejects_loop_touching_an_already_versioned_buffer():
    """A nested pipeline cannot replace an existing buffer's rotation variable."""
    ir = tuned_ir()
    outer_loop, outer_children = m_loop_and_children(ir)
    ir = SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=outer_loop, stages=(0, 0, 1), order=(0, 1, 2)))
    load_block = next(
        nid
        for nid in ir.tree.blocks()
        if sum(1 for desc in ir.tree.descendants(nid) if isinstance(ir.tree.data(desc), ISANode)) == 1
        and any(
            isinstance(ir.tree.data(desc), ISANode) and ir.tree.isa(desc).op_cls.__name__ == "NKILoad"
            for desc in ir.tree.descendants(nid)
        )
    )
    inner_loop = outer_children[1]
    ir = CodeMotion().apply(ir, CodeMotionOption(block_nid=load_block, target_loop_nid=inner_loop, index=0))
    option = SoftwarePipelineOption(loop_nid=inner_loop, stages=(0, 1), order=(0, 1))
    with pytest.raises(TransformLegalityError):
        SoftwarePipeline().apply(ir, option)
    assert option not in SoftwarePipeline().analyze(ir)


def _check_pipeline_rejects_versioned_buffer_liveout() -> None:
    """A versioned intermediate cannot have an unrotated reader after the pipeline."""
    environment = KernelMDP(f_nkigym, INPUT_SPECS, transforms=[Split(), Fuse(), Reorder(), CodeMotion()])
    ir = environment.reset()
    for transform, transform_option in TRACE:
        if isinstance(transform, BufferCompaction):
            break
        ir = environment.step(ir, (transform, transform_option))

    append_root_buffers(
        ir, (Buffer(name="sbuf_liveout", shape=ir.buffer("psum_prod").shape, dtype="bfloat16", location="sbuf"),)
    )
    liveout = required_spec(ir, NKITensorCopy, {"src": "psum_prod", "dst": "sbuf_liveout"}, {"P": "d1", "F": "d2"}, {})
    liveout_block = append_block(ir.tree, liveout)
    ir.tree.graph.add_edge(ir.tree.root, liveout_block)
    finalize_rewrite(ir)

    loop_nid, _children = m_loop_and_children(ir)
    option = SoftwarePipelineOption(loop_nid=loop_nid, stages=(0, 0, 1), order=(0, 1, 2))
    with pytest.raises(TransformLegalityError):
        SoftwarePipeline().apply(ir, option)
    assert option not in SoftwarePipeline().analyze(ir)


def _check_code_motion_invalidates_changed_pipeline_children():
    """Moving a direct child into a staged loop drops stale stages and versions."""
    ir = tuned_ir()
    loop_nid, children = m_loop_and_children(ir)
    ir = SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=loop_nid, stages=(0, 0, 1), order=(0, 1, 2)))
    option = next(
        candidate
        for candidate in CodeMotion().analyze(ir)
        if ir.tree.parent(candidate.block_nid) == ir.tree.root and candidate.target_loop_nid == loop_nid
    )
    transformed = CodeMotion().apply(ir, option)
    assert transformed.tree.children(loop_nid) != children
    assert transformed.buffer("psum_prod").versions == 1
    assert all(
        "software_pipeline" not in transformed.tree.block(block_nid).annotations
        for block_nid in transformed.tree.blocks()
    )
    render(transformed)


def test_increment1_sim_matches_numpy(tmp_path) -> None:
    """The pipelined kernel computes lhs_T.T @ rhs (fp32 CPU sim)."""
    ir = tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    new_ir = SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 2)))
    assert_matmul_ir_simulates(new_ir, tmp_path, "software_pipeline")


def test_pipeline_analysis_and_apply_contract() -> None:
    """Analysis enumerates valid stages and apply derives versions and annotations."""
    _check_analyze_enumerates_nondecreasing_labelings()
    _check_large_body_option_generation_is_bounded()
    _check_apply_derives_versions_and_annotates()


def test_pipeline_rejects_invalid_stage_and_order_assignments() -> None:
    """Consumer-before-producer stages and duplicate order values fail."""
    _check_apply_rejects_consumer_before_producer_stage()
    _check_apply_rejects_duplicate_order()
    _check_apply_rejects_malformed_stages()


def test_pipeline_composes_with_listed_buffer_versioning() -> None:
    """Analysis and apply compose list layout with pipeline versions in either order."""
    _check_analyze_composes_pipeline_and_list_layout()
    _check_apply_composes_in_both_orders()


def test_pipeline_rejects_inconsistent_versioned_accesses() -> None:
    """Partial writes, live-outs, and nested pipelines cannot create inconsistent versions."""
    _check_pipeline_rejects_partial_version_write_with_wider_read()
    _check_pipeline_rejects_loop_touching_an_already_versioned_buffer()
    _check_pipeline_rejects_versioned_buffer_liveout()


def test_structural_rewrite_invalidates_stale_pipeline() -> None:
    """A post-pipeline child-list rewrite remains renderable."""
    _check_code_motion_invalidates_changed_pipeline_children()
