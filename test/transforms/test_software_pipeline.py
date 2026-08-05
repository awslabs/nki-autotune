"""Tests for nkigym.transforms.SoftwarePipeline (Tier B)."""

from __future__ import annotations

from test._simulation import assert_matmul_ir_simulates
from test.transforms._fixtures import build_canonical_ir
from test.transforms._pipeline_fixtures import m_loop_and_children, tuned_ir

import pytest

from nkigym.environment import Action
from nkigym.ir import KernelIR
from nkigym.ir.tree import ISANode
from nkigym.transforms import (
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
    """apply((0,0,1)) sets psum versions=2 and writes the annotation; tree unchanged."""
    ir = tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    n_nodes_before = ir.tree.graph.number_of_nodes()
    new_ir = SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 2)))
    assert new_ir.buffer("psum_prod").versions == 2
    assert new_ir.tree.graph.number_of_nodes() == n_nodes_before
    anns = [new_ir.tree.block(nid).annotations.get("software_pipeline") for nid in new_ir.tree.blocks()]
    assert any(a and a["stages"] == (0, 0, 1) for a in anns)
    assert any(a and a["versioned_buffers"] == ("psum_prod",) for a in anns)


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


def _check_analyze_omits_pipeline_that_would_version_a_list_buffer():
    """Pipeline analysis excludes options that would multi-version a listed buffer."""
    ir = tuned_ir()
    object.__setattr__(ir.buffer("psum_prod"), "shape", (2048, 2048))
    m_loop, _children = m_loop_and_children(ir)
    listed_ir = BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=16))
    option = SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 2))
    assert option not in SoftwarePipeline().analyze(listed_ir)


def _check_apply_rejects_pipeline_that_would_version_a_list_buffer():
    """Direct apply rejects an option that would multi-version a listed buffer."""
    ir = tuned_ir()
    object.__setattr__(ir.buffer("psum_prod"), "shape", (2048, 2048))
    m_loop, _children = m_loop_and_children(ir)
    listed_ir = BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=16))
    option = SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 2))
    with pytest.raises(TransformLegalityError):
        SoftwarePipeline().apply(listed_ir, option)


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


def test_increment1_sim_matches_numpy(tmp_path) -> None:
    """The pipelined kernel computes lhs_T.T @ rhs (fp32 CPU sim)."""
    ir = tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    new_ir = SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 2)))
    assert_matmul_ir_simulates(new_ir, tmp_path, "software_pipeline")


def test_pipeline_analysis_and_apply_contract() -> None:
    """Analysis enumerates valid stages and apply derives versions and annotations."""
    _check_analyze_enumerates_nondecreasing_labelings()
    _check_apply_derives_versions_and_annotates()


def test_pipeline_rejects_invalid_stage_and_order_assignments() -> None:
    """Consumer-before-producer stages and duplicate order values fail."""
    _check_apply_rejects_consumer_before_producer_stage()
    _check_apply_rejects_duplicate_order()


def test_pipeline_rejects_listed_buffer_versioning() -> None:
    """Analysis and direct apply reject versioning an already listed buffer."""
    _check_analyze_omits_pipeline_that_would_version_a_list_buffer()
    _check_apply_rejects_pipeline_that_would_version_a_list_buffer()


def test_pipeline_rejects_inconsistent_versioned_accesses() -> None:
    """Partial writes and nested pipelines cannot create inconsistent versions."""
    _check_pipeline_rejects_partial_version_write_with_wider_read()
    _check_pipeline_rejects_loop_touching_an_already_versioned_buffer()
