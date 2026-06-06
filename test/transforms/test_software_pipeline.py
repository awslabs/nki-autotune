"""Tests for nkigym.transforms.SoftwarePipeline (Tier B)."""

from __future__ import annotations

import importlib.util
import os
import tempfile

import numpy as np
import pytest

import kernel_transforms as KT
from nkigym.codegen import render
from nkigym.synthesis.simulate_nki import simulate_fp32
from nkigym.transforms import SoftwarePipeline, SoftwarePipelineOption, TransformLegalityError
from test.transforms._ladder_compare import assert_matches_hand
from test.transforms._pipeline_fixtures import m_loop_and_children, tuned_ir


def test_analyze_enumerates_nondecreasing_labelings():
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


def test_apply_derives_versions_and_annotates():
    """apply((0,0,1)) sets psum versions=2 and writes the annotation; tree unchanged."""
    ir = tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    n_nodes_before = ir.tree.graph.number_of_nodes()
    new_ir = SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 2)))
    assert new_ir.buffer("psum_prod").versions == 2
    assert new_ir.tree.graph.number_of_nodes() == n_nodes_before
    anns = [new_ir.tree.data(nid).annotations.get("software_pipeline") for nid in new_ir.tree.blocks()]
    assert any(a and a["stages"] == (0, 0, 1) for a in anns)


def test_apply_rejects_consumer_before_producer_stage():
    """A stage assignment putting a consumer earlier than its producer raises."""
    ir = tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    with pytest.raises(TransformLegalityError):
        SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=m_loop, stages=(1, 0, 1), order=(0, 1, 2)))


def test_apply_rejects_duplicate_order():
    """An order array that is not a permutation raises."""
    ir = tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    with pytest.raises(TransformLegalityError):
        SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 1)))


def test_increment1_matches_kernel_15_byte_exact():
    """render(apply(tuned, (0,0,1))) reproduces the validated rotate_only kernel."""
    ir = tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    new_ir = SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 2)))
    assert_matches_hand(render(new_ir), KT.kernel_15)


def test_increment1_sim_matches_numpy():
    """The pipelined kernel computes lhs_T.T @ rhs (fp32 CPU sim)."""
    ir = tuned_ir()
    m_loop, _children = m_loop_and_children(ir)
    new_ir = SoftwarePipeline().apply(ir, SoftwarePipelineOption(loop_nid=m_loop, stages=(0, 0, 1), order=(0, 1, 2)))
    source = render(new_ir)
    rng = np.random.default_rng(0)
    inputs = {
        "lhs_T": rng.standard_normal((2048, 2048)).astype(np.float32),
        "rhs": rng.standard_normal((2048, 2048)).astype(np.float32),
    }
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    scratch = os.path.join(tempfile.gettempdir(), "sp_sim_scratch.py")
    with open(scratch, "w", encoding="utf-8") as handle:
        handle.write(source)
    spec = importlib.util.spec_from_file_location("sp_dumped", scratch)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = np.asarray(simulate_fp32(module.nki_f_nkigym)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
