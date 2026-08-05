"""Tests for the matmul example drivers."""

import inspect

from examples import matmul_lhsT_rhs_agentic_search, transpose_layout_demo
from examples._matmul_workloads import TRANSFORMS
from nkigym.codegen import render
from nkigym.search import codex_policy
from nkigym.search.engine import ProfilerGuidedRefinement
from nkigym.search.policy_json import REASONING_POLICY_PROMPT


def test_refinement_has_no_initial_state_injection() -> None:
    """Every refinement root comes from the environment reset."""
    parameters = inspect.signature(ProfilerGuidedRefinement).parameters
    assert "initial_state" not in parameters


def test_matmul_policy_context_contains_no_manual_target() -> None:
    """Policy-visible text contains no comparison schedule."""
    visible_text = (
        REASONING_POLICY_PROMPT
        + "\n"
        + matmul_lhsT_rhs_agentic_search.MATMUL_GUIDANCE
        + "\n"
        + inspect.getsource(matmul_lhsT_rhs_agentic_search._run_search)
    ).lower()
    forbidden = ("kernel_35", "kernel_target", "manual_ladder", "90.9", "reference action", "target sequence")
    assert all(term not in visible_text for term in forbidden)


def test_matmul_policy_uses_gpt_5_6_sol_without_claude_adapter() -> None:
    """The example uses the requested isolated GPT policy."""
    source = inspect.getsource(matmul_lhsT_rhs_agentic_search) + inspect.getsource(codex_policy)
    assert 'default="openai.gpt-5.6-sol"' in source
    assert all(term not in source.lower() for term in ("anthropic", "claude", "opus"))


def test_matmul_search_uses_every_shipped_transform() -> None:
    """The search examples expose every shipped transform."""
    expected = {
        "BufferCompaction",
        "BufferLayout",
        "CancelTransposePair",
        "CodeMotion",
        "Fuse",
        "InsertTransposePair",
        "Reorder",
        "RFactor",
        "SoftwarePipeline",
        "Split",
        "TransposeThroughLoad",
        "TransposeThroughMatmul",
        "TransposeThroughTensorCopy",
    }
    assert {type(transform).__name__ for transform in TRANSFORMS} == expected


def test_transpose_demo_hardcodes_two_transform_traces() -> None:
    """The demo contains no reasoning policy or transform search."""
    without_names = [type(transform).__name__ for transform, _option in transpose_layout_demo.WITHOUT_TRANSPOSE_TRACE]
    with_names = [type(transform).__name__ for transform, _option in transpose_layout_demo.WITH_TRANSPOSE_TRACE]
    assert without_names == ["CodeMotion", "BufferCompaction", "BufferLayout"]
    assert with_names == [
        "TransposeThroughLoad",
        "InsertTransposePair",
        "TransposeThroughMatmul",
        "TransposeThroughTensorCopy",
    ]
    source = inspect.getsource(transpose_layout_demo)
    assert "ProfilerGuidedRefinement" not in source
    assert "CodexTransformPolicy" not in source
    assert "MATMUL_GUIDANCE" not in source


def test_transpose_demo_traces_render_the_expected_transpose_paths() -> None:
    """The fixed comparison uses both requested transpose commutes."""
    without_source = render(transpose_layout_demo._apply_trace(transpose_layout_demo.WITHOUT_TRANSPOSE_TRACE))
    with_source = render(transpose_layout_demo._apply_trace(transpose_layout_demo.WITH_TRANSPOSE_TRACE))
    assert without_source.count("nisa.nc_transpose") == 1
    assert without_source.count("nisa.dma_transpose") == 0
    assert with_source.count("nisa.nc_transpose") == 0
    assert with_source.count("nisa.dma_transpose") == 2


def test_transpose_demo_uses_the_constructed_skewed_workload() -> None:
    """The fixed traces use the intended narrow-output matmul."""
    assert transpose_layout_demo.WORKLOAD.input_specs == {
        "lhs": ((4096, 1024), "bfloat16"),
        "rhs": ((1024, 128), "bfloat16"),
    }
    assert (transpose_layout_demo.M, transpose_layout_demo.N) == (4096, 128)
