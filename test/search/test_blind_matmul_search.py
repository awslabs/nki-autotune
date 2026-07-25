"""Tests for canonical profiler-feedback matmul refinement."""

import inspect
import random

import numpy as np

from examples import matmul_lhsT_rhs_agentic_search, random_rollout
from nkigym.environment import KernelMDP
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


def test_combined_rollout_uses_every_shipped_transform() -> None:
    """The shared rollout exposes both workloads and every transform."""
    expected = {
        "BufferCompaction",
        "BufferLayout",
        "CodeMotion",
        "Fuse",
        "LoadTranspose",
        "MatmulTranspose",
        "Reorder",
        "RFactor",
        "SoftwarePipeline",
        "Split",
    }
    assert set(random_rollout.WORKLOADS) == {"matmul_lhsT_rhs", "matmul_lhs_rhs"}
    assert {type(transform).__name__ for transform in random_rollout.TRANSFORMS} == expected


def test_combined_rollout_graphs_match_numpy() -> None:
    """Each configured canonical graph matches its NumPy function."""
    rng = np.random.default_rng(0)
    cases = (
        (
            random_rollout.LHS_T_RHS,
            {
                "lhs_T": rng.standard_normal((3, 2)).astype(np.float32),
                "rhs": rng.standard_normal((3, 4)).astype(np.float32),
            },
        ),
        (
            random_rollout.LHS_RHS,
            {
                "lhs": rng.standard_normal((2, 3)).astype(np.float32),
                "rhs": rng.standard_normal((3, 4)).astype(np.float32),
            },
        ),
    )
    for workload, inputs in cases:
        expected = workload.f_numpy(**inputs)
        actual = np.asarray(workload.f_nkigym(**inputs))
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_lhs_random_rollouts_cover_both_transpose_rewrites() -> None:
    """Retained lhs seeds immediately exercise both transpose strategies."""
    workload = random_rollout.LHS_RHS
    environment = KernelMDP(workload.f_nkigym, workload.input_specs, transforms=random_rollout.TRANSFORMS)
    initial = environment.reset()
    actions = environment.legal_actions(initial)
    selected = {type(random.Random(seed).choice(actions)[0]).__name__ for seed in workload.rollout_seeds}
    assert selected == {"LoadTranspose", "MatmulTranspose"}
