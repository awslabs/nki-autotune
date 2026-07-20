"""Tests for the blind canonical matmul experiment contract."""

import inspect

from autotune.search import codex_policy
from autotune.search.engine import AgenticSearch
from autotune.search.policy_json import REASONING_POLICY_PROMPT, STRATEGY_POLICY_PROMPT
from examples import agentic_matmul_search
from examples.matmul_lhsT_rhs import TRANSFORMS


def test_search_engine_has_no_initial_state_injection() -> None:
    """Every search root must come from its environment's canonical reset."""
    parameters = inspect.signature(AgenticSearch).parameters
    assert "initial_state" not in parameters


def test_matmul_policy_context_contains_no_manual_target() -> None:
    """Policy-visible text and construction contain no comparison schedule."""
    visible_text = (
        REASONING_POLICY_PROMPT
        + "\n"
        + STRATEGY_POLICY_PROMPT
        + "\n"
        + agentic_matmul_search.MATMUL_GUIDANCE
        + "\n"
        + inspect.getsource(agentic_matmul_search._run_search)
    ).lower()
    forbidden = ("kernel_35", "kernel_target", "manual_transforms", "90.9", "reference action", "target sequence")
    assert all(term not in visible_text for term in forbidden)


def test_matmul_policy_uses_gpt_5_6_sol_without_claude_adapter() -> None:
    """The experiment uses the requested isolated GPT policy implementation."""
    source = inspect.getsource(agentic_matmul_search) + inspect.getsource(codex_policy)
    assert 'default="openai.gpt-5.6-sol"' in source
    assert all(term not in source.lower() for term in ("anthropic", "claude", "opus"))


def test_matmul_search_uses_every_shipped_transform() -> None:
    """The experiment exposes the complete shipped transform namespace."""
    names = {type(transform).__name__ for transform in TRANSFORMS}
    assert names == {
        "BufferCompaction",
        "BufferLayout",
        "CodeMotion",
        "Fuse",
        "Reorder",
        "RFactor",
        "SoftwarePipeline",
        "Split",
    }
