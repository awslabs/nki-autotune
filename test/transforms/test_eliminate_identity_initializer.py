"""Tests for contract-driven identity-initializer elimination."""

from __future__ import annotations

from examples import online_fusion_attention as demo
from nkigym.codegen import render
from nkigym.environment import KernelMDP
from nkigym.ir import KernelIR
from nkigym.transforms import EliminateIdentityInitializer


def _state_before_elimination() -> KernelIR:
    """Build the hardcoded attention rung immediately before elimination."""
    input_specs = demo._input_specs(demo.VALIDATION_QUERY_LENGTH)
    environment = KernelMDP(demo.f_nkigym, input_specs, [transform for transform, _option in demo.ACTIONS])
    state = environment.reset()
    elimination_index = next(
        index
        for index, (transform, _option) in enumerate(demo.ACTIONS)
        if isinstance(transform, EliminateIdentityInitializer)
    )
    for action in demo.ACTIONS[:elimination_index]:
        state = environment.step(state, action)
    return state


def test_eliminates_only_one_step_score_initializer() -> None:
    """A fresh single-call QK tile overwrites, while long PV accumulation stays initialized."""
    original = _state_before_elimination()
    transform = EliminateIdentityInitializer()
    options = transform.analyze(original)
    assert len(options) == 1
    assert options[0].tensor == "psum_scores"

    transformed = transform.apply(original, options[0])
    source = render(transformed)
    assert "nisa.memset(dst=psum_scores" not in source
    assert "nisa.memset(dst=psum_output" in source
    assert transform.analyze(transformed) == []
