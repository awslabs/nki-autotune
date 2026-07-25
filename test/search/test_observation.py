"""Tests for measured-state refinement observations."""

from pathlib import Path

from examples.random_rollout import LHS_T_RHS, TRANSFORMS
from nkigym.environment import KernelMDP
from nkigym.search.observation import describe_actions, format_observation, state_fingerprint
from nkigym.search.types import Evaluation, SearchConfig, SearchNode

WORKLOAD = LHS_T_RHS


def test_describe_actions_adds_semantic_loop_and_buffer_context() -> None:
    """Descriptions expose operation scopes and named buffer fields."""
    environment = KernelMDP(WORKLOAD.f_nkigym, WORKLOAD.input_specs, TRANSFORMS)
    state = environment.reset()
    descriptions = [item.description for item in describe_actions(state, environment.legal_actions(state))]
    assert any(
        "Split: split loop" in text and "nc_matmul" in text and "factors (2, 8)" in text for text in descriptions
    )
    assert any("BufferLayout: set psum_prod.list_len=16" in text for text in descriptions)


def test_state_fingerprint_is_render_stable_and_transform_sensitive() -> None:
    """Equivalent canonical builds match while a rendered transform differs."""
    environment = KernelMDP(WORKLOAD.f_nkigym, WORKLOAD.input_specs, TRANSFORMS)
    first = environment.reset()
    second = environment.reset()
    split_action = next(
        action for action in environment.legal_actions(first) if "target_nid=2, factors=(2, 8)" in repr(action[1])
    )
    transformed = environment.step(first, split_action)
    assert state_fingerprint(first) == state_fingerprint(second)
    assert state_fingerprint(first) != state_fingerprint(transformed)


def test_observation_contains_current_profile_and_bounded_history() -> None:
    """The policy receives current metrics, recent measurements, and legal actions."""
    environment = KernelMDP(WORKLOAD.f_nkigym, WORKLOAD.input_specs, TRANSFORMS)
    state = environment.reset()
    nodes = [
        SearchNode(
            node_id=index,
            state=state,
            parent_id=None if index == 0 else index - 1,
            action_id=None,
            action_description=f"state {index}",
            rationale=f"decision {index}",
            evaluation=Evaluation(
                score=float(index),
                metrics={"mfu_percent": float(index), "tensor_engine_active_percent": 90.0},
                message=f"MFU={index}",
            ),
        )
        for index in range(30)
    ]
    actions = describe_actions(state, environment.legal_actions(state))

    observation = format_observation(
        state=state,
        nodes=nodes,
        actions=actions,
        config=SearchConfig(cache_dir=Path("/tmp/unused"), max_iterations=100, workload_guidance="test guidance"),
    )

    assert "best measured state: N029 score=29.0000" in observation
    assert "# Current Neuron Profile" in observation
    assert "- tensor_engine_active_percent: 90.0" in observation
    assert "- 6 earlier states omitted" in observation
    assert "N006" in observation
    assert "N005" not in observation
    assert '"kind":"apply"' in observation
    assert '"kind":"finish"' in observation
    assert '"kind":"checkout"' not in observation
    assert '"kind":"evaluate"' not in observation
