"""Tests for semantic search observations."""

from pathlib import Path

from autotune.search.observation import (
    describe_actions,
    format_observation,
    search_state_fingerprint,
    state_fingerprint,
)
from autotune.search.types import Evaluation, SearchConfig, SearchEvent, SearchNode
from examples.matmul_lhsT_rhs import INPUT_SPECS, TRANSFORMS, f_nkigym
from nkigym.environment import KernelMDP


def test_describe_actions_adds_semantic_loop_and_buffer_context() -> None:
    """Descriptions expose operation scopes and named buffer fields."""
    environment = KernelMDP(f_nkigym, INPUT_SPECS, TRANSFORMS)
    state = environment.reset()
    descriptions = [item.description for item in describe_actions(state, environment.legal_actions(state))]
    assert any(
        "Split: split loop" in text and "nc_matmul" in text and "factors (2, 8)" in text for text in descriptions
    )
    assert any("BufferLayout: set psum_prod.list_len=16" in text for text in descriptions)


def test_state_fingerprint_is_render_stable_and_transform_sensitive() -> None:
    """Equivalent canonical builds match while a real transform changes the digest."""
    environment = KernelMDP(f_nkigym, INPUT_SPECS, TRANSFORMS)
    first = environment.reset()
    second = environment.reset()
    split_action = next(
        action for action in environment.legal_actions(first) if "target_nid=2, factors=(2, 8)" in repr(action[1])
    )
    transformed = environment.step(first, split_action)
    assert state_fingerprint(first) == state_fingerprint(second)
    assert state_fingerprint(first) != state_fingerprint(transformed)


def test_search_fingerprint_includes_future_action_surface() -> None:
    """Render-equivalent states remain distinct when future choices differ."""
    environment = KernelMDP(f_nkigym, INPUT_SPECS, TRANSFORMS)
    state = environment.reset()
    actions = environment.legal_actions(state)
    assert search_state_fingerprint(state, actions[:1]) != search_state_fingerprint(state, actions[:2])


def test_observation_prioritizes_leaders_and_bounds_stale_history() -> None:
    """Long searches retain useful leaders, active context, and recent events."""
    environment = KernelMDP(f_nkigym, INPUT_SPECS, TRANSFORMS)
    state = environment.reset()
    nodes = [
        SearchNode(
            node_id=index,
            state=state,
            fingerprint=str(index),
            parent_id=None if index == 0 else 0,
            action_id=None,
            action_description=f"state {index}",
            evaluation=(
                Evaluation(score=float(index), metrics={"score": index}, message=f"score={index}")
                if index < 20
                else None
            ),
        )
        for index in range(60)
    ]
    events = [
        SearchEvent(
            decision=index,
            active_before=0,
            active_after=0,
            kind="checkout",
            action_id=None,
            node_id=0,
            rationale=f"decision {index}",
            raw_response="{}",
        )
        for index in range(1, 61)
    ]

    observation = format_observation(
        state=state,
        nodes=nodes,
        active_node_id=59,
        actions=[],
        config=SearchConfig(
            cache_dir=Path("/tmp/unused"),
            resume_dir=None,
            max_transforms=100,
            max_evaluations=100,
            min_evaluations=0,
            max_decisions=100,
            workload_guidance="test",
        ),
        transforms_applied=0,
        evaluations_run=20,
        events=events,
    )

    assert observation.index("N019") < observation.index("N018")
    assert "20 unique rendered hardware evaluations" in observation
    assert "27 older off-path states omitted" in observation
    assert "12 earlier decisions are retained in events.jsonl" in observation
    assert "D013" in observation
    assert "D012" not in observation
