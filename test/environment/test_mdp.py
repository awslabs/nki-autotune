"""Tests for :class:`nkigym.environment.KernelMDP`."""

from __future__ import annotations

import random
from test.environment._fixtures import INPUT_SPECS, f_matmul

import pytest

from nkigym.environment import KernelMDP
from nkigym.ir import KernelIR, build_initial_ir
from nkigym.transforms import Fuse, Reorder, Split, SplitOption, TransformLegalityError


def _trees_structurally_equal(a: KernelIR, b: KernelIR) -> bool:
    """Return ``True`` iff the two trees have identical node ids and per-node payloads.

    ``KernelIR`` has no ``__eq__``; we compare by walking ``preorder`` and
    comparing ``data(nid)`` for every node id.
    """
    a_nids = list(a.tree.preorder())
    b_nids = list(b.tree.preorder())
    equal = a_nids == b_nids
    if equal:
        for nid in a_nids:
            if a.tree.data(nid) != b.tree.data(nid):
                equal = False
                break
    return equal


def test_reset_returns_canonical_ir() -> None:
    """``reset`` must return a tree structurally identical to ``build_initial_ir``."""
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse()])
    state = env.reset()
    expected = build_initial_ir(f_matmul, INPUT_SPECS)
    assert _trees_structurally_equal(state, expected)


def test_legal_actions_non_empty_on_reset() -> None:
    """``legal_actions`` on the canonical IR must have at least one entry."""
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse()])
    state = env.reset()
    actions = env.legal_actions(state)
    assert len(actions) > 0


def test_legal_actions_membership_matches_per_transform_analyze() -> None:
    """Every option from each transform's ``analyze`` appears exactly once,
    paired with the env's instance of that transform (compared by identity)."""
    split = Split()
    fuse = Fuse()
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[split, fuse])
    state = env.reset()
    actions = env.legal_actions(state)

    expected_split = split.analyze(state)
    expected_fuse = fuse.analyze(state)
    assert len(actions) == len(expected_split) + len(expected_fuse)

    """Group actions by transform identity, compare option lists."""
    split_options = [opt for tr, opt in actions if tr is split]
    fuse_options = [opt for tr, opt in actions if tr is fuse]
    assert split_options == expected_split
    assert fuse_options == expected_fuse


def test_legal_actions_with_only_split() -> None:
    """An env with only ``Split`` returns only Split's analyze options."""
    split = Split()
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[split])
    state = env.reset()
    actions = env.legal_actions(state)
    assert all(tr is split for tr, _ in actions)
    assert [opt for _, opt in actions] == split.analyze(state)


def _snapshot_tree(state: KernelIR) -> tuple[int, dict[int, object]]:
    """Capture a structural snapshot of ``state.tree`` for mutation checks."""
    return state.tree.num_nodes, {nid: state.tree.data(nid) for nid in state.tree.preorder()}


def test_step_does_not_mutate_input_state() -> None:
    """``step`` must leave the input state's tree byte-identical."""
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse()])
    state = env.reset()
    before = _snapshot_tree(state)
    actions = env.legal_actions(state)
    env.step(state, actions[0])
    after = _snapshot_tree(state)
    assert before == after


def test_step_matches_direct_transform_apply() -> None:
    """``step(state, (transform, option))`` and ``transform.apply(state, option)``
    must produce structurally identical trees."""
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse()])
    state = env.reset()
    actions = env.legal_actions(state)
    """Pick a Split action so we exercise the dispatch."""
    split_actions = [(tr, opt) for tr, opt in actions if isinstance(tr, Split)]
    assert split_actions, "expected at least one Split action on the canonical IR"
    transform, option = split_actions[0]
    via_step = env.step(state, (transform, option))
    via_apply = transform.apply(state, option)
    assert _trees_structurally_equal(via_step, via_apply)


def test_step_raises_on_illegal_action() -> None:
    """An action whose option is illegal for the current state must raise loudly."""
    split = Split()
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[split])
    state = env.reset()
    """Construct an obviously illegal SplitOption: factors product != trip,
    and target_nid is the root (a RootNode, not a ForNode)."""
    illegal = SplitOption(target_nid=state.tree.root, factors=(2, 2))
    with pytest.raises(TransformLegalityError):
        env.step(state, (split, illegal))


def test_is_terminal_false_on_canonical_ir() -> None:
    """The canonical IR has plenty of legal actions; ``is_terminal`` must be False."""
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse()])
    state = env.reset()
    assert env.is_terminal(state) is False


def test_is_terminal_true_with_no_transforms() -> None:
    """An env with an empty transform list has no legal actions, so every state is terminal."""
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[])
    state = env.reset()
    assert env.is_terminal(state) is True


def test_mdp_with_reorder_random_rollout() -> None:
    """A random rollout with [Split, Fuse, Reorder] runs without raising."""
    rng = random.Random(42)
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse(), Reorder()])
    state = env.reset()

    """5 random steps; legality is checked inside step (loud failure on illegal)."""
    for _ in range(5):
        actions = env.legal_actions(state)
        if not actions:
            break
        action = rng.choice(actions)
        state = env.step(state, action)
        """Confirm the resulting state is itself legal: legal_actions runs analyze
        across all transforms, which exercises tree-walk over the new IR."""
        env.legal_actions(state)


def test_mdp_legal_actions_includes_reorder_options_on_canonical() -> None:
    """``legal_actions`` on the canonical IR must include at least one Reorder action."""
    reorder = Reorder()
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse(), reorder])
    state = env.reset()
    actions = env.legal_actions(state)
    reorder_actions = [(tr, opt) for tr, opt in actions if tr is reorder]
    assert reorder_actions, "expected at least one Reorder action on canonical IR"
