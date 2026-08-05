"""Tests for :class:`nkigym.environment.KernelMDP`."""

from __future__ import annotations

from test.environment._fixtures import INPUT_SPECS, f_matmul

from nkigym.environment import KernelMDP
from nkigym.transforms import Fuse, Split


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
