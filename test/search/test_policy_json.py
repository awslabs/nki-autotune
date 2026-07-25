"""Tests for strict next-transform decision parsing."""

import pytest

from nkigym.search.policy_json import parse_decision


def test_parse_decision_accepts_json_fence_and_fields() -> None:
    """A fenced valid response is normalized."""
    decision = parse_decision('```json\n{"kind":"apply","action_id":"A012","rationale":"tile N first"}\n```')
    assert decision.kind == "apply"
    assert decision.action_id == "A012"
    assert decision.rationale == "tile N first"


def test_parse_decision_rejects_missing_apply_action() -> None:
    """Apply without an action identifier fails loudly."""
    with pytest.raises(ValueError, match="apply requires action_id"):
        parse_decision('{"kind":"apply","action_id":null,"rationale":"invalid"}')


def test_parse_decision_rejects_finish_action() -> None:
    """Finish cannot carry an action identifier."""
    with pytest.raises(ValueError, match="finish requires action_id=null"):
        parse_decision('{"kind":"finish","action_id":"A000","rationale":"invalid"}')


def test_parse_decision_rejects_removed_graph_operations() -> None:
    """Evaluate and checkout are not policy decisions in linear refinement."""
    with pytest.raises(ValueError, match="unknown kind 'evaluate'"):
        parse_decision('{"kind":"evaluate","action_id":null,"rationale":"invalid"}')
