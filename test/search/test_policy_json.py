"""Tests for strict model decision parsing."""

import pytest

from autotune.search.policy_json import parse_decision


def test_parse_decision_accepts_json_fence_and_fields() -> None:
    """A fenced but otherwise valid response is normalized."""
    decision = parse_decision(
        '```json\n{"kind":"apply","action_id":"A012",' '"node_id":null,"rationale":"tile N first"}\n```'
    )
    assert decision.kind == "apply"
    assert decision.action_id == "A012"
    assert decision.rationale == "tile N first"


def test_parse_decision_rejects_missing_apply_action() -> None:
    """Apply without an action identifier fails loudly."""
    with pytest.raises(ValueError, match="apply requires action_id"):
        parse_decision('{"kind":"apply","action_id":null,' '"node_id":null,"rationale":"invalid"}')


def test_parse_decision_rejects_irrelevant_fields() -> None:
    """Non-null fields that do not belong to the selected operation fail."""
    with pytest.raises(ValueError, match="finish requires action_id=null"):
        parse_decision('{"kind":"finish","action_id":"A000",' '"node_id":null,"rationale":"invalid"}')
