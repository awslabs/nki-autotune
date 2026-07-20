"""Validation helpers for persisted search transcript artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from autotune.search.types import DecisionKind, Evaluation, EvaluationMetric, SearchEvent


def parse_event(line: str) -> SearchEvent:
    """Validate one serialized search event."""
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("resume event must be a JSON object")
    decision = payload.get("decision")
    active_before = payload.get("active_before")
    active_after = payload.get("active_after")
    raw_kind = payload.get("kind")
    action_id = payload.get("action_id")
    node_id = payload.get("node_id")
    rationale = payload.get("rationale")
    raw_response = payload.get("raw_response")
    if not isinstance(decision, int) or not isinstance(active_before, int) or not isinstance(active_after, int):
        raise ValueError("resume event decision and active node IDs must be integers")
    if raw_kind not in {"apply", "evaluate", "checkout", "finish"}:
        raise ValueError(f"resume event has unknown kind {raw_kind!r}")
    if action_id is not None and not isinstance(action_id, str):
        raise ValueError("resume event action_id must be a string or null")
    if node_id is not None and not isinstance(node_id, int):
        raise ValueError("resume event node_id must be an integer or null")
    if not isinstance(rationale, str) or not isinstance(raw_response, str):
        raise ValueError("resume event rationale and raw_response must be strings")
    kind = cast(DecisionKind, raw_kind)
    _validate_operation_fields(kind, action_id, node_id)
    return SearchEvent(
        decision=decision,
        active_before=active_before,
        active_after=active_after,
        kind=kind,
        action_id=action_id,
        node_id=node_id,
        rationale=rationale,
        raw_response=raw_response,
    )


def parse_evaluation(path: Path) -> Evaluation:
    """Validate one cached evaluation artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation must be a JSON object: {path}")
    score = payload.get("score")
    raw_metrics = payload.get("metrics")
    message = payload.get("message")
    if score is not None and not isinstance(score, (int, float)):
        raise ValueError(f"evaluation score must be numeric or null: {path}")
    if not isinstance(raw_metrics, dict) or not isinstance(message, str):
        raise ValueError(f"evaluation lacks metrics or message: {path}")
    metrics: dict[str, EvaluationMetric] = {}
    for name, value in raw_metrics.items():
        if not isinstance(name, str) or not isinstance(value, (float, int, str, bool, type(None))):
            raise ValueError(f"evaluation metric has unsupported type: {path}")
        metrics[name] = value
    return Evaluation(score=float(score) if score is not None else None, metrics=metrics, message=message)


def resume_evaluation_path(resume_dir: Path, node_id: int) -> Path:
    """Return one evaluation artifact path in a prior run."""
    return resume_dir / "nodes" / f"node_{node_id:03d}" / "evaluation.json"


def _validate_operation_fields(kind: DecisionKind, action_id: str | None, node_id: int | None) -> None:
    """Validate operation-specific transcript fields."""
    if kind == "apply" and (action_id is None or node_id is not None):
        raise ValueError("resume apply event requires action_id and node_id=null")
    if kind == "checkout" and (node_id is None or action_id is not None):
        raise ValueError("resume checkout event requires node_id and action_id=null")
    if kind in {"evaluate", "finish"} and (action_id is not None or node_id is not None):
        raise ValueError(f"resume {kind} event requires action_id=null and node_id=null")


__all__ = ["parse_evaluation", "parse_event", "resume_evaluation_path"]
