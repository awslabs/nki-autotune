"""JSON contract for branching transform reasoning."""

from __future__ import annotations

import json
from typing import cast

from nkigym.search.types import MAX_TRANSFORMS_PER_REASONING_STEP, AgentDecision, DecisionKind

REASONING_POLICY_PROMPT = """\
You choose branches and legal transforms in a measured NKI kernel search tree.

The orchestrator owns the IR. It profiles the canonical kernel on Neuron before
the first turn and profiles every transformed kernel before the next turn. Each
observation contains the complete measured trace and the full details for one
active node. You may apply one to three ordered, compatible listed actions from
that active node, revisit another listed branchable node to inspect and branch from
it on the next turn, or finish. Revisit earlier nodes when an irreversible
transform blocked a better sequence; no previous decision is final.

Every selected transform remains a separate atomic action, and the orchestrator
profiles every intermediate state in order. There is no profile feedback
between actions selected in the same response, so select one action when the
next choice depends on its measured result. Never write source or invent an
action or node.

Transform semantics:
- Split exposes hardware tile and outer-loop factors.
- Fuse removes an unhelpful split.
- Reorder swaps one adjacent loop pair.
- CodeMotion changes producer or consumer placement.
- RFactor creates independent partial reductions and a later fold.
- SoftwarePipeline stages sibling work and derives buffer versions.
- BufferLayout changes packed tiles into separate ndarray list entries.
- BufferPlacement moves one on-chip declaration to its lifetime-safe LCA scope.
- BufferCompaction shrinks one buffer's logical shape and normalizes its index frame.

All legal transforms preserve behavior, but compilation can fail from resource
pressure and performance can regress. Treat the next automatic Neuron profiles
as direct feedback on the selected sequence. Compare the full trace, revisit
promising parents when appropriate, and choose the shortest compatible sequence
that tests one concrete hypothesis. Finish only when neither an active action
nor an unexplored branch is worth another profile.

Return exactly the JSON object requested by the observation. Keep the rationale
concise and technical.
"""

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["apply", "revisit", "finish"]},
        "base_node_id": {"type": ["integer", "null"], "minimum": 0},
        "action_ids": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": MAX_TRANSFORMS_PER_REASONING_STEP,
            "uniqueItems": True,
        },
        "rationale": {"type": "string", "minLength": 1},
    },
    "required": ["kind", "base_node_id", "action_ids", "rationale"],
    "additionalProperties": False,
}


def parse_decision(reply: str) -> AgentDecision:
    """Parse and validate one apply, revisit, or finish response."""
    candidate = _extract_json_object(reply)
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("response must decode to a JSON object")
    required_fields = {"kind", "base_node_id", "action_ids", "rationale"}
    missing_fields = sorted(required_fields - payload.keys())
    if missing_fields:
        raise ValueError(f"response is missing required fields {missing_fields}")
    extra_fields = sorted(payload.keys() - required_fields)
    if extra_fields:
        raise ValueError(f"response contains unexpected fields {extra_fields}")
    raw_kind = payload.get("kind")
    if raw_kind not in {"apply", "revisit", "finish"}:
        raise ValueError(f"unknown kind {raw_kind!r}")
    kind = cast(DecisionKind, raw_kind)
    base_node_id = payload.get("base_node_id")
    if base_node_id is not None and (
        isinstance(base_node_id, bool) or not isinstance(base_node_id, int) or base_node_id < 0
    ):
        raise ValueError("base_node_id must be null or a non-negative integer")
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be a non-empty string")
    raw_action_ids = payload.get("action_ids")
    if not isinstance(raw_action_ids, list) or any(not isinstance(item, str) for item in raw_action_ids):
        raise ValueError("action_ids must be an array of strings")
    action_ids = tuple(raw_action_ids)
    if len(action_ids) > MAX_TRANSFORMS_PER_REASONING_STEP:
        raise ValueError(f"at most {MAX_TRANSFORMS_PER_REASONING_STEP} action_ids are allowed")
    if len(set(action_ids)) != len(action_ids):
        raise ValueError("action_ids must not contain duplicates")
    if kind == "apply" and (base_node_id is None or not action_ids):
        raise ValueError("apply requires base_node_id and at least one action_id")
    if kind == "revisit" and (base_node_id is None or action_ids):
        raise ValueError("revisit requires base_node_id and action_ids=[]")
    if kind == "finish" and (base_node_id is not None or action_ids):
        raise ValueError("finish requires base_node_id=null and action_ids=[]")
    return AgentDecision(
        kind=kind, base_node_id=base_node_id, rationale=rationale.strip(), raw_response=reply, action_ids=action_ids
    )


def _extract_json_object(reply: str) -> str:
    """Extract the outermost JSON object from a plain or fenced response."""
    start = reply.find("{")
    end = reply.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response contains no JSON object")
    return reply[start : end + 1]


__all__ = ["DECISION_SCHEMA", "REASONING_POLICY_PROMPT", "parse_decision"]
