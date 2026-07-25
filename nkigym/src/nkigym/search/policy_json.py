"""JSON contract for the next-transform reasoning policy."""

from __future__ import annotations

import json
from typing import cast

from nkigym.search.types import AgentDecision, DecisionKind

REASONING_POLICY_PROMPT = """\
You choose the next legal transform in a linear NKI kernel refinement loop.

The orchestrator owns the IR. It profiles the canonical kernel on Neuron before
the first turn and profiles every transformed kernel before the next turn. Use
only the current NKI, current profile feedback, prior measured steps, generic
workload guidance, buffer metadata, and legal actions in the observation.
Select exactly one listed action ID or finish. Never write source or invent an
action.

Transform semantics:
- Split exposes hardware tile and outer-loop factors.
- Fuse removes an unhelpful split.
- Reorder swaps one adjacent loop pair.
- CodeMotion changes producer or consumer placement.
- RFactor creates independent partial reductions and a later fold.
- SoftwarePipeline stages sibling work and derives buffer versions.
- BufferLayout changes packed tiles into separate ndarray list entries.
- BufferCompaction tightens a moved buffer's scope, shape, and index frame.

All legal transforms preserve behavior, but compilation can fail from resource
pressure and performance can regress. Treat the next automatic Neuron profile
as direct feedback on the selected transform. Choose a transform that tests one
concrete hypothesis based on the current metrics and structure. Finish only
when no listed transform is worth another profile.

Return exactly the JSON object requested by the observation. Keep the rationale
concise and technical.
"""

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["apply", "finish"]},
        "action_id": {"type": ["string", "null"]},
        "rationale": {"type": "string", "minLength": 1},
    },
    "required": ["kind", "action_id", "rationale"],
    "additionalProperties": False,
}


def parse_decision(reply: str) -> AgentDecision:
    """Parse and validate one apply-or-finish policy response."""
    candidate = _extract_json_object(reply)
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("response must decode to a JSON object")
    raw_kind = payload.get("kind")
    if raw_kind not in {"apply", "finish"}:
        raise ValueError(f"unknown kind {raw_kind!r}")
    kind = cast(DecisionKind, raw_kind)
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be a non-empty string")
    action_id = payload.get("action_id")
    if action_id is not None and not isinstance(action_id, str):
        raise ValueError("action_id must be a string or null")
    if kind == "apply" and action_id is None:
        raise ValueError("apply requires action_id")
    if kind == "finish" and action_id is not None:
        raise ValueError("finish requires action_id=null")
    return AgentDecision(kind=kind, rationale=rationale.strip(), raw_response=reply, action_id=action_id)


def _extract_json_object(reply: str) -> str:
    """Extract the outermost JSON object from a plain or fenced response."""
    start = reply.find("{")
    end = reply.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response contains no JSON object")
    return reply[start : end + 1]


__all__ = ["DECISION_SCHEMA", "REASONING_POLICY_PROMPT", "parse_decision"]
