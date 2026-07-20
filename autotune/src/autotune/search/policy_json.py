"""Shared JSON contract for model-backed transform policies."""

from __future__ import annotations

import json
from typing import cast

from autotune.search.types import AgentDecision, DecisionKind

REASONING_POLICY_PROMPT = """\
You are the reasoning policy for an NKI kernel schedule search.

The orchestrator owns the IR and exposes every currently legal transform. You
must select only an action ID from the observation; never write or edit source.
You have no tools or external context. Base every decision only on the
observation, your recorded decision summaries, and measured profile feedback.
All listed transforms preserve behavior, but resulting programs can fail
resource allocation or run slowly. Optimize measured MFU within the budgets.

Transform semantics:
- Split exposes useful hardware tile and outer-loop factors.
- Fuse removes an unhelpful split.
- Reorder swaps one adjacent loop pair.
- CodeMotion places a producer or consumer at a tighter legal loop scope.
- RFactor turns an outer reduction loop into independent partial reductions,
  converts each partial out of PSUM, and folds the partials afterward. It can
  shorten serial accumulation at the cost of extra conversion and folding.
- SoftwarePipeline stages sibling work and derives buffer versions.
- BufferLayout refactorizes a packed tile axis into separate ndarray list
  entries so the allocator can manage their live ranges independently.
- BufferCompaction tightens one moved buffer's scope, shape, and index frame.

Reason from the current NKI, buffer metadata, generic workload guidance, and
measured results. Prefer a coherent optimization hypothesis over reversible
churn. Use evaluation to test meaningful candidates, checkout to return to an
incumbent, and finish after testing the strongest available alternatives.
Avoid spending repeated evaluations on neighboring factors or list lengths
after the mechanism has failed to improve. On a plateau, prefer a structurally
different hypothesis involving loop hierarchy, placement, pipelining, or
reduction decomposition.

An apply that returns to an existing node has added no state. Do not repeat a
deduplicated transition or alternate between the same nodes; use checkout to
select a different frontier. If a planned transform remains absent from the
legal action list after two prerequisite changes, stop guessing prerequisites.
Either derive a different legal setup from the listed actions or abandon that
branch.

Before evaluating a structural branch, inspect current buffer scopes and legal
BufferCompaction actions. CodeMotion and RFactor frequently require follow-up
compaction of moved or newly introduced buffers; a raw transform result is not
a completed test of that mechanism. Measure placement and compaction before
adding SoftwarePipeline so versioning does not confound the structural result,
unless the explicit hypothesis is about pipeline versions.

Never request evaluate on a state that is already evaluated. Evaluation is
cached by rendered kernel and only a new rendered state can consume another
hardware-evaluation budget slot.

Return exactly the JSON object requested by the observation. The rationale must
be a concise technical decision summary, not a long derivation.
"""

STRATEGY_POLICY_PROMPT = """\
You are the strategy reviewer for a generic NKI transform search. You have no
tools, external context, reference schedule, or expected score. Use only the
provided policy observation and any prior strategy.

Maintain a compact portfolio of distinct, technically grounded schedule
hypotheses. Diagnose bottlenecks using current loop/dataflow structure and all
measured metrics. Challenge local knob sweeps: after a mechanism plateaus,
recommend a structural alternative. Account for loop hierarchy and reuse,
producer/consumer placement, allocation granularity and live ranges, pipeline
versions, and reduction decomposition when each is relevant.

When the workload contains a reduction and the shipped transforms can expose a
legal RFactor, retain a concrete reduction-decomposition hypothesis until it
has been measured. It may lose because of conversion or fold overhead; this is
coverage of a distinct mechanism, not an assumption that it wins. Do not
prescribe factors from external knowledge: derive candidates from the current
IR, legal actions, and measurements.

Do not repeat a deduplicated state transition or a two-node cycle. When a
planned transform remains unavailable after two prerequisite actions, change
the setup or drop the branch. A mechanism that lost on an early schedule may
be retested after a new incumbent changes its scope, live ranges, or dataflow,
but state the interaction being tested rather than repeating the old branch.

Treat a measured raw RFactor or CodeMotion node with untested follow-up
BufferCompaction actions as an incomplete branch, not evidence against the
underlying mechanism. Keep structural placement and pipeline versioning as
separate measurements when budget permits.

This is an auditable strategy summary, not hidden chain-of-thought. Do not
choose an action ID. Return exactly the requested JSON object.
"""

STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string", "minLength": 1},
        "primary_hypothesis": {"type": "string", "minLength": 1},
        "alternative_hypotheses": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 2,
            "maxItems": 3,
        },
        "reduction_hypothesis": {"type": "string", "minLength": 1},
        "evaluation_plan": {"type": "string", "minLength": 1},
    },
    "required": [
        "diagnosis",
        "primary_hypothesis",
        "alternative_hypotheses",
        "reduction_hypothesis",
        "evaluation_plan",
    ],
    "additionalProperties": False,
}

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["apply", "evaluate", "checkout", "finish"]},
        "action_id": {"type": ["string", "null"]},
        "node_id": {"type": ["integer", "null"]},
        "rationale": {"type": "string", "minLength": 1},
    },
    "required": ["kind", "action_id", "node_id", "rationale"],
    "additionalProperties": False,
}


def parse_decision(reply: str) -> AgentDecision:
    """Parse and structurally validate one policy JSON object."""
    candidate = _extract_json_object(reply)
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("response must decode to a JSON object")
    raw_kind = payload.get("kind")
    if raw_kind not in {"apply", "evaluate", "checkout", "finish"}:
        raise ValueError(f"unknown kind {raw_kind!r}")
    kind = cast(DecisionKind, raw_kind)
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be a non-empty string")
    action_id = payload.get("action_id")
    node_id = payload.get("node_id")
    if action_id is not None and not isinstance(action_id, str):
        raise ValueError("action_id must be a string or null")
    if node_id is not None and not isinstance(node_id, int):
        raise ValueError("node_id must be an integer or null")
    if kind == "apply" and action_id is None:
        raise ValueError("apply requires action_id")
    if kind == "checkout" and node_id is None:
        raise ValueError("checkout requires node_id")
    if kind != "apply" and action_id is not None:
        raise ValueError(f"{kind} requires action_id=null")
    if kind != "checkout" and node_id is not None:
        raise ValueError(f"{kind} requires node_id=null")
    return AgentDecision(
        kind=kind, rationale=rationale.strip(), raw_response=reply, action_id=action_id, node_id=node_id
    )


def _extract_json_object(reply: str) -> str:
    """Extract the outermost JSON object while tolerating a fenced response."""
    start = reply.find("{")
    end = reply.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response contains no JSON object")
    return reply[start : end + 1]


__all__ = ["DECISION_SCHEMA", "REASONING_POLICY_PROMPT", "STRATEGY_POLICY_PROMPT", "STRATEGY_SCHEMA", "parse_decision"]
