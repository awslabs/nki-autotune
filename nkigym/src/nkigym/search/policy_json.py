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
next choice depends on its measured result. Optimize complete schedules rather
than treating transforms as independent hill-climbing steps. A small regression
or resource failure after an enabling transform does not invalidate a dependent
sequence. Continue from that child when it exposes the locality, factorization,
placement, compaction, or pipelining needed to test the hypothesis. Never write
source or invent an action or node.

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
- OnlineFusion replaces compatible materialized reductions and consumers with a
  chunked online recurrence. Start with the largest listed legal chunk to
  reduce recurrence overhead, and apply compatible matches at the same chunk
  size. For attention, apply the first match and its compatible completion
  consecutively before CopyPropagation, pointwise transforms, or any other
  cleanup. Intermediate resource failures are expected while the recurrence
  is still materialized as separate full-tensor sweeps. Keep the largest-chunk
  branch until the entire producer-consumer body, not only its PSUM operations,
  is row-local and compacted. Reduce the chunk only after that complete
  localized candidate still fails or no legal path to it remains.

Build structural schedules before micro-optimizing them:
1. Complete all compatible OnlineFusion matches at the largest chunk.
2. Then apply pointwise fusion, broadcast decomposition, copy propagation, and
   common-subexpression elimination exposed by the online recurrence.
3. Expose reduction and matrix tiling with Split, RFactor, and Reorder.
4. Use CodeMotion to gather consecutive sweeps over the same tile axis under
   one shared loop. Move the complete producer-consumer chain in dependency
   order; an intermediate state may regress or fail before the chain is whole.
5. Apply BufferPlacement and BufferCompaction after lifetimes become local.
6. Pipeline the complete sibling body, then consider final splits or batch
   permutation. Do not pipeline fragmented full-tensor sweeps.

For online attention, a complete row body includes the score initializer and
matmul; scaling, max and sum recurrence; probability transpose; output
initializer and matmul; output-chunk copy; carry load, update, and store; and
row-max and row-sum state copies. A loop containing only score/softmax work or
only output/carry work is partial while another separate d1 row sweep remains.
Build this schedule as one dependent sequence:
- After both OnlineFusion matches complete, apply available pointwise fusion,
  broadcast-subtract decomposition, copy propagation, and
  common-subexpression elimination before factorization or loop motion.
- Factor both row reductions with an inner factor of 512: select exactly
  (16, 512) for an 8192-wide chunk or (8, 512) for a 4096-wide chunk. Use the
  same factors for max and sum; smaller inner factors underutilize the
  reduction tile. Place and compact the resulting partial and folded buffers
  before rejecting an RFactor because its immediate profile regressed.
- Reorder the output matmul's d2/d1 loops so d1 is outer, then use CodeMotion
  to move its initializer, matmul, copy, carry operations, and the complete
  softmax chain into the same d1 loop in dependency order. The reorder can
  temporarily distort MFU even when latency improves; finish merging the chain
  before judging that branch.
- When repeatedly moving blocks to child slot 0 of that d1 loop, select them
  in this exact order: probability transpose; row-sum state update; stage-1
  correction; row-sum partial fold; exponent/sum reduction; negative current
  max; row-max combine; row-max partial fold; scaled-score/max reduction and
  any still-present score copy; score matmul; score initializer. Then append
  the output initializer, output matmul, carry load, output state update,
  carry store, row-max state copy, and row-sum state copy in dependency order.
  This order is performance-critical because SoftwarePipeline requires
  contiguous stages; in particular, stage-1 correction must follow the
  negative-max, exponent/sum reduction, and row-sum fold blocks.
- Complete those row-body moves first. Then split and nest both score
  initializer/reduction paths as (16, 512), split the output matmul's 64-trip
  reduction loop directly into (4, 16), and RFactor the new outer trip-4 loop.
  Do not apply output RFactor earlier and do not replace the direct split with
  recursively nested factors: the outer reduction loop must remain available
  to RFactor. Continue through its fold and dependent copy cleanup before
  judging the branch.
- Place every on-chip buffer whose legal lifetime-safe scope was exposed,
  including key/value chunks, output and deferred epilogue buffers, and
  RFactor partials. Compact every legal row-local or tile-local buffer before
  identity-initializer elimination and pipelining.
- After identity-initializer elimination, move the probability transpose to
  child slot 5, immediately after exponentiation and sum reduction. Then move
  the row-sum state copy to child slot 9, after the row-sum update and before
  the output initializer. The resulting order must be score
  matmul/reduction, max fold, max combine, negative max, exponent/sum
  reduction, probability transpose, sum fold, stage-1 correction, sum update,
  sum-state copy, output initializer, output RFactor body, carry load, output
  update, carry store, max-state copy.
- Never choose SoftwarePipeline while separate d1 sweeps for any of the above
  operations remain. Pipeline only the unified body after placement,
  compaction, and identity-initializer elimination. For the 16-child unified
  attention body, use stages exactly
  (0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2); do not substitute a
  two-stage assignment.
- Keep sbuf_probability_t.list_len=1 through this schedule. If an earlier
  BufferLayout action changed it, restore list_len=1 after compaction and
  before the final probability-transpose split. BatchPermutation requires its
  transposed on-chip buffer to have list_len=1.
- After pipelining the unified body and confirming list_len=1, split the
  64-trip probability-transpose loop directly into (32, 2), then apply
  BatchPermutation to the new inner trip-2 loop. Do not split or relayout the
  value load, output matmul, or output buffers to align them with this
  transpose. Do not permute the unsplit trip-64 loop. A split regression alone
  does not disprove this sequence because the subsequent BatchPermutation
  provides its principal gain.

All legal transforms preserve behavior, but compilation can fail from resource
pressure and performance can regress. A compiler error naming an oversized
buffer is evidence to shorten its lifetime or physical extent on the same
branch, not to abandon the schedule immediately. BufferLayout can make tile
storage legal, but it does not replace loop locality and compaction. Treat
automatic Neuron profiles as feedback on complete hypotheses, compare the full
trace, and revisit only when the hypothesis is disproved or a better branch is
clear. Finish only when neither an active action nor an unexplored branch is
worth another profile.

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
