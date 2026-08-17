"""Public types for iterative schedule refinement."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

from nkigym.ir import KernelIR
from nkigym.ir.tree import ISANode
from nkigym.profile.types import ProfileMetrics
from nkigym.transforms import Transform, TransformOption

Action = tuple[Transform[Any], TransformOption]
_TRANSFORM_PRIORITY = {
    "CommonSubexpressionElimination": 100,
    "SetFirstWriteOverwrite": 96,
    "EliminateIdentityInitializer": 95,
    "FusePointwise": 90,
    "CopyPropagation": 85,
    "EliminateDeadProducer": 84,
    "DecomposeBroadcastSubtract": 80,
    "OnlineFusion": 75,
    "CodeMotion": 70,
    "BufferPlacement": 65,
    "BufferCompaction": 60,
    "RFactor": 55,
    "SoftwarePipeline": 50,
    "BufferLayout": 45,
    "Fuse": 40,
    "Split": 35,
    "Reorder": 30,
    "BatchPermutation": 25,
    "TransposeThroughLoad": 20,
    "TransposeThroughMatmul": 20,
    "TransposeThroughTensorCopy": 20,
    "TransposePair": 15,
}


@dataclass(frozen=True)
class PolicyContext:
    """State exposed to a policy before one refinement step."""

    state: KernelIR
    transforms: tuple[Transform[Any], ...]
    legal_actions: tuple[Action, ...]
    evaluations: tuple[ProfileMetrics, ...]
    max_transforms: int


class Policy:
    """Deterministically refine legal actions using generic schedule heuristics."""

    def __init__(self) -> None:
        """Initialize the set of options already selected by this policy."""
        self._selected: set[tuple[str, str]] = set()

    def select_actions(self, context: PolicyContext) -> tuple[Action, ...]:
        """Return an ordered transform sequence, or an empty tuple to finish."""
        budget = context.max_transforms
        if len(context.evaluations) > 1 and context.evaluations[-1].latency_ms > context.evaluations[-2].latency_ms:
            budget = 1
        legal_actions = context.legal_actions
        state = context.state
        selected: list[Action] = []
        for _index in range(budget):
            candidates = [action for action in legal_actions if _action_key(action) not in self._selected]
            if not candidates:
                break
            action = max(candidates, key=lambda candidate: _action_score(state, candidate))
            transform, option = action
            state = transform.apply(state, option)
            selected.append(action)
            self._selected.add(_action_key(action))
            legal_actions = tuple(
                (candidate_transform, candidate_option)
                for candidate_transform in context.transforms
                for candidate_option in candidate_transform.analyze(state)
            )
        return tuple(selected)


def _action_key(action: Action) -> tuple[str, str]:
    """Return a stable policy-local identity for one legal action."""
    transform, option = action
    payload = vars(option)
    fields = (
        "tensor",
        "block_nid",
        "copy_block_nid",
        "producer_block_nid",
        "redundant_block_nid",
        "target_nid",
        "target_nids",
        "loop_nid",
        "outer_nid",
        "consumer_nid",
    )
    identity = tuple((name, payload[name]) for name in fields if name in payload)
    return (type(transform).__name__, repr(identity or option))


def _action_score(state: KernelIR, action: Action) -> tuple[int, int, str]:
    """Rank one action by generic transform semantics and IR structure."""
    transform, option = action
    transform_name = type(transform).__name__
    priority = _TRANSFORM_PRIORITY.get(transform_name, 0)
    structural = 0
    payload = vars(option)
    target_loop_nid = payload.get("target_loop_nid")
    tensor = payload.get("tensor")
    list_len = payload.get("list_len")
    factors = payload.get("factors")
    stages = payload.get("stages")
    if transform_name == "CodeMotion" and isinstance(target_loop_nid, int):
        leaves = [
            nid
            for nid in (target_loop_nid, *state.tree.descendants(target_loop_nid))
            if isinstance(state.tree.data(nid), ISANode)
        ]
        structural = sum(
            len(state.dependency.direct_producers(nid)) + len(state.dependency.direct_consumers(nid)) for nid in leaves
        )
    elif transform_name in {"BufferPlacement", "BufferCompaction"} and isinstance(tensor, str):
        structural = prod(state.buffer(tensor).shape)
    elif transform_name == "BufferLayout" and isinstance(list_len, int):
        structural = list_len
    elif transform_name == "Split" and isinstance(factors, tuple) and all(isinstance(value, int) for value in factors):
        structural = -abs(factors[0] - factors[1])
    elif transform_name == "SoftwarePipeline" and isinstance(stages, tuple):
        structural = -abs(stages.count(0) - stages.count(1))
    return (priority, structural, repr(option))


@dataclass(frozen=True)
class SearchResult:
    """Summary of one completed refinement run."""

    best_latency_ms: float
    transforms_applied: int
    evaluations_run: int
    finish_reason: str


__all__ = ["Action", "Policy", "PolicyContext", "SearchResult"]
