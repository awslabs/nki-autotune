"""Types for branching profiler-guided ``nkigym`` transform refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from nkigym.ir import KernelIR

DecisionKind = Literal["apply", "revisit", "finish"]
EvaluationMetric = float | int | str | bool | None
InputSpecs = dict[str, tuple[tuple[int, ...], str]]
MAX_TRANSFORMS_PER_REASONING_STEP = 3


@dataclass(frozen=True)
class AgentDecision:
    """One branch-selection or ordered transform decision returned by the policy."""

    kind: DecisionKind
    base_node_id: int | None
    rationale: str
    raw_response: str
    action_ids: tuple[str, ...]


@dataclass(frozen=True)
class Evaluation:
    """One scored state returned by a search evaluator."""

    score: float | None
    metrics: dict[str, EvaluationMetric]
    message: str


@dataclass(frozen=True)
class SearchNode:
    """One measured state in the branching refinement trace."""

    node_id: int
    state: KernelIR
    parent_id: int | None
    action_id: str | None
    action_description: str | None
    rationale: str | None
    evaluation: Evaluation


@dataclass(frozen=True)
class SearchConfig:
    """Artifacts, optional reasoning limit, and workload guidance."""

    cache_dir: Path
    max_reasoning_steps: int | None
    workload_guidance: str


@dataclass(frozen=True)
class SearchResult:
    """Completed branching refinement trace and best measured state."""

    nodes: tuple[SearchNode, ...]
    best_node_id: int | None
    active_node_id: int
    transforms_applied: int
    reasoning_steps: int
    evaluations_run: int
    finish_reason: str

    @property
    def current_node(self) -> SearchNode:
        """Return the trace node active when refinement finished."""
        return self.nodes[self.active_node_id]

    @property
    def best_node(self) -> SearchNode:
        """Return the highest-scoring successful state."""
        if self.best_node_id is None:
            raise RuntimeError("refinement produced no successful evaluation")
        return self.nodes[self.best_node_id]


class ReasoningPolicy(Protocol):
    """Policy that revisits a measured node, applies transforms, or finishes."""

    async def decide(self, observation: str) -> AgentDecision:
        """Choose the next ordered refinement operations."""
        ...


class StateEvaluator(Protocol):
    """Evaluator that compiles and profiles one state on Neuron."""

    def evaluate(self, state: KernelIR, node_id: int, cache_dir: Path) -> Evaluation:
        """Evaluate ``state`` with a higher-is-better score."""
        ...


__all__ = [
    "AgentDecision",
    "DecisionKind",
    "Evaluation",
    "EvaluationMetric",
    "InputSpecs",
    "MAX_TRANSFORMS_PER_REASONING_STEP",
    "ReasoningPolicy",
    "SearchConfig",
    "SearchNode",
    "SearchResult",
    "StateEvaluator",
]
