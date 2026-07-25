"""Types for linear profiler-guided ``nkigym`` transform refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from nkigym.ir import KernelIR

DecisionKind = Literal["apply", "finish"]
EvaluationMetric = float | int | str | bool | None


@dataclass(frozen=True)
class AgentDecision:
    """One next-transform decision returned by the reasoning policy."""

    kind: DecisionKind
    rationale: str
    raw_response: str
    action_id: str | None


@dataclass(frozen=True)
class Evaluation:
    """One Neuron compile and profile result."""

    score: float | None
    metrics: dict[str, EvaluationMetric]
    message: str


@dataclass(frozen=True)
class SearchNode:
    """One measured state in the linear refinement history."""

    node_id: int
    state: KernelIR
    parent_id: int | None
    action_id: str | None
    action_description: str | None
    rationale: str | None
    evaluation: Evaluation


@dataclass(frozen=True)
class SearchConfig:
    """Artifacts, iteration limit, and workload guidance for one run."""

    cache_dir: Path
    max_iterations: int
    workload_guidance: str


@dataclass(frozen=True)
class SearchResult:
    """Completed linear refinement history and best measured state."""

    nodes: tuple[SearchNode, ...]
    best_node_id: int | None
    transforms_applied: int
    evaluations_run: int
    finish_reason: str

    @property
    def current_node(self) -> SearchNode:
        """Return the last state reached by refinement."""
        return self.nodes[-1]

    @property
    def best_node(self) -> SearchNode:
        """Return the highest-scoring successful state."""
        if self.best_node_id is None:
            raise RuntimeError("refinement produced no successful evaluation")
        return self.nodes[self.best_node_id]


class ReasoningPolicy(Protocol):
    """Policy that chooses one listed transform or finishes."""

    async def decide(self, observation: str) -> AgentDecision:
        """Choose the next refinement operation."""
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
    "ReasoningPolicy",
    "SearchConfig",
    "SearchNode",
    "SearchResult",
    "StateEvaluator",
]
