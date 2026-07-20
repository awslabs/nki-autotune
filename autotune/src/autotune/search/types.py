"""Shared types for reasoning-driven transform search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from nkigym.ir import KernelIR

DecisionKind = Literal["apply", "evaluate", "checkout", "finish"]
EvaluationMetric = float | int | str | bool | None


@dataclass(frozen=True)
class AgentDecision:
    """One policy command returned to the search engine.

    Attributes:
        kind: Operation to perform.
        rationale: Concise, auditable technical justification.
        raw_response: Original policy response for the transcript.
        action_id: Per-observation action identifier for ``apply``.
        node_id: Previously discovered node identifier for ``checkout``.
    """

    kind: DecisionKind
    rationale: str
    raw_response: str
    action_id: str | None
    node_id: int | None


@dataclass(frozen=True)
class Evaluation:
    """Result of evaluating one rendered state.

    ``score`` is maximized by the engine. A failed compile or hardware run has
    ``score=None`` and remains useful feedback for the reasoning policy.
    """

    score: float | None
    metrics: dict[str, EvaluationMetric]
    message: str


@dataclass
class SearchNode:
    """One distinct semantic transform state in the explored graph."""

    node_id: int
    state: KernelIR
    fingerprint: str
    parent_id: int | None
    action_id: str | None
    action_description: str | None
    evaluation: Evaluation | None


@dataclass(frozen=True)
class SearchEvent:
    """One executed policy decision recorded in the search transcript."""

    decision: int
    active_before: int
    active_after: int
    kind: DecisionKind
    action_id: str | None
    node_id: int | None
    rationale: str
    raw_response: str


@dataclass(frozen=True)
class SearchConfig:
    """Budgets, artifacts, and workload guidance for one search run."""

    cache_dir: Path
    resume_dir: Path | None
    max_transforms: int
    max_evaluations: int
    min_evaluations: int
    max_decisions: int
    workload_guidance: str


@dataclass(frozen=True)
class SearchResult:
    """Completed search graph and selected best evaluated node."""

    nodes: tuple[SearchNode, ...]
    active_node_id: int
    best_node_id: int | None
    transforms_applied: int
    evaluations_run: int
    finish_reason: str

    @property
    def active_node(self) -> SearchNode:
        """Return the node active when the search stopped."""
        return self.nodes[self.active_node_id]

    @property
    def best_node(self) -> SearchNode:
        """Return the highest-scoring successful node."""
        if self.best_node_id is None:
            raise RuntimeError("search produced no successful evaluation")
        return self.nodes[self.best_node_id]

    def trace_to(self, node_id: int) -> list[SearchNode]:
        """Return the root-to-node path for ``node_id``."""
        path: list[SearchNode] = []
        cursor: int | None = node_id
        while cursor is not None:
            node = self.nodes[cursor]
            path.append(node)
            cursor = node.parent_id
        path.reverse()
        return path


class ReasoningPolicy(Protocol):
    """Policy that converts a complete textual observation into one decision."""

    async def decide(self, observation: str) -> AgentDecision:
        """Choose the next bounded search operation."""
        ...


class StateEvaluator(Protocol):
    """Evaluator that scores a state and writes any detailed artifacts."""

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
    "SearchEvent",
    "SearchNode",
    "SearchResult",
    "StateEvaluator",
]
