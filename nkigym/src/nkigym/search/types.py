"""Public types for iterative schedule refinement."""

from __future__ import annotations

from dataclasses import dataclass

from nkigym.environment import Action
from nkigym.ir import KernelIR
from nkigym.profile.types import ProfileMetrics


@dataclass(frozen=True)
class PolicyContext:
    """State exposed to a policy before one refinement step."""

    state: KernelIR
    legal_actions: tuple[Action, ...]
    evaluations: tuple[ProfileMetrics, ...]
    max_transforms: int


class Policy:
    """Choose transforms while the backend owns evaluation and stopping."""

    def select_actions(self, context: PolicyContext) -> tuple[Action, ...]:
        """Return an ordered transform sequence, or an empty tuple to finish."""
        raise NotImplementedError("search policy is not implemented")


@dataclass(frozen=True)
class SearchResult:
    """Summary of one completed refinement run."""

    best_latency_ms: float
    transforms_applied: int
    evaluations_run: int
    finish_reason: str


__all__ = ["Policy", "PolicyContext", "SearchResult"]
