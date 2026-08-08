"""Types for deterministic heuristic schedule search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from nkigym.ir import KernelIR
from nkigym.search.heuristics import ScheduleStep

EvaluationMetric: TypeAlias = float | int | str | bool | None
InputSpecs: TypeAlias = dict[str, tuple[tuple[int, ...], str]]


@dataclass(frozen=True)
class Evaluation:
    """Measured hardware result for one schedule candidate."""

    score: float | None
    metrics: dict[str, EvaluationMetric]
    message: str


@dataclass(frozen=True)
class ScheduleCandidate:
    """One fully lowered schedule and its optional hardware measurement."""

    candidate_id: int
    family: str
    strategy: str
    state: KernelIR
    steps: tuple[ScheduleStep, ...]
    evaluation: Evaluation


@dataclass(frozen=True)
class SearchResult:
    """Completed deterministic schedule search."""

    candidates: tuple[ScheduleCandidate, ...]
    best_candidate_id: int | None
    transforms_applied: int
    evaluations_run: int
    finish_reason: str

    @property
    def best_candidate(self) -> ScheduleCandidate:
        """Return the highest-MFU successfully measured schedule."""
        if self.best_candidate_id is None:
            raise RuntimeError("heuristic search produced no successful evaluation")
        return self.candidates[self.best_candidate_id]


__all__ = ["Evaluation", "EvaluationMetric", "InputSpecs", "ScheduleCandidate", "SearchResult"]
