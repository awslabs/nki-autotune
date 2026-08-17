"""Minimal iterative refinement loop."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nkigym.codegen import render
from nkigym.ir import KernelIR
from nkigym.profile import InputSpecs, ProfileMetrics, profile_metrics
from nkigym.search.types import Action, Policy, PolicyContext, SearchResult
from nkigym.transforms import Transform


@dataclass(frozen=True)
class SearchConfig:
    """Limits, profiling controls, and artifacts for one refinement run."""

    trace_dir: Path
    profile_host: str
    input_specs: InputSpecs
    neuronx_cc_args: tuple[str, ...]
    lnc: int
    profile_timeout_s: int
    max_transforms_per_evaluation: int
    max_evaluations: int
    target_latency_ms: float | None

    def __post_init__(self) -> None:
        """Reject invalid process controls."""
        if self.max_transforms_per_evaluation < 1:
            raise ValueError("max_transforms_per_evaluation must be positive")
        if self.max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")
        target = self.target_latency_ms
        if target is not None and (not math.isfinite(target) or target < 0):
            raise ValueError("target_latency_ms must be finite and non-negative")


class IterativeRefinement:
    """Apply policy-selected transforms and periodically evaluate the result."""

    def __init__(
        self, initial_state: KernelIR, transforms: tuple[Transform[Any], ...], policy: Policy, config: SearchConfig
    ) -> None:
        """Store the fixed process collaborators."""
        self.initial_state = initial_state
        self.transforms = transforms
        self.policy = policy
        self.config = config

    def run(self) -> SearchResult:
        """Refine linearly until the policy or a process limit stops."""
        self._prepare_trace()
        state = self.initial_state
        evaluations = [self._evaluate(state, (), 0)]
        transforms_applied = 0
        finish_reason = self._stop_after_evaluation(evaluations)

        while finish_reason is None:
            legal_actions = self._legal_actions(state)
            if not legal_actions:
                finish_reason = "no legal transforms remain"
                continue

            context = PolicyContext(
                state=state,
                transforms=self.transforms,
                legal_actions=legal_actions,
                evaluations=tuple(evaluations),
                max_transforms=self.config.max_transforms_per_evaluation,
            )
            actions = self.policy.select_actions(context)
            if not actions:
                finish_reason = "policy finished"
                continue
            if len(actions) > self.config.max_transforms_per_evaluation:
                raise ValueError(
                    f"policy returned {len(actions)} transforms; "
                    f"limit is {self.config.max_transforms_per_evaluation}"
                )

            state = self._apply_actions(state, actions)
            transforms_applied += len(actions)
            evaluations.append(self._evaluate(state, actions, len(evaluations)))
            finish_reason = self._stop_after_evaluation(evaluations)

        result = SearchResult(
            best_latency_ms=_best_latency(evaluations),
            transforms_applied=transforms_applied,
            evaluations_run=len(evaluations),
            finish_reason=finish_reason,
        )
        self._write_result(result)
        return result

    def _legal_actions(self, state: KernelIR) -> tuple[Action, ...]:
        """Return every legal action in transform and option order."""
        return tuple((transform, option) for transform in self.transforms for option in transform.analyze(state))

    def _apply_actions(self, state: KernelIR, actions: tuple[Action, ...]) -> KernelIR:
        """Apply an ordered sequence while re-checking every action."""
        current = state
        for action in actions:
            if action not in self._legal_actions(current):
                raise ValueError("policy returned an action that is not legal after its predecessors")
            transform, option = action
            current = transform.apply(current, option)
        return current

    def _evaluate(self, state: KernelIR, actions: tuple[Action, ...], evaluation_id: int) -> ProfileMetrics:
        """Persist and profile one candidate."""
        directory = self.config.trace_dir / "evaluations" / f"evaluation_{evaluation_id:03d}"
        metrics = profile_metrics(
            host=self.config.profile_host,
            kernel=render(state),
            func_name=f"nki_{state.func_name}",
            input_specs=self.config.input_specs,
            cache_dir=directory,
            neuronx_cc_args=self.config.neuronx_cc_args,
            lnc=self.config.lnc,
            timeout_s=self.config.profile_timeout_s,
        )
        payload = {"actions": [_action_description(action) for action in actions]}
        (directory / "actions.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return metrics

    def _stop_after_evaluation(self, evaluations: list[ProfileMetrics]) -> str | None:
        """Return a target or budget stop reason."""
        finish_reason: str | None = None
        target = self.config.target_latency_ms
        if target is not None and _best_latency(evaluations) <= target:
            finish_reason = "target latency reached"
        elif len(evaluations) >= self.config.max_evaluations:
            finish_reason = "evaluation budget exhausted"
        return finish_reason

    def _prepare_trace(self) -> None:
        """Reset the trace directory."""
        shutil.rmtree(self.config.trace_dir, ignore_errors=True)
        self.config.trace_dir.mkdir(parents=True)

    def _write_result(self, result: SearchResult) -> None:
        """Persist the final summary."""
        payload = {
            "best_latency_ms": result.best_latency_ms,
            "transforms_applied": result.transforms_applied,
            "evaluations_run": result.evaluations_run,
            "finish_reason": result.finish_reason,
        }
        (self.config.trace_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _action_description(action: Action) -> str:
    """Return a generic transform description."""
    transform, option = action
    return f"{type(transform).__name__}: {option!r}"


def _best_latency(evaluations: list[ProfileMetrics]) -> float:
    """Return the lowest measured latency."""
    return min(evaluation.latency_ms for evaluation in evaluations)


__all__ = ["IterativeRefinement", "SearchConfig"]
