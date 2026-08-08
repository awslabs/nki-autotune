"""Deterministic heuristic schedule search with batched hardware profiling."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nkigym.codegen import render
from nkigym.ir import build_initial_ir
from nkigym.profile import profile_many
from nkigym.search.heuristics import SchedulePlan, build_heuristic_plans
from nkigym.search.types import Evaluation, InputSpecs, ScheduleCandidate, SearchResult


@dataclass(frozen=True)
class SearchConfig:
    """Hardware controls and artifact location for one search."""

    profile_host: str
    input_specs: InputSpecs
    cache_dir: Path
    neuronx_cc_args: tuple[str, ...]
    lnc: int
    timeout_s: int

    def __post_init__(self) -> None:
        """Reject invalid search controls before schedule construction."""
        if not self.profile_host.strip():
            raise ValueError("profile_host must not be empty")
        if self.lnc not in {1, 2}:
            raise ValueError("lnc must be 1 or 2")
        if self.timeout_s < 1:
            raise ValueError("timeout_s must be positive")


class HeuristicScheduleSearch:
    """Lower semantic schedule rules and measure their final candidates."""

    def __init__(self, kernel_func: Callable[..., Any], config: SearchConfig) -> None:
        """Store the kernel and immutable search controls."""
        self.kernel_func = kernel_func
        self.config = config

    def run(self) -> SearchResult:
        """Build, persist, batch-profile, and select heuristic schedules."""
        self._prepare_cache()
        canonical = build_initial_ir(self.kernel_func, self.config.input_specs)
        plans = build_heuristic_plans(canonical)
        for index, plan in enumerate(plans):
            self._write_plan(index, plan)
        evaluations = self._profile_plans(plans)
        candidates = tuple(
            ScheduleCandidate(
                candidate_id=index,
                family=candidate.family,
                strategy=candidate.strategy,
                state=candidate.state,
                steps=candidate.steps,
                evaluation=evaluations[index],
            )
            for index, candidate in enumerate(plans)
        )
        successful = [candidate for candidate in candidates if candidate.evaluation.score is not None]
        best_candidate_id = max(successful, key=_candidate_score).candidate_id if successful else None
        result = SearchResult(
            candidates=candidates,
            best_candidate_id=best_candidate_id,
            transforms_applied=sum(len(candidate.steps) for candidate in candidates),
            evaluations_run=len(candidates),
            finish_reason="heuristic candidates exhausted",
        )
        self._write_result(result)
        return result

    def _prepare_cache(self) -> None:
        """Create an empty artifact directory for the search."""
        shutil.rmtree(self.config.cache_dir, ignore_errors=True)
        self.config.cache_dir.mkdir(parents=True)

    def _write_plan(self, candidate_id: int, plan: SchedulePlan) -> None:
        """Persist the optimized kernel and its complete atomic action trace."""
        candidate_dir = self.config.cache_dir / "candidates" / f"candidate_{candidate_id:03d}"
        plan.state.dump(candidate_dir)
        payload = {
            "candidate_id": candidate_id,
            "family": plan.family,
            "strategy": plan.strategy,
            "fingerprint": _fingerprint(plan),
            "transforms": [
                {
                    "index": index,
                    "transform": type(step.action[0]).__name__,
                    "option": repr(step.action[1]),
                    "rationale": step.rationale,
                }
                for index, step in enumerate(plan.steps, 1)
            ],
        }
        (candidate_dir / "plan.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _profile_plans(self, plans: tuple[SchedulePlan, ...]) -> tuple[Evaluation, ...]:
        """Profile all final schedule candidates in one remote batch."""
        labels = tuple(f"candidate_{index:03d}" for index in range(len(plans)))
        report = profile_many(
            host=self.config.profile_host,
            kernels={label: render(plan.state) for label, plan in zip(labels, plans, strict=True)},
            func_name=f"nki_{plans[0].state.func_name}",
            input_specs=self.config.input_specs,
            cache_dir=self.config.cache_dir / "profiles",
            neuronx_cc_args=self.config.neuronx_cc_args,
            lnc=self.config.lnc,
            max_workers=len(plans),
            timeout_s=self.config.timeout_s,
        )
        successes = report.get("successes")
        if not isinstance(successes, dict):
            raise RuntimeError("batch profiler returned malformed successes")
        failures = _profile_failures(self.config.cache_dir / "profiles")
        evaluations = tuple(
            _evaluation_from_measurement(index, label, successes.get(label), failures.get(label))
            for index, label in enumerate(labels)
        )
        return evaluations

    def _write_result(self, result: SearchResult) -> None:
        """Persist the measured candidates and selected endpoint."""
        payload = {
            "best_candidate_id": result.best_candidate_id,
            "transforms_applied": result.transforms_applied,
            "evaluations_run": result.evaluations_run,
            "finish_reason": result.finish_reason,
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "strategy": candidate.strategy,
                    "transforms": len(candidate.steps),
                    "score": candidate.evaluation.score,
                    "metrics": candidate.evaluation.metrics,
                    "message": candidate.evaluation.message,
                }
                for candidate in result.candidates
            ],
        }
        (self.config.cache_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fingerprint(plan: SchedulePlan) -> str:
    """Return a stable fingerprint for one rendered schedule."""
    fingerprint = hashlib.sha256(render(plan.state).encode("utf-8")).hexdigest()
    return fingerprint


def _profile_failures(profile_dir: Path) -> dict[str, str]:
    """Read per-candidate failures from the aggregate profile artifact."""
    path = profile_dir / "results.json"
    if not path.is_file():
        raise RuntimeError(f"batch profiler did not write {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = payload.get("failures")
    if not isinstance(failures, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in failures.items()
    ):
        raise RuntimeError("batch profiler returned malformed failures")
    return failures


def _evaluation_from_measurement(candidate_id: int, label: str, measurement: object, failure: str | None) -> Evaluation:
    """Convert one batch result entry into a search evaluation."""
    if isinstance(measurement, dict):
        mfu = measurement.get("mfu_percent")
        latency = measurement.get("latency_ms")
        if isinstance(mfu, bool) or not isinstance(mfu, (int, float)):
            raise RuntimeError(f"{label} returned invalid mfu_percent {mfu!r}")
        if isinstance(latency, bool) or not isinstance(latency, (int, float)):
            raise RuntimeError(f"{label} returned invalid latency_ms {latency!r}")
        evaluation = Evaluation(
            score=float(mfu),
            metrics={"profile_succeeded": True, "mfu_percent": float(mfu), "latency_ms": float(latency)},
            message=f"candidate {candidate_id}: MFU={float(mfu):.2f}%, latency={float(latency):.4f} ms",
        )
    else:
        detail = failure or "profile failed without a diagnostic"
        evaluation = Evaluation(
            score=None, metrics={"profile_succeeded": False}, message=f"candidate {candidate_id}: {detail}"
        )
    return evaluation


def _candidate_score(candidate: ScheduleCandidate) -> float:
    """Return one successful candidate's measured MFU."""
    if candidate.evaluation.score is None:
        raise ValueError(f"candidate {candidate.candidate_id} has no successful score")
    return candidate.evaluation.score


__all__ = ["HeuristicScheduleSearch", "SearchConfig"]
