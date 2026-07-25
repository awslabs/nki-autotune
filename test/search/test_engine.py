"""Behavioral tests for linear profiler-guided refinement."""

from __future__ import annotations

import asyncio
import copy
import re
from dataclasses import dataclass, replace
from pathlib import Path
from test.environment._fixtures import INPUT_SPECS, f_matmul
from typing import cast

import pytest

from nkigym.environment import KernelMDP
from nkigym.ir import KernelIR
from nkigym.search import AgentDecision, Evaluation, ProfilerGuidedRefinement, SearchConfig
from nkigym.transforms import Split


class DescriptionPolicy:
    """Apply actions selected by semantic description, then finish."""

    def __init__(self, needles: list[str]) -> None:
        """Store ordered action-description substrings."""
        self.needles = needles
        self.index = 0
        self.observations: list[str] = []

    async def decide(self, observation: str) -> AgentDecision:
        """Choose the next matching action or finish."""
        self.observations.append(observation)
        decision: AgentDecision
        if self.index < len(self.needles):
            needle = self.needles[self.index]
            line = next(line for line in observation.splitlines() if line.startswith("- A") and needle in line)
            match = re.match(r"- (A\d+):", line)
            if match is None:
                raise AssertionError(f"could not parse action line {line!r}")
            self.index += 1
            decision = AgentDecision(
                kind="apply", rationale=f"apply {needle}", raw_response=line, action_id=match.group(1)
            )
        else:
            decision = AgentDecision(
                kind="finish", rationale="test sequence complete", raw_response='{"kind":"finish"}', action_id=None
            )
        return decision


class FirstActionPolicy:
    """Always apply the first listed legal action."""

    async def decide(self, observation: str) -> AgentDecision:
        """Select the first action identifier."""
        line = next(line for line in observation.splitlines() if line.startswith("- A"))
        match = re.match(r"- (A\d+):", line)
        if match is None:
            raise AssertionError(f"could not parse action line {line!r}")
        return AgentDecision(kind="apply", rationale="apply first", raw_response=line, action_id=match.group(1))


class UnknownActionPolicy:
    """Return an action ID absent from the observation."""

    async def decide(self, observation: str) -> AgentDecision:
        """Return one invalid action ID."""
        return AgentDecision(kind="apply", rationale="invalid", raw_response="{}", action_id="A999")


class NodeCountEvaluator:
    """Score larger trees and record profile calls."""

    def __init__(self) -> None:
        """Initialize call tracking."""
        self.calls: list[int] = []

    def evaluate(self, state: KernelIR, node_id: int, cache_dir: Path) -> Evaluation:
        """Return node count as deterministic profiler feedback."""
        self.calls.append(node_id)
        return Evaluation(
            score=float(state.tree.num_nodes),
            metrics={"tree_nodes": state.tree.num_nodes},
            message=f"tree_nodes={state.tree.num_nodes}",
        )


class AnnotationTransform:
    """Marker transform for a render-neutral test environment."""


@dataclass(frozen=True)
class AnnotationOption:
    """Option that records a metadata phase."""

    phase: str


class AnnotationEnvironment:
    """Expose one render-neutral action that changes future legality."""

    def __init__(self) -> None:
        """Create a canonical state and one metadata transform."""
        self._base = KernelMDP(f_matmul, INPUT_SPECS, transforms=[]).reset()
        self._transform = AnnotationTransform()
        self._option = AnnotationOption(phase="complete")

    def reset(self) -> KernelIR:
        """Return a fresh canonical state."""
        return copy.deepcopy(self._base)

    def legal_actions(self, state: KernelIR) -> list[object]:
        """Offer the annotation until it has been applied."""
        block = state.tree.block(state.tree.root)
        actions: list[object] = []
        if block.annotations.get("phase") != "complete":
            actions.append((self._transform, self._option))
        return actions

    def step(self, state: KernelIR, action: object) -> KernelIR:
        """Apply the annotation without changing rendered NKI."""
        if action != (self._transform, self._option):
            raise ValueError("unknown annotation action")
        result = copy.deepcopy(state)
        root = result.tree.root
        block = result.tree.block(root)
        result.tree.graph.nodes[root]["data"] = replace(block, annotations={**block.annotations, "phase": "complete"})
        return result


def test_refinement_profiles_root_and_every_transform(tmp_path: Path) -> None:
    """Neuron feedback is automatic before and after each policy decision."""
    policy = DescriptionPolicy(["Split: split loop nid=2"])
    evaluator = NodeCountEvaluator()
    refinement = ProfilerGuidedRefinement(
        environment=KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split()]),
        policy=policy,
        evaluator=evaluator,
        config=SearchConfig(
            cache_dir=tmp_path / "search", max_iterations=3, workload_guidance="Exercise one measured transform."
        ),
    )

    result = asyncio.run(refinement.run())

    assert evaluator.calls == [0, 1]
    assert result.transforms_applied == 1
    assert result.evaluations_run == 2
    assert result.finish_reason == "policy finished"
    assert "tree_nodes=" in policy.observations[0]
    assert "The orchestrator profiles every applied transform automatically." in policy.observations[0]
    assert (tmp_path / "search" / "nodes" / "node_000" / "evaluation.json").is_file()
    assert (tmp_path / "search" / "observations" / "iteration_001.md").is_file()
    assert (tmp_path / "search" / "decisions" / "iteration_001.json").is_file()
    assert (tmp_path / "search" / "result.json").is_file()


def test_refinement_stops_at_transform_budget(tmp_path: Path) -> None:
    """The iteration limit bounds both transforms and new profiles."""
    evaluator = NodeCountEvaluator()
    refinement = ProfilerGuidedRefinement(
        environment=KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split()]),
        policy=FirstActionPolicy(),
        evaluator=evaluator,
        config=SearchConfig(cache_dir=tmp_path / "search", max_iterations=1, workload_guidance="Apply one transform."),
    )

    result = asyncio.run(refinement.run())

    assert result.transforms_applied == 1
    assert result.evaluations_run == 2
    assert result.finish_reason == "iteration budget exhausted"


def test_refinement_reuses_render_identical_profile(tmp_path: Path) -> None:
    """A metadata-only transform receives cached Neuron feedback."""
    evaluator = NodeCountEvaluator()
    refinement = ProfilerGuidedRefinement(
        environment=cast(KernelMDP, AnnotationEnvironment()),
        policy=FirstActionPolicy(),
        evaluator=evaluator,
        config=SearchConfig(
            cache_dir=tmp_path / "search", max_iterations=1, workload_guidance="Exercise render caching."
        ),
    )

    result = asyncio.run(refinement.run())

    assert len(result.nodes) == 2
    assert evaluator.calls == [0]
    assert result.evaluations_run == 1
    assert result.nodes[0].evaluation == result.nodes[1].evaluation


def test_refinement_rejects_unknown_action(tmp_path: Path) -> None:
    """A policy cannot invent a transform action."""
    refinement = ProfilerGuidedRefinement(
        environment=KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split()]),
        policy=UnknownActionPolicy(),
        evaluator=NodeCountEvaluator(),
        config=SearchConfig(
            cache_dir=tmp_path / "search", max_iterations=1, workload_guidance="Reject unknown actions."
        ),
    )

    with pytest.raises(ValueError, match="unknown action_id 'A999'"):
        asyncio.run(refinement.run())


def test_refinement_rejects_negative_iteration_limit(tmp_path: Path) -> None:
    """The transform iteration limit must be non-negative."""
    with pytest.raises(ValueError, match="max_iterations must be non-negative"):
        ProfilerGuidedRefinement(
            environment=KernelMDP(f_matmul, INPUT_SPECS, transforms=[]),
            policy=FirstActionPolicy(),
            evaluator=NodeCountEvaluator(),
            config=SearchConfig(
                cache_dir=tmp_path / "search", max_iterations=-1, workload_guidance="Validate the budget."
            ),
        )
