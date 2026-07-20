"""Behavioral tests for the bounded agentic search engine."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from test.environment._fixtures import INPUT_SPECS, f_matmul
from typing import cast

import pytest

from autotune.search import AgentDecision, AgenticSearch, Evaluation, SearchConfig
from autotune.search.observation import state_fingerprint
from autotune.search.types import DecisionKind
from nkigym.environment import KernelMDP
from nkigym.ir import KernelIR
from nkigym.transforms import Fuse, Split


class DescriptionPolicy:
    """Choose actions by semantic text, then finish."""

    def __init__(self, needles: list[str]) -> None:
        """Store ordered action-description substrings."""
        self.needles = needles
        self.index = 0

    async def decide(self, observation: str) -> AgentDecision:
        """Select the next matching action or finish after the sequence."""
        decision: AgentDecision
        if self.index < len(self.needles):
            needle = self.needles[self.index]
            line = next(line for line in observation.splitlines() if line.startswith("- A") and needle in line)
            match = re.match(r"- (A\d+):", line)
            if match is None:
                raise AssertionError(f"could not parse action line {line!r}")
            self.index += 1
            decision = AgentDecision(
                kind="apply",
                rationale=f"test applies {needle}",
                raw_response=line,
                action_id=match.group(1),
                node_id=None,
            )
        else:
            decision = AgentDecision(
                kind="finish",
                rationale="test sequence complete",
                raw_response='{"kind":"finish"}',
                action_id=None,
                node_id=None,
            )
        return decision


class NodeCountEvaluator:
    """Score larger trees and record calls."""

    def __init__(self) -> None:
        """Initialize call tracking."""
        self.calls: list[int] = []

    def evaluate(self, state: KernelIR, node_id: int, cache_dir: Path) -> Evaluation:
        """Return node count as a deterministic higher-is-better score."""
        self.calls.append(node_id)
        return Evaluation(
            score=float(state.tree.num_nodes),
            metrics={"tree_nodes": state.tree.num_nodes},
            message=f"tree_nodes={state.tree.num_nodes}",
        )


class ScriptedPolicy:
    """Execute a fixed sequence using the first listed action for apply."""

    def __init__(self, operations: list[DecisionKind]) -> None:
        """Store operation names in decision order."""
        self.operations = operations
        self.index = 0

    async def decide(self, observation: str) -> AgentDecision:
        """Return the next operation in the script."""
        operation = self.operations[self.index]
        self.index += 1
        action_id: str | None = None
        if operation == "apply":
            line = next(line for line in observation.splitlines() if line.startswith("- A"))
            match = re.match(r"- (A\d+):", line)
            if match is None:
                raise AssertionError(f"could not parse action line {line!r}")
            action_id = match.group(1)
        return AgentDecision(
            kind=operation, rationale=f"test {operation}", raw_response=operation, action_id=action_id, node_id=None
        )


class AnnotationTransform:
    """Marker transform for a render-neutral test environment."""


@dataclass(frozen=True)
class AnnotationOption:
    """Option that records a completed metadata phase."""

    phase: str


class AnnotationEnvironment:
    """Expose one render-neutral action that changes future legality."""

    def __init__(self) -> None:
        """Create a canonical state and one opaque transform marker."""
        self._base = KernelMDP(f_matmul, INPUT_SPECS, transforms=[]).reset()
        self._transform = AnnotationTransform()
        self._option = AnnotationOption(phase="complete")

    def reset(self) -> KernelIR:
        """Return a fresh canonical state."""
        return copy.deepcopy(self._base)

    def legal_actions(self, state: KernelIR) -> list[object]:
        """Offer the annotation action until it has been applied."""
        block = state.tree.block(state.tree.root)
        actions: list[object] = []
        if block.annotations.get("phase") != "complete":
            actions.append((self._transform, self._option))
        return actions

    def step(self, state: KernelIR, action: object) -> KernelIR:
        """Apply the sole action by changing non-rendered block metadata."""
        if action != (self._transform, self._option):
            raise ValueError("unknown annotation action")
        result = copy.deepcopy(state)
        root = result.tree.root
        block = result.tree.block(root)
        annotations = {**block.annotations, "phase": "complete"}
        result.tree.graph.nodes[root]["data"] = replace(block, annotations=annotations)
        return result


def test_search_deduplicates_split_fuse_cycle_and_writes_artifacts(tmp_path: Path) -> None:
    """Split followed by its inverse Fuse returns to the existing root node."""
    environment = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse()])
    policy = DescriptionPolicy(["Split: split loop nid=2", "Fuse: fuse outer loops"])
    evaluator = NodeCountEvaluator()
    search = AgenticSearch(
        environment=environment,
        policy=policy,
        evaluator=evaluator,
        config=SearchConfig(
            cache_dir=tmp_path / "search",
            resume_dir=None,
            max_transforms=4,
            max_evaluations=2,
            min_evaluations=0,
            max_decisions=5,
            workload_guidance="Exercise a reversible transform pair.",
        ),
    )

    result = asyncio.run(search.run())

    assert len(result.nodes) == 2
    assert result.active_node_id == 0
    assert result.transforms_applied == 2
    assert result.evaluations_run == 1
    assert evaluator.calls == [0]
    assert (tmp_path / "search" / "events.jsonl").is_file()
    assert (tmp_path / "search" / "observations" / "decision_001.md").is_file()
    assert (tmp_path / "search" / "nodes" / "node_001" / "kernel.py").is_file()
    assert (tmp_path / "search" / "result.json").is_file()


def test_search_replays_prior_actions_and_evaluations_from_canonical(tmp_path: Path) -> None:
    """Resume reconstructs legal states and reuses hardware feedback."""
    environment = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split()])
    first_evaluator = NodeCountEvaluator()
    first = AgenticSearch(
        environment=environment,
        policy=DescriptionPolicy(["Split: split loop nid=2"]),
        evaluator=first_evaluator,
        config=SearchConfig(
            cache_dir=tmp_path / "first",
            resume_dir=None,
            max_transforms=3,
            max_evaluations=2,
            min_evaluations=0,
            max_decisions=3,
            workload_guidance="Create one replayable state.",
        ),
    )
    first_result = asyncio.run(first.run())
    resumed_evaluator = NodeCountEvaluator()
    resumed = AgenticSearch(
        environment=environment,
        policy=DescriptionPolicy([]),
        evaluator=resumed_evaluator,
        config=SearchConfig(
            cache_dir=tmp_path / "resumed",
            resume_dir=tmp_path / "first",
            max_transforms=3,
            max_evaluations=2,
            min_evaluations=1,
            max_decisions=4,
            workload_guidance="Continue from the replayed graph.",
        ),
    )

    resumed_result = asyncio.run(resumed.run())

    assert first_result.transforms_applied == 1
    assert resumed_result.transforms_applied == 1
    assert resumed_result.evaluations_run == 1
    assert len(resumed_result.nodes) == 2
    assert resumed_result.best_node.fingerprint == first_result.best_node.fingerprint
    assert resumed_evaluator.calls == []


def test_search_replays_semantic_states_and_counts_unique_render_evaluations(tmp_path: Path) -> None:
    """Resume preserves render-neutral states and hardware cache accounting."""
    environment = cast(KernelMDP, AnnotationEnvironment())
    first_evaluator = NodeCountEvaluator()
    first = AgenticSearch(
        environment=environment,
        policy=ScriptedPolicy(["evaluate", "apply", "evaluate", "finish"]),
        evaluator=first_evaluator,
        config=SearchConfig(
            cache_dir=tmp_path / "first",
            resume_dir=None,
            max_transforms=1,
            max_evaluations=2,
            min_evaluations=0,
            max_decisions=4,
            workload_guidance="Exercise render-neutral metadata.",
        ),
    )
    first_result = asyncio.run(first.run())
    resumed_evaluator = NodeCountEvaluator()
    resumed = AgenticSearch(
        environment=environment,
        policy=ScriptedPolicy([]),
        evaluator=resumed_evaluator,
        config=SearchConfig(
            cache_dir=tmp_path / "resumed",
            resume_dir=tmp_path / "first",
            max_transforms=1,
            max_evaluations=2,
            min_evaluations=0,
            max_decisions=4,
            workload_guidance="Replay render-neutral metadata.",
        ),
    )

    resumed_result = asyncio.run(resumed.run())

    assert len(first_result.nodes) == 2
    assert first_result.evaluations_run == 1
    assert first_evaluator.calls == [0]
    assert resumed_result.active_node_id == 1
    assert resumed_result.evaluations_run == 1
    assert resumed_evaluator.calls == []


def test_search_replays_hybrid_legacy_and_semantic_transitions(tmp_path: Path) -> None:
    """Recorded node transitions disambiguate a legacy-to-semantic transcript."""
    environment = cast(KernelMDP, AnnotationEnvironment())
    first = AgenticSearch(
        environment=environment,
        policy=ScriptedPolicy(["apply", "finish"]),
        evaluator=NodeCountEvaluator(),
        config=SearchConfig(
            cache_dir=tmp_path / "first",
            resume_dir=None,
            max_transforms=1,
            max_evaluations=1,
            min_evaluations=0,
            max_decisions=2,
            workload_guidance="Create a semantic metadata transition.",
        ),
    )
    asyncio.run(first.run())
    events_path = tmp_path / "first" / "events.jsonl"
    original = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    legacy_apply = {**original[0], "active_after": 0}
    semantic_apply = {**original[0], "decision": 2}
    finish = {**original[1], "decision": 3}
    events_path.write_text(
        "\n".join(json.dumps(event) for event in (legacy_apply, semantic_apply, finish)) + "\n", encoding="utf-8"
    )
    resumed = AgenticSearch(
        environment=environment,
        policy=ScriptedPolicy([]),
        evaluator=NodeCountEvaluator(),
        config=SearchConfig(
            cache_dir=tmp_path / "resumed",
            resume_dir=tmp_path / "first",
            max_transforms=2,
            max_evaluations=1,
            min_evaluations=0,
            max_decisions=3,
            workload_guidance="Replay a hybrid metadata transition.",
        ),
    )

    result = asyncio.run(resumed.run())

    assert len(result.nodes) == 2
    assert result.transforms_applied == 2
    assert result.active_node_id == 1


def test_search_replays_legacy_render_fingerprints(tmp_path: Path) -> None:
    """A render-only transcript is upgraded after canonical replay."""
    environment = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split()])
    first = AgenticSearch(
        environment=environment,
        policy=DescriptionPolicy(["Split: split loop nid=2"]),
        evaluator=NodeCountEvaluator(),
        config=SearchConfig(
            cache_dir=tmp_path / "first",
            resume_dir=None,
            max_transforms=1,
            max_evaluations=1,
            min_evaluations=0,
            max_decisions=2,
            workload_guidance="Create one legacy-compatible state.",
        ),
    )
    asyncio.run(first.run())
    root_path = tmp_path / "first" / "nodes" / "node_000" / "node.json"
    payload = json.loads(root_path.read_text(encoding="utf-8"))
    payload["fingerprint"] = state_fingerprint(first.nodes[0].state)
    root_path.write_text(json.dumps(payload), encoding="utf-8")
    resumed = AgenticSearch(
        environment=environment,
        policy=ScriptedPolicy([]),
        evaluator=NodeCountEvaluator(),
        config=SearchConfig(
            cache_dir=tmp_path / "resumed",
            resume_dir=tmp_path / "first",
            max_transforms=1,
            max_evaluations=1,
            min_evaluations=0,
            max_decisions=2,
            workload_guidance="Replay one legacy-compatible state.",
        ),
    )

    result = asyncio.run(resumed.run())

    assert len(result.nodes) == 2
    assert result.active_node_id == 1


def test_search_rejects_repeated_evaluation_of_active_node(tmp_path: Path) -> None:
    """A live policy cannot spend decisions re-evaluating a cached state."""
    search = AgenticSearch(
        environment=KernelMDP(f_matmul, INPUT_SPECS, transforms=[]),
        policy=ScriptedPolicy(["evaluate", "evaluate"]),
        evaluator=NodeCountEvaluator(),
        config=SearchConfig(
            cache_dir=tmp_path / "search",
            resume_dir=None,
            max_transforms=0,
            max_evaluations=2,
            min_evaluations=0,
            max_decisions=2,
            workload_guidance="Reject a repeated evaluation.",
        ),
    )

    with pytest.raises(RuntimeError, match="already evaluated node N000"):
        asyncio.run(search.run())


@pytest.mark.parametrize("field", ["max_transforms", "max_evaluations", "min_evaluations", "max_decisions"])
def test_search_rejects_negative_budgets(tmp_path: Path, field: str) -> None:
    """Every transform, evaluation, and decision budget must be non-negative."""
    config = SearchConfig(
        cache_dir=tmp_path / "search",
        resume_dir=None,
        max_transforms=1,
        max_evaluations=1,
        min_evaluations=0,
        max_decisions=1,
        workload_guidance="Validate search budgets.",
    )

    with pytest.raises(ValueError, match=f"{field} must be non-negative"):
        AgenticSearch(
            environment=KernelMDP(f_matmul, INPUT_SPECS, transforms=[]),
            policy=ScriptedPolicy([]),
            evaluator=NodeCountEvaluator(),
            config=replace(config, **{field: -1}),
        )
