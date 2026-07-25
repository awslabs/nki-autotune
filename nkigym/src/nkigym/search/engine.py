"""Linear transform refinement with automatic Neuron profiling."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from nkigym.environment import KernelMDP
from nkigym.ir import KernelIR
from nkigym.search.observation import DescribedAction, describe_actions, format_observation, state_fingerprint
from nkigym.search.types import (
    AgentDecision,
    Evaluation,
    ReasoningPolicy,
    SearchConfig,
    SearchNode,
    SearchResult,
    StateEvaluator,
)


class ProfilerGuidedRefinement:
    """Apply one policy-selected transform and profile every resulting state."""

    def __init__(
        self, environment: KernelMDP, policy: ReasoningPolicy, evaluator: StateEvaluator, config: SearchConfig
    ) -> None:
        """Store collaborators and initialize empty run state."""
        if config.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        self.environment = environment
        self.policy = policy
        self.evaluator = evaluator
        self.config = config
        self.nodes: list[SearchNode] = []
        self._evaluation_by_render: dict[str, Evaluation] = {}
        self._evaluations_run = 0

    async def run(self) -> SearchResult:
        """Profile the root, then apply and profile one transform per turn."""
        self._prepare_run()
        finish_reason: str | None = None
        while finish_reason is None and len(self.nodes) - 1 < self.config.max_iterations:
            current = self.nodes[-1]
            actions = describe_actions(current.state, self.environment.legal_actions(current.state))
            if not actions:
                finish_reason = "no legal transforms remain"
                break
            iteration = len(self.nodes)
            observation = format_observation(state=current.state, nodes=self.nodes, actions=actions, config=self.config)
            self._write_observation(iteration, observation)
            decision = await self.policy.decide(observation)
            self._write_decision(iteration, decision)
            if decision.kind == "finish":
                finish_reason = "policy finished"
            else:
                selected = _selected_action(decision, actions)
                state = self.environment.step(current.state, selected.action)
                node = self._add_measured_node(
                    state=state,
                    parent_id=current.node_id,
                    action_id=selected.action_id,
                    action_description=selected.description,
                    rationale=decision.rationale,
                )
                _log(f"N{node.node_id:03d}: {selected.description}; {node.evaluation.message}")
        if finish_reason is None:
            finish_reason = "iteration budget exhausted"
        result = self._build_result(finish_reason)
        self._write_result(result)
        return result

    def _prepare_run(self) -> None:
        """Reset artifacts and profile the canonical state."""
        shutil.rmtree(self.config.cache_dir, ignore_errors=True)
        self.config.cache_dir.mkdir(parents=True)
        self.nodes.clear()
        self._evaluation_by_render.clear()
        self._evaluations_run = 0
        root = self._add_measured_node(
            state=self.environment.reset(), parent_id=None, action_id=None, action_description=None, rationale=None
        )
        _log(f"N{root.node_id:03d}: canonical; {root.evaluation.message}")

    def _add_measured_node(
        self,
        state: KernelIR,
        parent_id: int | None,
        action_id: str | None,
        action_description: str | None,
        rationale: str | None,
    ) -> SearchNode:
        """Persist and profile one state, reusing render-identical feedback."""
        node_id = len(self.nodes)
        node_dir = self._node_dir(node_id)
        state.dump(node_dir)
        fingerprint = state_fingerprint(state)
        evaluation = self._evaluation_by_render.get(fingerprint)
        if evaluation is None:
            evaluation = self.evaluator.evaluate(state, node_id, node_dir)
            self._evaluation_by_render[fingerprint] = evaluation
            self._evaluations_run += 1
        node = SearchNode(
            node_id=node_id,
            state=state,
            parent_id=parent_id,
            action_id=action_id,
            action_description=action_description,
            rationale=rationale,
            evaluation=evaluation,
        )
        self.nodes.append(node)
        self._write_node(node, fingerprint)
        return node

    def _build_result(self, finish_reason: str) -> SearchResult:
        """Select the highest-scoring measured state."""
        successful = [node for node in self.nodes if node.evaluation.score is not None]
        best_node_id = max(successful, key=_successful_score).node_id if successful else None
        return SearchResult(
            nodes=tuple(self.nodes),
            best_node_id=best_node_id,
            transforms_applied=len(self.nodes) - 1,
            evaluations_run=self._evaluations_run,
            finish_reason=finish_reason,
        )

    def _write_node(self, node: SearchNode, fingerprint: str) -> None:
        """Write node metadata and Neuron feedback."""
        node_dir = self._node_dir(node.node_id)
        metadata = {
            "node_id": node.node_id,
            "fingerprint": fingerprint,
            "parent_id": node.parent_id,
            "action_id": node.action_id,
            "action_description": node.action_description,
            "rationale": node.rationale,
        }
        evaluation = {
            "score": node.evaluation.score,
            "metrics": node.evaluation.metrics,
            "message": node.evaluation.message,
        }
        (node_dir / "node.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (node_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")

    def _write_observation(self, iteration: int, observation: str) -> None:
        """Persist the exact policy-visible prompt."""
        directory = self.config.cache_dir / "observations"
        directory.mkdir(exist_ok=True)
        (directory / f"iteration_{iteration:03d}.md").write_text(observation + "\n", encoding="utf-8")

    def _write_decision(self, iteration: int, decision: AgentDecision) -> None:
        """Persist one parsed decision and its raw model response."""
        directory = self.config.cache_dir / "decisions"
        directory.mkdir(exist_ok=True)
        payload = {
            "kind": decision.kind,
            "action_id": decision.action_id,
            "rationale": decision.rationale,
            "raw_response": decision.raw_response,
        }
        (directory / f"iteration_{iteration:03d}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    def _write_result(self, result: SearchResult) -> None:
        """Write the measured linear history and selected best node."""
        payload = {
            "best_node_id": result.best_node_id,
            "current_node_id": result.current_node.node_id,
            "transforms_applied": result.transforms_applied,
            "evaluations_run": result.evaluations_run,
            "finish_reason": result.finish_reason,
            "history": [
                {
                    "node_id": node.node_id,
                    "action": node.action_description,
                    "score": node.evaluation.score,
                    "message": node.evaluation.message,
                }
                for node in result.nodes
            ],
        }
        (self.config.cache_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _node_dir(self, node_id: int) -> Path:
        """Return the artifact directory for one measured state."""
        return self.config.cache_dir / "nodes" / f"node_{node_id:03d}"


def _selected_action(decision: AgentDecision, actions: list[DescribedAction]) -> DescribedAction:
    """Resolve one policy action ID against the current legal list."""
    selected = next((item for item in actions if item.action_id == decision.action_id), None)
    if selected is None:
        raise ValueError(f"policy selected unknown action_id {decision.action_id!r}")
    return selected


def _successful_score(node: SearchNode) -> float:
    """Return one successfully measured score."""
    if node.evaluation.score is None:
        raise ValueError(f"node N{node.node_id:03d} has no successful score")
    return node.evaluation.score


def _log(message: str) -> None:
    """Print one immediately visible refinement update."""
    print(f"[refinement] {message}", flush=True)


__all__ = ["ProfilerGuidedRefinement"]
