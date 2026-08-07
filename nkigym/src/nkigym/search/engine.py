"""Branching atomic-transform refinement with automatic Neuron profiling."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from nkigym.environment import Action, KernelMDP
from nkigym.ir import KernelIR
from nkigym.search.observation import DescribedAction, describe_actions, format_observation, state_fingerprint
from nkigym.search.types import (
    MAX_TRANSFORMS_PER_REASONING_STEP,
    AgentDecision,
    Evaluation,
    ReasoningPolicy,
    SearchConfig,
    SearchNode,
    SearchResult,
    StateEvaluator,
)


class ProfilerGuidedRefinement:
    """Branch from measured states, apply atomic transforms, and profile every result."""

    def __init__(
        self, environment: KernelMDP, policy: ReasoningPolicy, evaluator: StateEvaluator, config: SearchConfig
    ) -> None:
        """Store collaborators and initialize empty run state."""
        if config.max_reasoning_steps is not None and config.max_reasoning_steps < 0:
            raise ValueError("max_reasoning_steps must be non-negative")
        self.environment = environment
        self.policy = policy
        self.evaluator = evaluator
        self.config = config
        self.nodes: list[SearchNode] = []
        self._evaluation_by_render: dict[str, Evaluation] = {}
        self._actions_by_node: dict[int, tuple[DescribedAction, ...]] = {}
        self._explored_edges: set[tuple[int, str]] = set()
        self._active_node_id = 0
        self._evaluations_run = 0
        self._reasoning_steps = 0

    async def run(self) -> SearchResult:
        """Profile the root, then explore policy-selected branches."""
        self._prepare_run()
        finish_reason: str | None = None
        can_revisit = True
        while finish_reason is None and (
            self.config.max_reasoning_steps is None or self._reasoning_steps < self.config.max_reasoning_steps
        ):
            branchable_node_ids = self._branchable_node_ids()
            if not branchable_node_ids:
                finish_reason = "no unexplored transforms remain in trace"
                break
            active = self.nodes[self._active_node_id]
            actions = self._available_actions(active.node_id)
            reasoning_step = self._reasoning_steps + 1
            observation_branchable_node_ids = branchable_node_ids if can_revisit else (self._active_node_id,)
            branch_action_types = {
                node_id: _action_type_names(self._available_actions(node_id))
                for node_id in observation_branchable_node_ids
            }
            observation = format_observation(
                active=active,
                nodes=self.nodes,
                actions=actions,
                branchable_node_ids=observation_branchable_node_ids,
                branch_action_types=branch_action_types,
                config=self.config,
                reasoning_step=reasoning_step,
            )
            self._write_observation(reasoning_step, observation)
            decision = await self.policy.decide(observation)
            self._reasoning_steps = reasoning_step
            self._write_decision(reasoning_step, decision)
            if decision.kind == "finish":
                self._check_finish_decision(decision)
                if self._target_unmet():
                    target = self.config.target_score
                    if target is None:
                        raise RuntimeError("target status changed while checking a finish decision")
                    _log(f"rejected finish below target score {target:.4f}")
                else:
                    finish_reason = "policy finished"
            elif decision.kind == "revisit":
                if can_revisit:
                    self._revisit(decision, branchable_node_ids)
                    can_revisit = False
                else:
                    _log(f"rejected consecutive revisit to N{decision.base_node_id:03d}; apply or finish is required")
            elif decision.kind == "apply":
                self._check_apply_base(decision, active.node_id)
                selected = _selected_actions(decision, actions)
                planned_steps = _plan_states(self.environment, active.state, selected)
                parent_id = active.node_id
                for item, state in planned_steps:
                    self._explored_edges.add((parent_id, _action_key(item.action)))
                    node = self._add_measured_node(
                        state=state,
                        parent_id=parent_id,
                        action_id=item.action_id,
                        action_description=item.description,
                        rationale=decision.rationale,
                    )
                    parent_id = node.node_id
                    _log(f"N{node.node_id:03d}: {item.description}; {node.evaluation.message}")
                self._active_node_id = parent_id
                can_revisit = True
            else:
                raise ValueError(f"policy selected unknown decision kind {decision.kind!r}")
        if finish_reason is None:
            finish_reason = "reasoning step budget exhausted"
        result = self._build_result(finish_reason)
        self._write_result(result)
        return result

    def _prepare_run(self) -> None:
        """Reset artifacts and profile the canonical state."""
        shutil.rmtree(self.config.cache_dir, ignore_errors=True)
        self.config.cache_dir.mkdir(parents=True)
        self.nodes.clear()
        self._evaluation_by_render.clear()
        self._actions_by_node.clear()
        self._explored_edges.clear()
        self._active_node_id = 0
        self._evaluations_run = 0
        self._reasoning_steps = 0
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

    def _all_actions(self, node_id: int) -> tuple[DescribedAction, ...]:
        """Return one node's stable complete legal-action catalog."""
        actions = self._actions_by_node.get(node_id)
        if actions is None:
            node = self.nodes[node_id]
            actions = tuple(describe_actions(node.state, self.environment.legal_actions(node.state)))
            self._actions_by_node[node_id] = actions
        return actions

    def _available_actions(self, node_id: int) -> list[DescribedAction]:
        """Return legal parent edges not already measured from one node."""
        actions = [
            item
            for item in self._all_actions(node_id)
            if (node_id, _action_key(item.action)) not in self._explored_edges
        ]
        return actions

    def _branchable_node_ids(self) -> tuple[int, ...]:
        """Return every measured node with an unexplored legal transform."""
        node_ids = tuple(node.node_id for node in self.nodes if self._available_actions(node.node_id))
        return node_ids

    def _check_finish_decision(self, decision: AgentDecision) -> None:
        """Reject malformed finish decisions from non-JSON policy implementations."""
        if decision.base_node_id is not None or decision.action_ids:
            raise ValueError("finish requires base_node_id=None and no action_ids")

    def _target_unmet(self) -> bool:
        """Return whether measured evidence is still below a configured target."""
        target = self.config.target_score
        scores = [node.evaluation.score for node in self.nodes if node.evaluation.score is not None]
        return target is not None and (not scores or max(scores) < target)

    def _check_apply_base(self, decision: AgentDecision, active_node_id: int) -> None:
        """Require action IDs to be interpreted against the active node's menu."""
        base_node_id = decision.base_node_id
        if isinstance(base_node_id, bool) or not isinstance(base_node_id, int) or base_node_id < 0:
            raise ValueError("apply requires a non-negative integer base_node_id")
        if base_node_id != active_node_id:
            raise ValueError(
                f"apply base_node_id={base_node_id!r} does not match active node N{active_node_id:03d}; "
                "revisit that node before applying its actions"
            )

    def _revisit(self, decision: AgentDecision, branchable_node_ids: tuple[int, ...]) -> None:
        """Make an earlier branchable measured state active without profiling."""
        target = decision.base_node_id
        if target is None or decision.action_ids:
            raise ValueError("revisit requires base_node_id and no action_ids")
        if isinstance(target, bool) or not isinstance(target, int) or target < 0:
            raise ValueError("revisit requires a non-negative integer base_node_id")
        if target >= len(self.nodes):
            raise ValueError(f"revisit selected unknown base_node_id={target}")
        if target == self._active_node_id:
            raise ValueError(f"revisit selected already-active node N{target:03d}")
        if target not in branchable_node_ids:
            raise ValueError(f"revisit selected node N{target:03d} with no unexplored legal transforms")
        previous = self._active_node_id
        self._active_node_id = target
        _log(f"revisit N{target:03d} from N{previous:03d}")

    def _build_result(self, finish_reason: str) -> SearchResult:
        """Select the highest-scoring measured state."""
        successful = [node for node in self.nodes if node.evaluation.score is not None]
        best_node_id = max(successful, key=_successful_score).node_id if successful else None
        return SearchResult(
            nodes=tuple(self.nodes),
            best_node_id=best_node_id,
            active_node_id=self._active_node_id,
            transforms_applied=len(self.nodes) - 1,
            reasoning_steps=self._reasoning_steps,
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
            "base_node_id": decision.base_node_id,
            "action_ids": list(decision.action_ids),
            "rationale": decision.rationale,
            "raw_response": decision.raw_response,
        }
        (directory / f"iteration_{iteration:03d}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    def _write_result(self, result: SearchResult) -> None:
        """Write the measured search tree and selected best node."""
        payload = {
            "best_node_id": result.best_node_id,
            "active_node_id": result.active_node_id,
            "current_node_id": result.current_node.node_id,
            "transforms_applied": result.transforms_applied,
            "reasoning_steps": result.reasoning_steps,
            "evaluations_run": result.evaluations_run,
            "finish_reason": result.finish_reason,
            "branchable_node_ids": list(self._branchable_node_ids()),
            "history": [
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "action_id": node.action_id,
                    "action": node.action_description,
                    "rationale": node.rationale,
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


def _selected_actions(decision: AgentDecision, actions: list[DescribedAction]) -> tuple[DescribedAction, ...]:
    """Resolve and bound one ordered policy action sequence."""
    if not decision.action_ids:
        raise ValueError("apply decision selected no action_ids")
    if len(decision.action_ids) > MAX_TRANSFORMS_PER_REASONING_STEP:
        raise ValueError(
            f"policy selected {len(decision.action_ids)} actions; "
            f"at most {MAX_TRANSFORMS_PER_REASONING_STEP} are allowed per reasoning step"
        )
    if len(set(decision.action_ids)) != len(decision.action_ids):
        raise ValueError("policy selected duplicate action_ids")
    by_id = {item.action_id: item for item in actions}
    unknown = [action_id for action_id in decision.action_ids if action_id not in by_id]
    if unknown:
        raise ValueError(f"policy selected unknown action_ids {unknown}")
    selected = tuple(by_id[action_id] for action_id in decision.action_ids)
    return selected


def _plan_states(
    environment: KernelMDP, initial_state: KernelIR, selected: tuple[DescribedAction, ...]
) -> tuple[tuple[DescribedAction, KernelIR], ...]:
    """Apply the longest prefix whose actions remain legal in order."""
    state = initial_state
    steps: list[tuple[DescribedAction, KernelIR]] = []
    applied_ids: list[str] = []
    for item in selected:
        current_actions = describe_actions(state, environment.legal_actions(state))
        current_item = next((candidate for candidate in current_actions if candidate.action == item.action), None)
        if current_item is None:
            _log(
                f"rejected policy suffix beginning with {item.action_id!r}; "
                f"action is not legal after selected predecessors {applied_ids}"
            )
            break
        state = environment.step(state, item.action)
        steps.append((current_item, state))
        applied_ids.append(item.action_id)
    return tuple(steps)


def _action_key(action: Action) -> str:
    """Return a stable identity for one transform option within a run."""
    transform, option = action
    return f"{type(transform).__module__}.{type(transform).__qualname__}:{option!r}"


def _action_type_names(actions: list[DescribedAction]) -> tuple[str, ...]:
    """Return sorted transform types represented in one action menu."""
    return tuple(sorted({type(item.action[0]).__name__ for item in actions}))


def _successful_score(node: SearchNode) -> float:
    """Return one successfully measured score."""
    if node.evaluation.score is None:
        raise ValueError(f"node N{node.node_id:03d} has no successful score")
    return node.evaluation.score


def _log(message: str) -> None:
    """Print one immediately visible refinement update."""
    print(f"[refinement] {message}", flush=True)


__all__ = ["ProfilerGuidedRefinement"]
