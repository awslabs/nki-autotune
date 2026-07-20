"""Bounded graph search driven by a reasoning policy."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from autotune.search.observation import (
    DescribedAction,
    describe_actions,
    format_observation,
    search_state_fingerprint,
    state_fingerprint,
)
from autotune.search.transcript import parse_evaluation, parse_event, resume_evaluation_path
from autotune.search.types import (
    AgentDecision,
    Evaluation,
    ReasoningPolicy,
    SearchConfig,
    SearchEvent,
    SearchNode,
    SearchResult,
    StateEvaluator,
)
from nkigym.environment import KernelMDP
from nkigym.ir import KernelIR


class AgenticSearch:
    """Explore legal transform states under policy and evaluation budgets."""

    def __init__(
        self, environment: KernelMDP, policy: ReasoningPolicy, evaluator: StateEvaluator, config: SearchConfig
    ) -> None:
        """Store collaborators and initialize empty run state."""
        _validate_config(config)
        self.environment = environment
        self.policy = policy
        self.evaluator = evaluator
        self.config = config
        self.nodes: list[SearchNode] = []
        self._by_fingerprint: dict[str, int] = {}
        self._evaluation_by_render: dict[str, Evaluation] = {}
        self._events: list[SearchEvent] = []
        self._active_node_id = 0
        self._transforms_applied = 0
        self._evaluations_run = 0

    async def run(self) -> SearchResult:
        """Run until the policy finishes or a decision budget is exhausted."""
        self._prepare_run()
        finish_reason: str | None = None
        decisions = len(self._events)
        while finish_reason is None and decisions < self.config.max_decisions:
            described = self._legal_actions()
            _log(
                f"decision {decisions + 1}/{self.config.max_decisions}: "
                f"active=N{self._active_node_id:03d}, legal_actions={len(described)}"
            )
            observation = format_observation(
                state=self.nodes[self._active_node_id].state,
                nodes=self.nodes,
                active_node_id=self._active_node_id,
                actions=described,
                config=self.config,
                transforms_applied=self._transforms_applied,
                evaluations_run=self._evaluations_run,
                events=self._events,
            )
            self._write_observation(decisions + 1, observation)
            decision = await self.policy.decide(observation)
            _log(f"{decision.kind}: {decision.rationale}")
            decisions += 1
            finish_reason = self._execute(decision, described, decisions)
        if finish_reason is None:
            finish_reason = "decision budget exhausted"
        self._evaluate_active_if_possible()
        result = self._build_result(finish_reason)
        self._write_result(result)
        return result

    def _prepare_run(self) -> None:
        """Reset artifacts and create the canonical root node."""
        resume_dir = self.config.resume_dir
        if resume_dir is not None and resume_dir.resolve() == self.config.cache_dir.resolve():
            raise ValueError("resume_dir must differ from cache_dir")
        shutil.rmtree(self.config.cache_dir, ignore_errors=True)
        self.config.cache_dir.mkdir(parents=True)
        self.nodes.clear()
        self._by_fingerprint.clear()
        self._evaluation_by_render.clear()
        self._events.clear()
        self._transforms_applied = 0
        self._evaluations_run = 0
        root = self.environment.reset()
        if resume_dir is not None:
            self._validate_resume_root(root, resume_dir)
        self._add_node(root, parent_id=None, action_id=None, action_description=None)
        self._active_node_id = 0
        if resume_dir is not None:
            self._replay(resume_dir)

    def _validate_resume_root(self, root: KernelIR, resume_dir: Path) -> None:
        """Verify that a transcript starts from this canonical environment."""
        path = resume_dir / "nodes" / "node_000" / "node.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("fingerprint"), str):
            raise ValueError(f"resume root lacks a fingerprint: {path}")
        recorded = payload["fingerprint"]
        rendered = state_fingerprint(root)
        semantic = search_state_fingerprint(root, self.environment.legal_actions(root))
        if recorded not in {rendered, semantic}:
            raise ValueError("resume root fingerprint does not match the canonical environment reset")

    def _replay(self, resume_dir: Path) -> None:
        """Reconstruct a prior graph from canonical reset and recorded actions."""
        events_path = resume_dir / "events.jsonl"
        if not events_path.is_file():
            raise FileNotFoundError(f"resume transcript not found: {events_path}")
        prior_observations = resume_dir / "observations"
        if prior_observations.is_dir():
            shutil.copytree(prior_observations, self.config.cache_dir / "observations", dirs_exist_ok=True)
        for line in events_path.read_text(encoding="utf-8").splitlines():
            event = parse_event(line)
            self._replay_event(event, resume_dir)
            self._events.append(event)
            with (self.config.cache_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
        self._restore_missing_evaluations(resume_dir)
        if self._evaluations_run > self.config.max_evaluations:
            raise ValueError("resume transcript contains more evaluations than max_evaluations")
        if len(self._events) > self.config.max_decisions:
            raise ValueError("resume transcript contains more decisions than max_decisions")
        _log(
            f"replayed {len(self._events)} decisions, "
            f"{self._transforms_applied} transforms, "
            f"{self._evaluations_run} evaluations"
        )

    def _replay_event(self, event: SearchEvent, resume_dir: Path) -> None:
        """Apply one validated transcript event to reconstructed run state."""
        expected_decision = len(self._events) + 1
        if event.decision != expected_decision:
            raise ValueError(f"resume decision {event.decision} is not {expected_decision}")
        if event.active_before != self._active_node_id:
            raise ValueError(
                f"resume decision {event.decision} expected active "
                f"N{event.active_before:03d}, got N{self._active_node_id:03d}"
            )
        decision = AgentDecision(
            kind=event.kind,
            rationale=event.rationale,
            raw_response=event.raw_response,
            action_id=event.action_id,
            node_id=event.node_id,
        )
        if event.kind == "apply":
            self._replay_apply(decision, self._legal_actions(), event.active_after)
        elif event.kind == "evaluate":
            self._restore_evaluation(resume_dir)
        elif event.kind == "checkout":
            self._checkout(decision)
        elif event.kind != "finish":
            raise ValueError(f"unsupported resume decision {event.kind!r}")
        if event.active_after != self._active_node_id:
            raise ValueError(
                f"resume decision {event.decision} produced active "
                f"N{self._active_node_id:03d}, expected N{event.active_after:03d}"
            )

    def _replay_apply(self, decision: AgentDecision, actions: list[DescribedAction], expected_node_id: int) -> None:
        """Replay one apply using the transcript's recorded graph transition."""
        if self._transforms_applied >= self.config.max_transforms:
            raise ValueError("resume transcript contains more transforms than max_transforms")
        selected = _selected_action(decision, actions)
        parent_id = self._active_node_id
        state = self.environment.step(self.nodes[parent_id].state, selected.action)
        if expected_node_id == len(self.nodes):
            fingerprint = self._fingerprint(state)
            if fingerprint in self._by_fingerprint:
                raise ValueError(
                    f"resume apply expected new N{expected_node_id:03d}, "
                    f"but semantic state already exists as N{self._by_fingerprint[fingerprint]:03d}"
                )
            self._active_node_id = self._add_node(
                state, parent_id=parent_id, action_id=selected.action_id, action_description=selected.description
            )
            _log(f"created N{self._active_node_id:03d}: {selected.description}")
        elif 0 <= expected_node_id < len(self.nodes):
            target = self.nodes[expected_node_id]
            same_render = state_fingerprint(state) == state_fingerprint(target.state)
            same_semantics = self._fingerprint(state) == target.fingerprint
            if not same_render and not same_semantics:
                raise ValueError(f"resume apply does not reproduce deduplicated N{expected_node_id:03d}")
            self._active_node_id = expected_node_id
            _log(f"deduplicated to N{self._active_node_id:03d}: {selected.description}")
        else:
            raise ValueError(f"resume apply references unavailable N{expected_node_id:03d}")
        self._transforms_applied += 1

    def _restore_evaluation(self, resume_dir: Path) -> None:
        """Restore one cached evaluation without invoking hardware."""
        node = self.nodes[self._active_node_id]
        if node.evaluation is None:
            self._restore_node_evaluation(node, resume_dir)

    def _restore_missing_evaluations(self, resume_dir: Path) -> None:
        """Restore auto-final or otherwise unrecorded cached evaluations."""
        for node in self.nodes:
            path = resume_evaluation_path(resume_dir, node.node_id)
            if node.evaluation is None and path.is_file():
                self._restore_node_evaluation(node, resume_dir)

    def _restore_node_evaluation(self, node: SearchNode, resume_dir: Path) -> None:
        """Attach one persisted evaluation and count unique rendered kernels."""
        path = resume_evaluation_path(resume_dir, node.node_id)
        restored = parse_evaluation(path)
        render_fingerprint = state_fingerprint(node.state)
        cached = self._evaluation_by_render.get(render_fingerprint)
        if cached is not None and cached != restored:
            raise ValueError(f"render-equivalent nodes have conflicting evaluations: {path}")
        node.evaluation = restored if cached is None else cached
        if cached is None:
            self._evaluation_by_render[render_fingerprint] = restored
            self._evaluations_run += 1
        self._write_evaluation(node)

    def _legal_actions(self) -> list[DescribedAction]:
        """Describe active-state actions unless the transform budget is spent."""
        actions: list[DescribedAction] = []
        if self._transforms_applied < self.config.max_transforms:
            state = self.nodes[self._active_node_id].state
            actions = describe_actions(state, self.environment.legal_actions(state))
        return actions

    def _fingerprint(self, state: KernelIR) -> str:
        """Return semantic search identity for a state."""
        return search_state_fingerprint(state, self.environment.legal_actions(state))

    def _execute(self, decision: AgentDecision, actions: list[DescribedAction], decision_number: int) -> str | None:
        """Validate and execute one policy command."""
        before = self._active_node_id
        finish_reason: str | None = None
        if decision.kind == "apply":
            self._apply(decision, actions)
        elif decision.kind == "evaluate":
            self._evaluate_active()
        elif decision.kind == "checkout":
            self._checkout(decision)
        elif decision.kind == "finish":
            if self._evaluations_run >= self.config.min_evaluations:
                finish_reason = "policy finished"
            else:
                _log(
                    "finish deferred: "
                    f"{self._evaluations_run}/{self.config.min_evaluations} "
                    "minimum evaluations completed"
                )
        else:
            raise ValueError(f"unknown decision kind {decision.kind!r}")
        self._record_event(decision_number, before, decision)
        return finish_reason

    def _apply(self, decision: AgentDecision, actions: list[DescribedAction]) -> None:
        """Apply one listed action and activate its unique semantic state."""
        if self._transforms_applied >= self.config.max_transforms:
            raise RuntimeError("policy requested apply after transform budget was exhausted")
        selected = _selected_action(decision, actions)
        parent_id = self._active_node_id
        parent = self.nodes[parent_id]
        state = self.environment.step(parent.state, selected.action)
        fingerprint = self._fingerprint(state)
        if fingerprint in self._by_fingerprint:
            self._active_node_id = self._by_fingerprint[fingerprint]
            _log(f"deduplicated to N{self._active_node_id:03d}: {selected.description}")
        else:
            self._active_node_id = self._add_node(
                state, parent_id=parent_id, action_id=selected.action_id, action_description=selected.description
            )
            _log(f"created N{self._active_node_id:03d}: {selected.description}")
        self._transforms_applied += 1

    def _checkout(self, decision: AgentDecision) -> None:
        """Activate a previously discovered node."""
        if decision.node_id is None or decision.node_id < 0 or decision.node_id >= len(self.nodes):
            raise ValueError(f"policy selected unknown checkout node {decision.node_id!r}")
        self._active_node_id = decision.node_id

    def _evaluate_active(self) -> None:
        """Evaluate the active state once and cache its result."""
        node = self.nodes[self._active_node_id]
        if node.evaluation is not None:
            raise RuntimeError(f"policy requested evaluate for already evaluated node N{node.node_id:03d}")
        render_fingerprint = state_fingerprint(node.state)
        cached = self._evaluation_by_render.get(render_fingerprint)
        if cached is not None:
            node.evaluation = cached
            self._write_evaluation(node)
            _log(f"reused rendered evaluation for N{node.node_id:03d}: " f"{cached.message}")
        else:
            if self._evaluations_run >= self.config.max_evaluations:
                raise RuntimeError("policy requested evaluate after evaluation budget was exhausted")
            node_dir = self._node_dir(node.node_id)
            node.evaluation = self.evaluator.evaluate(node.state, node.node_id, node_dir)
            self._evaluation_by_render[render_fingerprint] = node.evaluation
            self._evaluations_run += 1
            self._write_evaluation(node)
            _log(f"evaluated N{node.node_id:03d}: {node.evaluation.message}")

    def _evaluate_active_if_possible(self) -> None:
        """Score an unevaluated final state when budget remains."""
        active = self.nodes[self._active_node_id]
        if active.evaluation is None and self._evaluations_run < self.config.max_evaluations:
            self._evaluate_active()

    def _add_node(
        self, state: KernelIR, parent_id: int | None, action_id: str | None, action_description: str | None
    ) -> int:
        """Add and persist one state, returning its node identifier."""
        fingerprint = self._fingerprint(state)
        node_id = len(self.nodes)
        node = SearchNode(
            node_id=node_id,
            state=state,
            fingerprint=fingerprint,
            parent_id=parent_id,
            action_id=action_id,
            action_description=action_description,
            evaluation=None,
        )
        self.nodes.append(node)
        self._by_fingerprint[fingerprint] = node_id
        self._write_node(node)
        return node_id

    def _record_event(self, number: int, before: int, decision: AgentDecision) -> None:
        """Append one decision to memory and the JSONL transcript."""
        event = SearchEvent(
            decision=number,
            active_before=before,
            active_after=self._active_node_id,
            kind=decision.kind,
            action_id=decision.action_id,
            node_id=decision.node_id,
            rationale=decision.rationale,
            raw_response=decision.raw_response,
        )
        self._events.append(event)
        with (self.config.cache_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")

    def _build_result(self, finish_reason: str) -> SearchResult:
        """Select the highest-scoring node and freeze run metadata."""
        scored = [node for node in self.nodes if node.evaluation is not None and node.evaluation.score is not None]
        best_node_id = max(scored, key=_successful_score).node_id if scored else None
        return SearchResult(
            nodes=tuple(self.nodes),
            active_node_id=self._active_node_id,
            best_node_id=best_node_id,
            transforms_applied=self._transforms_applied,
            evaluations_run=self._evaluations_run,
            finish_reason=finish_reason,
        )

    def _write_node(self, node: SearchNode) -> None:
        """Write the state envelope, rendered kernel, and graph metadata."""
        node_dir = self._node_dir(node.node_id)
        node.state.dump(node_dir)
        payload = {
            "node_id": node.node_id,
            "fingerprint": node.fingerprint,
            "parent_id": node.parent_id,
            "action_id": node.action_id,
            "action_description": node.action_description,
        }
        (node_dir / "node.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _write_evaluation(self, node: SearchNode) -> None:
        """Write one cached evaluation beside its state."""
        if node.evaluation is None:
            raise RuntimeError(f"node {node.node_id} has no evaluation to write")
        payload = {
            "score": node.evaluation.score,
            "metrics": node.evaluation.metrics,
            "message": node.evaluation.message,
        }
        path = self._node_dir(node.node_id) / "evaluation.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _write_observation(self, decision_number: int, observation: str) -> None:
        """Persist the exact policy-visible input for audit and replay."""
        directory = self.config.cache_dir / "observations"
        directory.mkdir(exist_ok=True)
        path = directory / f"decision_{decision_number:03d}.md"
        path.write_text(observation + "\n", encoding="utf-8")

    def _write_result(self, result: SearchResult) -> None:
        """Write selected node metadata and its root-to-node trace."""
        trace = []
        if result.best_node_id is not None:
            trace = [
                {
                    "node_id": node.node_id,
                    "action_id": node.action_id,
                    "action_description": node.action_description,
                    "score": node.evaluation.score if node.evaluation is not None else None,
                }
                for node in result.trace_to(result.best_node_id)
            ]
        payload = {
            "active_node_id": result.active_node_id,
            "best_node_id": result.best_node_id,
            "transforms_applied": result.transforms_applied,
            "evaluations_run": result.evaluations_run,
            "finish_reason": result.finish_reason,
            "trace": trace,
        }
        (self.config.cache_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _node_dir(self, node_id: int) -> Path:
        """Return the artifact directory for one node."""
        return self.config.cache_dir / "nodes" / f"node_{node_id:03d}"


def _selected_action(decision: AgentDecision, actions: list[DescribedAction]) -> DescribedAction:
    """Resolve one policy action ID against the current legal action list."""
    selected = next((item for item in actions if item.action_id == decision.action_id), None)
    if selected is None:
        raise ValueError(f"policy selected unknown action_id {decision.action_id!r}")
    return selected


def _successful_score(node: SearchNode) -> float:
    """Return a score after the caller has narrowed to successful nodes."""
    if node.evaluation is None or node.evaluation.score is None:
        raise ValueError(f"node {node.node_id} is not successfully evaluated")
    return node.evaluation.score


def _log(message: str) -> None:
    """Print one immediately visible search status line."""
    print(f"[search] {message}", flush=True)


def _validate_config(config: SearchConfig) -> None:
    """Reject inconsistent or negative search budgets."""
    budgets = {
        "max_transforms": config.max_transforms,
        "max_evaluations": config.max_evaluations,
        "min_evaluations": config.min_evaluations,
        "max_decisions": config.max_decisions,
    }
    negative = next((name for name, value in budgets.items() if value < 0), None)
    if negative is not None:
        raise ValueError(f"{negative} must be non-negative")
    if config.min_evaluations > config.max_evaluations:
        raise ValueError("min_evaluations cannot exceed max_evaluations")


__all__ = ["AgenticSearch"]
