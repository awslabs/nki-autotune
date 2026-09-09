"""Test-owned Codex search over public NKIGym transform ladders."""

from __future__ import annotations

import difflib
import inspect
import json
import math
import os
import shutil
import signal
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

from nkigym.codegen import render
from nkigym.ir import KernelIR, build_initial_ir
from nkigym.ir.program_sharding import configured_program_shards
from nkigym.profile import InputSpecs, ProfileMetrics, profile_metrics
from nkigym.transforms import Transform, TransformLegalityError, TransformOption, public_transforms

_PROFILE_FIELDS = (
    "total_active_time",
    "dma_active_time",
    "tensor_engine_active_time",
    "vector_engine_active_time",
    "scalar_engine_active_time",
    "gpsimd_engine_active_time",
    "hbm_read_bytes",
    "hbm_write_bytes",
    "event_count",
    "matmul_instruction_count",
)


@dataclass(frozen=True)
class AgenticLadderResult:
    """Fastest measured kernel found within one fixed search budget."""

    best_latency_ms: float
    best_kernel: str
    best_ladder: tuple[dict[str, object], ...]
    reasoning_steps: int
    profiles_run: int
    finish_reason: str
    trace_dir: Path
    lnc: int


@dataclass(frozen=True)
class _Action:
    """One legal public transform option from one retained state."""

    action_id: str
    transform: Transform[Any]
    option: TransformOption
    payload: dict[str, object]
    diff_file: str
    diff_preview: tuple[str, ...]


@dataclass
class _State:
    """One retained IR state with optional hardware feedback."""

    state_id: int
    ir: KernelIR
    kernel: str
    parent_state_id: int | None
    step: dict[str, object] | None
    actions: tuple[_Action, ...]
    metrics: ProfileMetrics | None = None
    profile_error: str | None = None


@dataclass(frozen=True)
class _Decision:
    """One validated Codex policy decision."""

    decision: str
    state_id: int | None
    action_id: str | None
    rationale: str


class AgenticLadderBuilder:
    """Retain, branch, and profile transform-only kernel states."""

    def __init__(
        self,
        kernel_func: Callable[..., Any],
        input_specs: InputSpecs,
        profile_host: str,
        trace_dir: Path,
        max_reasoning_steps: int,
        max_profiles: int,
        profile_timeout_s: int,
        policy_timeout_s: int,
        codex_executable: str,
        codex_model: str | None,
    ) -> None:
        """Create one bounded workload search."""
        if not callable(kernel_func) or not profile_host.strip() or not codex_executable.strip():
            raise ValueError("kernel, profile host, and Codex executable are required")
        for name, value in (
            ("max_reasoning_steps", max_reasoning_steps),
            ("max_profiles", max_profiles),
            ("profile_timeout_s", profile_timeout_s),
            ("policy_timeout_s", policy_timeout_s),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._initial_ir = build_initial_ir(kernel_func, input_specs)
        self._input_specs = dict(input_specs)
        self._host = profile_host
        self._trace_dir = trace_dir
        self._max_reasoning_steps = max_reasoning_steps
        self._max_profiles = max_profiles
        self._profile_timeout_s = profile_timeout_s
        self._policy_timeout_s = policy_timeout_s
        self._codex_executable = codex_executable
        self._codex_model = codex_model
        self._transforms = tuple(public_transforms())
        self._states: dict[int, _State] = {}
        self._attempted_edges: set[tuple[int, str]] = set()
        self._active_state_id = 0
        self._reasoning_steps = 0
        self._profiles_run = 0

    def run(self) -> AgenticLadderResult:
        """Run until Codex finishes or a fixed budget is exhausted."""
        self._prepare_trace()
        root = self._record_state(self._initial_ir, None, None)
        self._profile_state(root.state_id)
        finish_reason = "reasoning_budget_exhausted"
        while self._reasoning_steps < self._max_reasoning_steps:
            active = self._states[self._active_state_id]
            actions = self._available_actions(active)
            focusable = self._focusable_state_ids()
            profile_allowed = not self._has_profile_result(active) and self._profiles_run < self._max_profiles
            if not actions and not focusable and not profile_allowed:
                finish_reason = "no_untried_actions"
                break
            decision = self._policy_decision(active, actions, focusable, profile_allowed)
            self._reasoning_steps += 1
            if decision.decision == "apply":
                self._apply_decision(active, actions, decision)
            elif decision.decision == "profile":
                self._profile_state(cast(int, decision.state_id))
            elif decision.decision == "focus":
                self._active_state_id = cast(int, decision.state_id)
            else:
                finish_reason = "agent_finished"
                break
            if self._profiles_run >= self._max_profiles and not self._measured_states():
                finish_reason = "profile_budget_exhausted"
                break
        active = self._states[self._active_state_id]
        if not self._has_profile_result(active) and self._profiles_run < self._max_profiles:
            self._profile_state(active.state_id)
        best = self._best_state()
        result = AgenticLadderResult(
            best_latency_ms=cast(ProfileMetrics, best.metrics).latency_ms,
            best_kernel=best.kernel,
            best_ladder=self._ladder(best.state_id),
            reasoning_steps=self._reasoning_steps,
            profiles_run=self._profiles_run,
            finish_reason=finish_reason,
            trace_dir=self._trace_dir,
            lnc=max((1, *configured_program_shards(best.ir).values())),
        )
        (self._trace_dir / "best_kernel.py").write_text(result.best_kernel, encoding="utf-8")
        self._write_json(self._trace_dir / "best_ladder.json", list(result.best_ladder))
        self._write_json(
            self._trace_dir / "search_result.json",
            {
                "best_latency_ms": result.best_latency_ms,
                "reasoning_steps": result.reasoning_steps,
                "profiles_run": result.profiles_run,
                "finish_reason": result.finish_reason,
                "lnc": result.lnc,
            },
        )
        return result

    def _prepare_trace(self) -> None:
        """Create an empty deterministic trace tree."""
        shutil.rmtree(self._trace_dir, ignore_errors=True)
        self._trace_dir.mkdir(parents=True)
        for name in ("states", "reasoning", "profiles"):
            (self._trace_dir / name).mkdir()
        self._write_json(
            self._trace_dir / "transforms.json",
            [
                {"name": type(transform).__name__, "contract": inspect.getdoc(type(transform)) or ""}
                for transform in self._transforms
            ],
        )

    def _record_state(self, ir: KernelIR, parent_state_id: int | None, step: dict[str, object] | None) -> _State:
        """Render, enumerate, and persist one state."""
        state_id = len(self._states)
        kernel = render(ir)
        state_dir = self._trace_dir / "states" / f"state_{state_id:04d}"
        state_dir.mkdir()
        (state_dir / "kernel.py").write_text(kernel, encoding="utf-8")
        actions: list[_Action] = []
        actions_dir = state_dir / "actions"
        actions_dir.mkdir()
        for transform in self._transforms:
            for option in transform.analyze(ir):
                child_kernel = render(transform.apply(ir, option))
                action_id = f"s{state_id:04d}_a{len(actions):05d}"
                diff = "".join(
                    difflib.unified_diff(
                        kernel.splitlines(keepends=True),
                        child_kernel.splitlines(keepends=True),
                        fromfile=f"state_{state_id:04d}",
                        tofile=action_id,
                        n=2,
                    )
                )
                diff_name = f"actions/{action_id}.diff"
                (state_dir / diff_name).write_text(diff, encoding="utf-8")
                changes = tuple(
                    line[:240]
                    for line in diff.splitlines()
                    if (line.startswith("+") and not line.startswith("+++"))
                    or (line.startswith("-") and not line.startswith("---"))
                )
                actions.append(
                    _Action(
                        action_id=action_id,
                        transform=transform,
                        option=option,
                        payload=_option_payload(option),
                        diff_file=f"states/state_{state_id:04d}/{diff_name}",
                        diff_preview=changes[:4],
                    )
                )
        state = _State(state_id, ir, kernel, parent_state_id, step, tuple(actions))
        self._states[state_id] = state
        self._write_state(state)
        return state

    def _write_state(self, state: _State) -> None:
        """Persist one state summary and legal action catalog."""
        state_dir = self._trace_dir / "states" / f"state_{state.state_id:04d}"
        self._write_json(
            state_dir / "actions.json",
            [
                {
                    "action_id": action.action_id,
                    "transform": type(action.transform).__name__,
                    "option": action.payload,
                    "diff_file": action.diff_file,
                    "diff_preview": list(action.diff_preview),
                }
                for action in state.actions
            ],
        )
        self._write_json(state_dir / "state.json", self._state_payload(state))

    def _profile_state(self, state_id: int) -> None:
        """Profile one source-distinct state and retain failures."""
        state = self._states[state_id]
        duplicate = next(
            (
                candidate
                for candidate in self._states.values()
                if candidate.state_id != state_id
                and candidate.kernel == state.kernel
                and self._has_profile_result(candidate)
            ),
            None,
        )
        if duplicate is not None:
            state.metrics, state.profile_error = duplicate.metrics, duplicate.profile_error
            self._write_state(state)
            return
        if self._profiles_run >= self._max_profiles:
            raise RuntimeError("profile budget exhausted")
        profile_id = self._profiles_run
        self._profiles_run += 1
        try:
            state.metrics = profile_metrics(
                host=self._host,
                kernel=state.kernel,
                func_name=f"nki_{state.ir.func_name}",
                input_specs=self._input_specs,
                cache_dir=self._trace_dir / "profiles" / f"profile_{profile_id:04d}",
                lnc=max((1, *configured_program_shards(state.ir).values())),
                timeout_s=self._profile_timeout_s,
            )
        except RuntimeError as error:
            if _is_infrastructure_failure(str(error)):
                raise
            state.profile_error = str(error)
        self._write_state(state)

    def _available_actions(self, state: _State) -> dict[str, _Action]:
        """Return every untried edge from one state."""
        return {
            action.action_id: action
            for action in state.actions
            if (state.state_id, action.action_id) not in self._attempted_edges
        }

    def _focusable_state_ids(self) -> tuple[int, ...]:
        """Return retained branch points with untried actions."""
        return tuple(
            state.state_id
            for state in self._states.values()
            if state.state_id != self._active_state_id and self._available_actions(state)
        )

    def _apply_decision(self, state: _State, actions: Mapping[str, _Action], decision: _Decision) -> None:
        """Apply one selected currently legal transform."""
        action = actions[cast(str, decision.action_id)]
        try:
            if action.option not in action.transform.analyze(state.ir):
                raise TransformLegalityError("recorded option is no longer legal")
            child_ir = action.transform.apply(state.ir, action.option)
        except TransformLegalityError as error:
            raise RuntimeError(f"Codex selected illegal action {action.action_id}: {error}") from error
        self._attempted_edges.add((state.state_id, action.action_id))
        step = {"transform": type(action.transform).__name__, "option": action.payload, "rationale": decision.rationale}
        self._active_state_id = self._record_state(child_ir, state.state_id, step).state_id

    def _policy_decision(
        self, active: _State, actions: Mapping[str, _Action], focusable: tuple[int, ...], profile_allowed: bool
    ) -> _Decision:
        """Ask one isolated Codex subagent for the next measured-search action."""
        step_dir = self._trace_dir / "reasoning" / f"step_{self._reasoning_steps:04d}"
        step_dir.mkdir()
        choices = ["finish"]
        if actions:
            choices.append("apply")
        if profile_allowed:
            choices.append("profile")
        if focusable:
            choices.append("focus")
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "decision": {"enum": choices},
                "state_id": {"type": ["integer", "null"]},
                "action_id": {"enum": [None, *actions]},
                "rationale": {"type": "string", "minLength": 1},
            },
            "required": ["decision", "state_id", "action_id", "rationale"],
        }
        context = {
            "active_state": self._state_payload(active),
            "active_kernel_file": f"states/state_{active.state_id:04d}/kernel.py",
            "available_actions": [
                {
                    "action_id": action.action_id,
                    "transform": type(action.transform).__name__,
                    "option": action.payload,
                    "diff_file": action.diff_file,
                    "diff_preview": list(action.diff_preview[:1]),
                }
                for action in actions.values()
            ],
            "focusable_state_ids": list(focusable),
            "best_state_id": None if not self._measured_states() else self._best_state().state_id,
            "profiles_run": self._profiles_run,
            "profiles_remaining": self._max_profiles - self._profiles_run,
            "reasoning_steps_remaining": self._max_reasoning_steps - self._reasoning_steps,
        }
        schema_path, response_path = step_dir / "schema.json", step_dir / "response.json"
        self._write_json(schema_path, schema)
        self._write_json(step_dir / "context.json", context)
        prompt = _policy_prompt(context)
        (step_dir / "prompt.md").write_text(prompt, encoding="utf-8")
        command = _policy_command(
            self._codex_executable, self._codex_model, self._trace_dir, schema_path, response_path
        )
        _run_policy(command, prompt, step_dir / "events.jsonl", step_dir / "stderr.log", self._policy_timeout_s)
        decision = _parse_decision(response_path, active.state_id, actions, focusable, profile_allowed)
        self._write_json(step_dir / "decision.json", vars(decision))
        return decision

    def _state_payload(self, state: _State) -> dict[str, object]:
        """Return compact evidence for policy and trace consumers."""
        metrics = state.metrics
        return {
            "state_id": state.state_id,
            "parent_state_id": state.parent_state_id,
            "step": state.step,
            "status": "measured" if metrics else "failed" if state.profile_error else "unprofiled",
            "latency_ms": None if metrics is None else metrics.latency_ms,
            "mfu_percent": None if metrics is None else metrics.mfu_percent,
            "profile_error": None if state.profile_error is None else state.profile_error[-1500:],
            "profile_highlights": (
                {}
                if metrics is None
                else {
                    name: metrics.profiler_summary[name] for name in _PROFILE_FIELDS if name in metrics.profiler_summary
                }
            ),
            "legal_action_count": len(self._available_actions(state)),
        }

    def _has_profile_result(self, state: _State) -> bool:
        """Return whether profiling succeeded or failed for one state."""
        return state.metrics is not None or state.profile_error is not None

    def _measured_states(self) -> tuple[_State, ...]:
        """Return successfully measured states."""
        return tuple(state for state in self._states.values() if state.metrics is not None)

    def _best_state(self) -> _State:
        """Return the lowest-latency measured state."""
        measured = self._measured_states()
        if not measured:
            raise RuntimeError("agentic search produced no measurable kernel")
        return min(measured, key=lambda state: (cast(ProfileMetrics, state.metrics).latency_ms, state.state_id))

    def _ladder(self, state_id: int) -> tuple[dict[str, object], ...]:
        """Return the root-to-state transform sequence."""
        reverse: list[dict[str, object]] = []
        state = self._states[state_id]
        while state.parent_state_id is not None:
            if state.step is None:
                raise RuntimeError(f"state {state.state_id} has a parent but no transform step")
            reverse.append(state.step)
            state = self._states[state.parent_state_id]
        reverse.reverse()
        return tuple(reverse)

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        """Write deterministic JSON artifacts."""
        path.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_value(value: object) -> object:
    """Convert supported trace values to strict JSON."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"trace contains non-finite float {value!r}")
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_value(item) for item in sorted(value, key=repr)]
    raise TypeError(f"trace value of type {type(value).__name__} is not JSON-compatible")


def _option_payload(option: TransformOption) -> dict[str, object]:
    """Serialize one transform option."""
    if not is_dataclass(option):
        raise TypeError(f"transform option {type(option).__name__} is not a dataclass")
    payload: dict[str, object] = {"type": type(option).__name__}
    payload.update({field.name: _json_value(getattr(option, field.name)) for field in fields(option)})
    return payload


def _policy_prompt(context: dict[str, object]) -> str:
    """Return the minimal transform-only optimization prompt."""
    return f"""Use the current NKIGym transforms to derive the fastest kernel you can find.
You may use the available Neuron Explorer profiler information.
Current search state:
{json.dumps(context, indent=2, sort_keys=True)}
"""


def _policy_command(
    executable: str, model: str | None, trace_dir: Path, schema_path: Path, response_path: Path
) -> tuple[str, ...]:
    """Build one isolated Codex command."""
    command = [
        executable,
        "exec",
        "--ephemeral",
        "--json",
        "--color",
        "never",
        "--sandbox",
        "read-only",
        "--cd",
        str(trace_dir),
        "--skip-git-repo-check",
        "--ignore-rules",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(response_path),
        "-c",
        'model_reasoning_effort="max"',
        "-c",
        'web_search="disabled"',
        "-c",
        "mcp_servers={}",
        "-c",
        'approval_policy="never"',
    ]
    if model is not None:
        command.extend(("--model", model))
    command.append("-")
    return tuple(command)


def _run_policy(command: tuple[str, ...], prompt: str, event_path: Path, stderr_path: Path, timeout_s: int) -> None:
    """Execute one bounded Codex policy turn."""
    with event_path.open("w", encoding="utf-8") as events, stderr_path.open("w", encoding="utf-8") as errors:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=events, stderr=errors, text=True, start_new_session=True
        )
        try:
            process.communicate(prompt, timeout=timeout_s)
        except subprocess.TimeoutExpired as error:
            _terminate(process)
            raise TimeoutError(f"Codex policy exceeded {timeout_s} seconds") from error
        except KeyboardInterrupt:
            _terminate(process)
            raise
    if process.returncode != 0:
        raise RuntimeError(f"Codex policy exited with {process.returncode}: {stderr_path.read_text()[-3000:]}")


def _terminate(process: subprocess.Popen[str]) -> None:
    """Terminate one Codex process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.communicate(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()


def _parse_decision(
    response_path: Path,
    active_state_id: int,
    actions: Mapping[str, _Action],
    focusable: tuple[int, ...],
    profile_allowed: bool,
) -> _Decision:
    """Parse and validate one structured policy response."""
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Codex response must be an object")
    decision = _Decision(
        decision=cast(str, payload.get("decision")),
        state_id=cast(int | None, payload.get("state_id")),
        action_id=cast(str | None, payload.get("action_id")),
        rationale=cast(str, payload.get("rationale")),
    )
    valid = bool(decision.rationale)
    if decision.decision == "apply":
        valid = valid and decision.state_id in (None, active_state_id) and decision.action_id in actions
    elif decision.decision == "profile":
        valid = valid and profile_allowed and decision.state_id == active_state_id and decision.action_id is None
    elif decision.decision == "focus":
        valid = valid and decision.state_id in focusable and decision.action_id is None
    elif decision.decision == "finish":
        valid = valid and decision.action_id is None
    else:
        valid = False
    if not valid:
        raise ValueError(f"invalid Codex policy decision: {payload}")
    return decision


def _is_infrastructure_failure(message: str) -> bool:
    """Return whether one profile failure is transport rather than kernel evidence."""
    markers = ("SSH profile failed", "SSH execution failed", "returned no result.json", "profiler returned no summary")
    return any(marker in message for marker in markers)


__all__ = ["AgenticLadderBuilder", "AgenticLadderResult"]
