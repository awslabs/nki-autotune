"""GPT reasoning policy backed by non-interactive ``codex exec``."""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from autotune.search.policy_json import (
    DECISION_SCHEMA,
    REASONING_POLICY_PROMPT,
    STRATEGY_POLICY_PROMPT,
    STRATEGY_SCHEMA,
    parse_decision,
)
from autotune.search.types import AgentDecision

_DISABLED_FEATURES = (
    "apps",
    "browser_use",
    "computer_use",
    "goals",
    "hooks",
    "image_generation",
    "multi_agent",
    "plugins",
    "remote_plugin",
    "shell_tool",
    "tool_suggest",
    "unified_exec",
    "workspace_dependencies",
)


@dataclass(frozen=True)
class CodexPolicyConfig:
    """Controls for isolated, read-only Codex policy turns."""

    executable: str
    model: str
    model_provider: str | None
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"]
    timeout_s: int


class CodexTransformPolicy:
    """Use isolated GPT turns with persistent high-level strategy memory."""

    def __init__(self, config: CodexPolicyConfig) -> None:
        """Store subprocess controls."""
        self.config = config
        self._strategy: str | None = None
        self._last_reviewed_evaluation = 0

    async def decide(self, observation: str) -> AgentDecision:
        """Run the blocking CLI policy in a worker thread."""
        return await asyncio.to_thread(self._decide_sync, observation)

    def _decide_sync(self, observation: str) -> AgentDecision:
        """Refresh strategy when needed, then request one legal operation."""
        latest_evaluation = _latest_evaluation_decision(observation)
        if self._strategy is None or latest_evaluation > self._last_reviewed_evaluation:
            self._strategy = self._review_strategy(observation)
            self._last_reviewed_evaluation = latest_evaluation
        prompt = REASONING_POLICY_PROMPT + "\n\n# Current Strategy Review\n" + self._strategy + "\n\n" + observation
        reply = self._request(prompt, DECISION_SCHEMA, "decision")
        decision = parse_decision(reply)
        audit_response = json.dumps(
            {"strategy": json.loads(self._strategy), "decision_response": reply}, sort_keys=True
        )
        return replace(decision, raw_response=audit_response)

    def _review_strategy(self, observation: str) -> str:
        """Create or refresh a concise portfolio after measured feedback."""
        prior = self._strategy if self._strategy is not None else "No prior strategy."
        prompt = STRATEGY_POLICY_PROMPT + "\n\n# Prior Strategy\n" + prior + "\n\n" + observation
        reply = self._request(prompt, STRATEGY_SCHEMA, "strategy")
        payload = json.loads(reply)
        if not isinstance(payload, dict):
            raise ValueError("strategy response must be a JSON object")
        return json.dumps(payload, indent=2, sort_keys=True)

    def _request(self, prompt: str, schema: dict[str, object], label: str) -> str:
        """Run one tool-free structured Codex request."""
        with tempfile.TemporaryDirectory(prefix=f"autotune-codex-{label}-") as temp:
            temp_path = Path(temp)
            schema_path = temp_path / f"{label}.schema.json"
            output_path = temp_path / f"{label}.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            command = self._command(temp_path, schema_path, output_path)
            completed = subprocess.run(
                command, input=prompt, text=True, capture_output=True, timeout=self.config.timeout_s, check=False
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"codex {label} failed with exit {completed.returncode}: " f"{completed.stderr[-2000:]}"
                )
            reply = output_path.read_text(encoding="utf-8") if output_path.is_file() else completed.stdout
        return reply

    def _command(self, working_dir: Path, schema_path: Path, output_path: Path) -> list[str]:
        """Build a tool-free Codex command that inherits authentication only."""
        command = [
            self.config.executable,
            "exec",
            "--ignore-user-config",
            "--model",
            self.config.model,
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--skip-git-repo-check",
            "--ignore-rules",
            "--color",
            "never",
            "--cd",
            str(working_dir),
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "-c",
            f'model_reasoning_effort="{self.config.reasoning_effort}"',
            "-c",
            'web_search="disabled"',
            "-c",
            "mcp_servers={}",
        ]
        if self.config.model_provider is not None:
            command.extend(["-c", f"model_provider={json.dumps(self.config.model_provider)}"])
        for feature in _DISABLED_FEATURES:
            command.extend(["--disable", feature])
        command.append("-")
        return command


def _latest_evaluation_decision(observation: str) -> int:
    """Return the latest recorded evaluation decision, or zero before any."""
    matches = re.findall(r"^- D(\d+): evaluate ", observation, flags=re.MULTILINE)
    return int(matches[-1]) if matches else 0


__all__ = ["CodexPolicyConfig", "CodexTransformPolicy"]
