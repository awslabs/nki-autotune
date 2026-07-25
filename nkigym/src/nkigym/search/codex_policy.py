"""Tool-free Codex policy for choosing one next transform."""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from nkigym.search.policy_json import DECISION_SCHEMA, REASONING_POLICY_PROMPT, parse_decision
from nkigym.search.types import AgentDecision

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
    """Ask one isolated Codex turn for each next transform."""

    def __init__(self, config: CodexPolicyConfig) -> None:
        """Store subprocess controls."""
        self.config = config

    async def decide(self, observation: str) -> AgentDecision:
        """Run the blocking CLI request in a worker thread."""
        return await asyncio.to_thread(self._decide_sync, observation)

    def _decide_sync(self, observation: str) -> AgentDecision:
        """Request and parse one apply-or-finish decision."""
        reply = self._request(REASONING_POLICY_PROMPT + "\n\n" + observation)
        decision = parse_decision(reply)
        return replace(decision, raw_response=reply)

    def _request(self, prompt: str) -> str:
        """Run one tool-free structured Codex request."""
        with tempfile.TemporaryDirectory(prefix="nkigym-codex-transform-") as temp:
            temp_path = Path(temp)
            schema_path = temp_path / "decision.schema.json"
            output_path = temp_path / "decision.json"
            schema_path.write_text(json.dumps(DECISION_SCHEMA), encoding="utf-8")
            command = self._command(temp_path, schema_path, output_path)
            completed = subprocess.run(
                command, input=prompt, text=True, capture_output=True, timeout=self.config.timeout_s, check=False
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"codex transform decision failed with exit {completed.returncode}: " f"{completed.stderr[-2000:]}"
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


__all__ = ["CodexPolicyConfig", "CodexTransformPolicy"]
