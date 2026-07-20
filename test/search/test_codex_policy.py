"""Tests for isolated Codex policy command construction."""

from pathlib import Path

from autotune.search.codex_policy import CodexPolicyConfig, CodexTransformPolicy, _latest_evaluation_decision


def test_codex_policy_command_disables_external_context(tmp_path: Path) -> None:
    """Policy turns inherit authentication but expose no repository tools."""
    policy = CodexTransformPolicy(
        CodexPolicyConfig(
            executable="codex",
            model="openai.gpt-5.6-sol",
            model_provider="amazon-bedrock",
            reasoning_effort="max",
            timeout_s=600,
        )
    )
    command = policy._command(tmp_path, tmp_path / "schema.json", tmp_path / "output.json")

    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert 'web_search="disabled"' in command
    assert "mcp_servers={}" in command
    assert 'model_provider="amazon-bedrock"' in command
    disabled = {command[index + 1] for index, value in enumerate(command[:-1]) if value == "--disable"}
    assert {"apps", "browser_use", "computer_use", "multi_agent", "plugins", "shell_tool", "unified_exec"} <= disabled
    assert command[-1] == "-"


def test_latest_evaluation_decision_tracks_strategy_refresh_point() -> None:
    """Only completed evaluate operations trigger a strategy refresh."""
    observation = """\
# Decision History
- D001: apply N000->N001; split a loop
- D002: evaluate N001->N001; measure candidate
- D003: checkout N001->N000; try another branch
"""
    assert _latest_evaluation_decision(observation) == 2
