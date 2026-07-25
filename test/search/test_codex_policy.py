"""Tests for isolated Codex next-transform requests."""

from pathlib import Path

from nkigym.search.codex_policy import CodexPolicyConfig, CodexTransformPolicy


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
