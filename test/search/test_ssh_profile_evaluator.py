"""Tests for SSH profile transport validation and parsing."""

import json
from pathlib import Path

import pytest

from autotune.search.profile_evaluator import ProfileEvaluatorConfig
from autotune.search.ssh_profile_evaluator import SSHNKIProfileEvaluator, SSHProfileConfig, _parse_evaluations


def test_parse_evaluations_extracts_labeled_batch_from_mixed_output() -> None:
    """The final result marker wins over ordinary remote process output."""
    payload = {
        "evaluations": {
            "candidate": {"score": 87.5, "metrics": {"mfu_percent": 87.5, "compiled": True}, "message": "Trn2 success"},
            "failed": {"score": None, "metrics": {"wallclock_s": 2.0}, "message": "Trn2 failure"},
        }
    }
    stdout = "compiler log\n" "AUTOTUNE_PROFILE_RESULT=" + json.dumps(payload) + "\n"

    evaluations = _parse_evaluations(stdout)

    assert evaluations["candidate"].score == 87.5
    assert evaluations["candidate"].metrics["compiled"] is True
    assert evaluations["failed"].score is None


def test_ssh_evaluator_rejects_parent_traversal(tmp_path: Path) -> None:
    """Remote synchronization roots must stay below the remote home."""
    with pytest.raises(ValueError, match="remote_repo_subdir"):
        SSHNKIProfileEvaluator(
            profile_config=ProfileEvaluatorConfig(
                input_specs={"x": ((128, 128), "bfloat16")},
                output_shape=(128, 128),
                neuron_platform_target="trn2",
                neuronx_cc_args=(),
                seed=0,
            ),
            ssh_config=SSHProfileConfig(
                host="gym-1",
                local_repo=tmp_path,
                remote_repo_subdir="../repo",
                remote_cache_subdir=".cache/profiles",
                remote_activate="true",
                timeout_s=60,
            ),
        )
