"""Byte-exact gates for transform recipes and explicit manual matmul ladders."""

from __future__ import annotations

import inspect
import secrets
from pathlib import Path
from test.transforms import _matmul_lhs_rhs_manual as lhs_manual_ladder
from test.transforms import _matmul_lhsT_rhs_manual as lhs_t_manual_ladder
from test.transforms._ladder_compare import assert_matches_render_ordered
from test.transforms._matmul_lhs_rhs_ladder import _build_ladder as build_lhs_ladder
from test.transforms._matmul_lhsT_rhs_ladder import _build_ladder as build_lhs_t_ladder
from types import ModuleType

from autotune.search.profile_evaluator import ProfileEvaluatorConfig
from autotune.search.ssh_profile_evaluator import SSHNKIProfileEvaluator, SSHProfileConfig
from nkigym.codegen import render
from nkigym.ir import KernelIR

MIN_LHS_T_MFU_PERCENT = 90.0
LHS_T_INPUT_SPECS = {"lhs_T": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")}
SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")


def _assert_ladder_matches(ladder: list[tuple[str, KernelIR]], manual_ladder: ModuleType) -> None:
    """Assert every transform-produced rung matches its explicit NKI kernel."""
    for name, ir in ladder:
        hand_kernel = getattr(manual_ladder, name)
        try:
            assert_matches_render_ordered(render(ir), inspect.getsource(hand_kernel))
        except AssertionError as error:
            raise AssertionError(f"{manual_ladder.__name__}.{name}") from error


def _profile_lhs_t_endpoint(endpoint: KernelIR, cache_dir: Path) -> float:
    """Profile the generated endpoint on a real Trn2 NeuronCore."""
    evaluator = SSHNKIProfileEvaluator(
        profile_config=ProfileEvaluatorConfig(
            input_specs=LHS_T_INPUT_SPECS,
            output_shape=(2048, 2048),
            neuron_platform_target="trn2",
            neuronx_cc_args=SCHEDULER_OFF_ARGS,
            seed=secrets.randbits(32),
        ),
        ssh_config=SSHProfileConfig(
            host="gym-1",
            local_repo=Path(__file__).resolve().parents[2],
            remote_repo_subdir=".cache/nki-autotune-manual-ladder/repo",
            remote_cache_subdir=".cache/nki-autotune-manual-ladder/profiles",
            remote_activate='source "$HOME"/venvs/kernel-env/bin/activate',
            timeout_s=1800,
        ),
    )
    evaluation = evaluator.evaluate(endpoint, node_id=35, cache_dir=cache_dir)
    if evaluation.score is None:
        raise AssertionError(evaluation.message)
    return evaluation.score


def test_lhs_t_transform_ladder_matches_manual_ladder_and_reaches_90_percent_mfu(tmp_path: Path) -> None:
    """All 36 ``lhs_T.T @ rhs`` states match by hand and the endpoint reaches 90% MFU."""
    ladder = build_lhs_t_ladder()
    assert [name for name, _ir in ladder] == [f"kernel_{index}" for index in range(36)]
    _assert_ladder_matches(ladder, lhs_t_manual_ladder)
    mfu_percent = _profile_lhs_t_endpoint(ladder[-1][1], tmp_path)
    assert (
        mfu_percent >= MIN_LHS_T_MFU_PERCENT
    ), f"generated kernel_35 measured {mfu_percent:.2f}% MFU, expected at least {MIN_LHS_T_MFU_PERCENT:.2f}%"


def test_lhs_transform_ladder_is_byte_exact_with_manual_ladder() -> None:
    """All 32 ``lhs @ rhs`` transform states match the hand ladder."""
    ladder = build_lhs_ladder()
    assert [name for name, _ir in ladder] == [f"kernel_{index}" for index in range(32)]
    _assert_ladder_matches(ladder, lhs_manual_ladder)
