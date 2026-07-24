"""Tests for converting runner output into search scores."""

from pathlib import Path

from autotune.runner.output import ProfileOutput
from autotune.runner.types import KernelJob, ProfileResult
from autotune.search.profile_evaluator import NKIProfileEvaluator, ProfileEvaluatorConfig
from examples.random_rollout import LHS_T_RHS
from nkigym.ir import build_initial_ir

WORKLOAD = LHS_T_RHS


def test_profile_evaluator_uses_mfu_as_score(tmp_path: Path) -> None:
    """Successful profile output maps MFU percent to the maximized score."""

    def fake_profile(
        kernels: dict[str, KernelJob],
        cache_dir: str,
        seed: int,
        neuron_platform_target: str,
        collect_detailed_profile: bool,
    ) -> ProfileOutput:
        """Return one successful hardware result."""
        return ProfileOutput(
            results=[
                ProfileResult(
                    kernel_name="node_000",
                    hardware_output="[2048, 2048] bfloat16",
                    profiler_summary={
                        "total_time": 0.001,
                        "mfu_estimated_percent": 0.91,
                        "mbu_estimated_percent": 0.12,
                    },
                )
            ],
            compiler_logs={},
            elapsed_s=2.0,
            cache_dir=str(tmp_path),
        )

    evaluator = NKIProfileEvaluator(
        config=ProfileEvaluatorConfig(
            input_specs=WORKLOAD.input_specs,
            output_shape=(2048, 2048),
            neuron_platform_target="trn2",
            neuronx_cc_args=(),
            seed=0,
        ),
        profile_runner=fake_profile,
    )
    evaluation = evaluator.evaluate(build_initial_ir(WORKLOAD.f_nkigym, WORKLOAD.input_specs), 0, tmp_path)

    assert evaluation.score == 91.0
    assert evaluation.metrics["total_time_s"] == 0.001


def test_profile_evaluator_reports_specific_compiler_diagnostic(tmp_path: Path) -> None:
    """Failed profiles expose the compiler cause instead of wrapper traceback."""

    def fake_profile(
        kernels: dict[str, KernelJob],
        cache_dir: str,
        seed: int,
        neuron_platform_target: str,
        collect_detailed_profile: bool,
    ) -> ProfileOutput:
        """Return one compile failure with a more specific compiler log."""
        return ProfileOutput(
            results=[ProfileResult(kernel_name="node_000", hardware_output="CalledProcessError: compiler exited 70")],
            compiler_logs={
                "node_000": ("ERROR backend failed\n" "[NCC_INLA001] Allocated memory out of bound in PSUM\n")
            },
            elapsed_s=2.0,
            cache_dir=str(tmp_path),
        )

    evaluator = NKIProfileEvaluator(
        config=ProfileEvaluatorConfig(
            input_specs=WORKLOAD.input_specs,
            output_shape=(2048, 2048),
            neuron_platform_target="trn2",
            neuronx_cc_args=(),
            seed=0,
        ),
        profile_runner=fake_profile,
    )
    evaluation = evaluator.evaluate(build_initial_ir(WORKLOAD.f_nkigym, WORKLOAD.input_specs), 0, tmp_path)

    assert evaluation.score is None
    assert "[NCC_INLA001] Allocated memory out of bound in PSUM" in evaluation.message
