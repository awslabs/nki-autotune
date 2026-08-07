"""Run profiler-guided refinement for one persisted nkigym search program."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from nkigym.search.profiled_refinement import ReasoningEffort, run_profiled_refinement
from nkigym.search.program import load_nkigym_program


def _parser() -> argparse.ArgumentParser:
    """Build the standalone refinement parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--program-dir", required=True, type=Path)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"), default="high")
    parser.add_argument("--max-reasoning-steps", type=int)
    parser.add_argument("--target-score", type=float)
    parser.add_argument("--profile-timeout-seconds", type=int, default=1800)
    parser.add_argument("--policy-timeout-seconds", type=int, default=600)
    parser.add_argument("--lnc", type=int, choices=(1, 2), default=1)
    parser.add_argument("--codex-executable", default="codex")
    return parser


def _run(arguments: argparse.Namespace) -> None:
    """Load and refine one persisted nkigym callable."""
    program_dir = Path(arguments.program_dir).expanduser().resolve()
    trace_dir = Path(arguments.trace_dir).expanduser().resolve()
    program, kernel = load_nkigym_program(program_dir)
    run_profiled_refinement(
        kernel,
        program.input_specs,
        str(arguments.host),
        trace_dir,
        workload_guidance=program.workload_guidance,
        target_score=(None if arguments.target_score is None else float(arguments.target_score)),
        neuronx_cc_args=program.neuronx_cc_args,
        reasoning_effort=cast(ReasoningEffort, arguments.reasoning_effort),
        max_reasoning_steps=(None if arguments.max_reasoning_steps is None else int(arguments.max_reasoning_steps)),
        profile_timeout_s=int(arguments.profile_timeout_seconds),
        policy_timeout_s=int(arguments.policy_timeout_seconds),
        lnc=int(arguments.lnc),
        codex_executable=str(arguments.codex_executable),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run refinement and return a process exit code."""
    arguments = _parser().parse_args(argv)
    _run(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
