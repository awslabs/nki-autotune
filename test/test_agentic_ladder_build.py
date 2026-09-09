"""End-to-end agentic transform-ladder relative latency over the NAKB registry."""

from __future__ import annotations

import ast
import json
import os
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from _agentic_ladder import AgenticLadderBuilder

from kernel_library import NAKB_WORKLOADS, Workload
from nkigym.profile import profile_metrics
from nkigym.synthesis import synthesize_torch_to_nkigym

RELATIVE_LATENCY_TARGET = 0.9
ALL_WORKLOADS = {
    f"{workload_type}_{workload_index}": workload
    for workload_type, workloads in NAKB_WORKLOADS.items()
    for workload_index, workload in enumerate(workloads)
}
WORKLOAD_NAMES = tuple(ALL_WORKLOADS)


@dataclass(frozen=True)
class _WorkloadResult:
    """One workload's accepted agentic latency or retained failure."""

    workload_name: str
    nakb_latency_ms: float
    nkigym_latency_ms: float | None
    trace_dir: Path
    error: str | None


def _positive_int_env(name: str, default: int) -> int:
    """Return one positive integer environment control."""
    raw = os.environ.get(name)
    value = default if raw is None else int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validation_seed() -> int:
    """Return a fresh 63-bit seed or one explicitly supplied for reproduction."""
    configured = os.environ.get("NKIGYM_NAKB_VALIDATION_SEED")
    seed = random.SystemRandom().randrange(1 << 63) if configured is None else int(configured)
    if seed < 0 or seed >= 1 << 63:
        raise ValueError("NKIGYM_NAKB_VALIDATION_SEED must be in [0, 2**63)")
    return seed


VALIDATION_SEED = _validation_seed()


def _trace_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a retained caller path or one pytest-managed trace root."""
    configured = os.environ.get("NKIGYM_AGENTIC_TRACE_ROOT")
    root = (
        Path(configured).expanduser().resolve()
        if configured is not None
        else tmp_path_factory.mktemp("agentic-ladder-relative-latency")
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _nakb_comparison(actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> bool:
    """Apply NAKB's global-maximum tolerance with shape, dtype, and finiteness checks."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    try:
        actual_float = actual.astype(np.float32)
        expected_float = expected.astype(np.float32)
    except (TypeError, ValueError):
        return bool(np.array_equal(actual.view(np.uint8), expected.view(np.uint8)))
    if np.any(np.isfinite(actual_float) != np.isfinite(expected_float)):
        return False
    finite_expected = np.abs(expected_float[np.isfinite(expected_float)])
    threshold = atol + rtol * float(finite_expected.max(initial=0.0))
    with np.errstate(invalid="ignore"):
        difference = np.abs(actual_float - expected_float)
    matching_infinities = (
        np.isinf(actual_float) & np.isinf(expected_float) & (np.sign(actual_float) == np.sign(expected_float))
    )
    return bool(np.all(np.where(matching_infinities, 0.0, difference) <= threshold))


def _validate_kernel_source(kernel: str, func_name: str, parameter_names: tuple[str, ...]) -> None:
    """Require one safe NAKB-compatible NKI entry point with the derived ABI."""
    module = ast.parse(kernel)
    functions = [node for node in ast.walk(module) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(functions) != 1:
        raise AssertionError(f"kernel must define exactly one function, found {len(functions)}")
    function = functions[0]
    decorators = [ast.unparse(decorator) for decorator in function.decorator_list]
    actual_parameters = tuple(argument.arg for argument in function.args.args)
    if function.name != func_name or decorators != ["nki.jit"] or actual_parameters != parameter_names:
        raise AssertionError(
            f"kernel entry point must be @nki.jit {func_name}{parameter_names}, "
            f"found decorators={decorators} name={function.name} parameters={actual_parameters}"
        )
    forbidden_import = next(
        (
            node
            for node in ast.walk(module)
            if (
                isinstance(node, ast.Import)
                and any(alias.name == "nki.compiler" or alias.name.startswith("nki.compiler.") for alias in node.names)
            )
            or (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and (node.module == "nki.compiler" or node.module.startswith("nki.compiler."))
            )
        ),
        None,
    )
    if forbidden_import is not None or "NEURON_CC_FLAGS" in kernel:
        raise AssertionError("kernel must not use nki.compiler internals or manipulate NEURON_CC_FLAGS")


def _validate_final_kernel(
    workload_name: str, workload: Workload, kernel: str, lnc: int, host: str, trace_dir: Path
) -> float:
    """Run NAKB hardware numerics and return confirmed latency for one winner."""
    artifact = synthesize_torch_to_nkigym(workload["torch_ref"], workload["input_specs"])
    func_name = f"nki_{artifact.function.__name__}"
    _validate_kernel_source(kernel, func_name, tuple(artifact.input_specs))
    reference_inputs = workload["input_generator"](workload["input_specs"], VALIDATION_SEED)
    inputs = artifact.adapt_inputs({name: value.copy() for name, value in reference_inputs.items()})
    expected = artifact.adapt_output(
        workload["torch_ref"](**{name: value.copy() for name, value in reference_inputs.items()})
    )
    metrics = profile_metrics(
        host=host,
        kernel=kernel,
        func_name=func_name,
        input_specs=artifact.input_specs,
        cache_dir=trace_dir / "acceptance",
        lnc=lnc,
        confirmation=True,
        inputs=inputs,
    )
    expected_outputs = expected if isinstance(expected, tuple) else (expected,)
    if len(metrics.outputs) != len(expected_outputs):
        raise AssertionError(
            f"{workload_name}: hardware returned {len(metrics.outputs)} outputs, expected {len(expected_outputs)}"
        )
    comparisons = [
        _nakb_comparison(actual, golden, workload["atol"], workload["rtol"])
        for actual, golden in zip(metrics.outputs, expected_outputs, strict=True)
    ]
    if not all(comparisons):
        raise AssertionError(f"{workload_name}: NAKB correctness failed for outputs {comparisons}")
    return metrics.latency_ms


@pytest.fixture(scope="module")
def agentic_results(
    trn2_hosts: tuple[str, ...], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, _WorkloadResult]:
    """Run one independent bounded Codex ladder search per registered workload."""
    trace_root = _trace_root(tmp_path_factory)
    results: dict[str, _WorkloadResult] = {}
    print(f"nakb_validation_seed={VALIDATION_SEED}", flush=True)
    controls = {
        "max_reasoning_steps": _positive_int_env("NKIGYM_AGENTIC_MAX_REASONING_STEPS", 64),
        "max_profiles": _positive_int_env("NKIGYM_AGENTIC_MAX_PROFILES", 32),
        "profile_timeout_s": _positive_int_env("NKIGYM_AGENTIC_PROFILE_TIMEOUT_S", 1800),
        "policy_timeout_s": _positive_int_env("NKIGYM_AGENTIC_POLICY_TIMEOUT_S", 600),
        "codex_executable": os.environ.get("NKIGYM_CODEX_EXECUTABLE", "codex"),
        "codex_model": os.environ.get("NKIGYM_AGENTIC_MODEL"),
    }
    for workload_index, (workload_name, workload) in enumerate(ALL_WORKLOADS.items()):
        trace_dir = trace_root / workload_name
        host = trn2_hosts[workload_index % len(trn2_hosts)]
        try:
            artifact = synthesize_torch_to_nkigym(workload["torch_ref"], workload["input_specs"])
            search = AgenticLadderBuilder(
                kernel_func=artifact.function,
                input_specs=artifact.input_specs,
                profile_host=host,
                trace_dir=trace_dir,
                **controls,
            ).run()
            latency_ms = _validate_final_kernel(
                workload_name, workload, search.best_kernel, search.lnc, host, trace_dir
            )
            payload = {
                "workload": workload_name,
                "validation_seed": VALIDATION_SEED,
                "nakb_latency_ms": workload["nakb_latency_ms"],
                "nkigym_latency_ms": latency_ms,
                "search_latency_ms": search.best_latency_ms,
                "ladder_length": len(search.best_ladder),
                "reasoning_steps": search.reasoning_steps,
                "profiles_run": search.profiles_run,
                "finish_reason": search.finish_reason,
                "error": None,
            }
            result = _WorkloadResult(workload_name, workload["nakb_latency_ms"], latency_ms, trace_dir, None)
        except Exception:
            error = traceback.format_exc()
            payload = {
                "workload": workload_name,
                "validation_seed": VALIDATION_SEED,
                "nakb_latency_ms": workload["nakb_latency_ms"],
                "nkigym_latency_ms": None,
                "error": error,
            }
            result = _WorkloadResult(workload_name, workload["nakb_latency_ms"], None, trace_dir, error)
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        results[workload_name] = result
    successful = [result for result in results.values() if result.nkigym_latency_ms is not None]
    relative_latency = (
        sum(cast(float, result.nkigym_latency_ms) for result in successful)
        / sum(result.nakb_latency_ms for result in successful)
        if len(successful) == len(results)
        else None
    )
    (trace_root / "relative_latency.json").write_text(
        json.dumps(
            {
                "relative_latency": relative_latency,
                "relative_latency_target": RELATIVE_LATENCY_TARGET,
                "validation_seed": VALIDATION_SEED,
                "workload_count": len(results),
                "successful_workload_count": len(successful),
                "failed_workloads": [result.workload_name for result in results.values() if result.error is not None],
                "controls": controls,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return results


@pytest.mark.parametrize("workload_name", [pytest.param(name, id=name) for name in WORKLOAD_NAMES])
def test_agentic_ladder_builds_nakb_correct_kernel(
    workload_name: str, agentic_results: dict[str, _WorkloadResult]
) -> None:
    """Every workload produces one NAKB-correct measurable NKIGym kernel."""
    result = agentic_results[workload_name]
    assert result.error is None, result.error
    assert result.nkigym_latency_ms is not None and result.nkigym_latency_ms > 0.0


def test_agentic_relative_latency(agentic_results: dict[str, _WorkloadResult]) -> None:
    """Aggregate NKIGym/NAKB relative latency is at most the fixed target."""
    failures = [result.workload_name for result in agentic_results.values() if result.error is not None]
    assert not failures, f"agentic search failed workloads: {', '.join(failures)}"
    nkigym_total = sum(cast(float, result.nkigym_latency_ms) for result in agentic_results.values())
    nakb_total = sum(result.nakb_latency_ms for result in agentic_results.values())
    relative_latency = nkigym_total / nakb_total
    assert relative_latency <= RELATIVE_LATENCY_TARGET, (
        f"relative latency {relative_latency:.6f} exceeds target {RELATIVE_LATENCY_TARGET:.6f}; "
        f"NKIGym={nkigym_total:.9f} ms NAKB={nakb_total:.9f} ms"
    )
