"""Replay recorded NKIGym ladders and check correctness and aggregate latency."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

from kernel_library import BEST_NKIGYM_LADDERS, NAKB_WORKLOADS
from nkigym.codegen import render
from nkigym.ir import build_initial_ir
from nkigym.ir.program_sharding import configured_program_shards
from nkigym.profile import FP32SimulationCase, batch_simulate_fp32, profile_metrics
from nkigym.synthesis import synthesize_torch_to_nkigym

RELATIVE_LATENCY_TARGET = 0.9


def _matches_nakb(actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> bool:
    """Apply the workload's NAKB correctness rule."""
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


def test_nkigym_ladders(cpu_hosts: tuple[str, ...], trn2_hosts: tuple[str, ...], tmp_path: Path) -> None:
    """Replay every recorded ladder and check correctness and aggregate relative latency."""
    workloads = [
        (f"{workload_type}_{workload_index}", workload)
        for workload_type, grouped_workloads in NAKB_WORKLOADS.items()
        for workload_index, workload in enumerate(grouped_workloads)
    ]
    configured_seed = os.environ.get("NKIGYM_NAKB_VALIDATION_SEED")
    seed = int(configured_seed) if configured_seed is not None else random.SystemRandom().randrange(1 << 63)
    nkigym_total = 0.0
    nakb_total = 0.0
    print(f"nakb_validation_seed={seed}", flush=True)

    for workload_index, (workload_name, workload) in enumerate(workloads):
        artifact = synthesize_torch_to_nkigym(workload["torch_ref"], workload["input_specs"])
        reference_inputs = workload["input_generator"](workload["input_specs"], seed)
        inputs = artifact.adapt_inputs({name: value.copy() for name, value in reference_inputs.items()})
        expected = artifact.adapt_output(
            workload["torch_ref"](**{name: value.copy() for name, value in reference_inputs.items()})
        )
        func_name = f"nki_{artifact.function.__name__}"
        ir = build_initial_ir(artifact.function, artifact.input_specs)
        simulation_cases: list[FP32SimulationCase] = []
        for step_index, (transform, option) in enumerate(BEST_NKIGYM_LADDERS.get(workload_name, ()), start=1):
            assert option in transform.analyze(
                ir
            ), f"{workload_name} step {step_index}: {type(transform).__name__} did not discover {option!r}"
            ir = transform.apply(ir, option)
            simulation_cases.append(
                FP32SimulationCase(
                    label=f"{workload_name} step {step_index}",
                    kernel=render(ir),
                    func_name=func_name,
                    inputs=inputs,
                    expected=expected,
                )
            )
        if simulation_cases:
            completed = batch_simulate_fp32(
                hosts=list(cpu_hosts), cases=simulation_cases, atol=workload["atol"], rtol=workload["rtol"]
            )
            assert completed == len(simulation_cases)

        metrics = profile_metrics(
            host=trn2_hosts[workload_index % len(trn2_hosts)],
            kernel=render(ir),
            func_name=func_name,
            input_specs=artifact.input_specs,
            cache_dir=tmp_path / workload_name,
            lnc=max((1, *configured_program_shards(ir).values())),
            confirmation=True,
            inputs=inputs,
        )
        expected_outputs = expected if isinstance(expected, tuple) else (expected,)
        assert len(metrics.outputs) == len(
            expected_outputs
        ), f"{workload_name}: hardware returned {len(metrics.outputs)} outputs, expected {len(expected_outputs)}"
        for output_index, (actual, golden) in enumerate(zip(metrics.outputs, expected_outputs, strict=True)):
            assert _matches_nakb(
                actual, golden, workload["atol"], workload["rtol"]
            ), f"{workload_name} output {output_index} failed NAKB correctness"

        nkigym_total += metrics.latency_ms
        nakb_total += workload["nakb_latency_ms"]
        print(
            f"{workload_name}: nkigym_latency_ms={metrics.latency_ms:.9f} "
            f"nakb_latency_ms={workload['nakb_latency_ms']:.9f}",
            flush=True,
        )

    relative_latency = nkigym_total / nakb_total
    print(
        f"relative_latency={relative_latency:.6f} nkigym_total_ms={nkigym_total:.9f} "
        f"nakb_total_ms={nakb_total:.9f}",
        flush=True,
    )
    assert (
        relative_latency <= RELATIVE_LATENCY_TARGET
    ), f"relative latency {relative_latency:.6f} exceeds target {RELATIVE_LATENCY_TARGET:.6f}"
