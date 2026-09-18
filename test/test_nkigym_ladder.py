"""Replay recorded NKIGym ladders and check correctness and mean relative latency."""

from __future__ import annotations

import os
import random
from pathlib import Path
from statistics import fmean

from benchmark import NAKB_WORKLOADS, accuracy_validation, validate_nakb_outputs
from kernel_library import BEST_NKIGYM_LADDERS
from nkigym.codegen import render
from nkigym.ir import build_initial_ir
from nkigym.ir.program_sharding import configured_program_shards
from nkigym.profile import FP32SimulationCase, batch_simulate_fp32, profile_metrics
from nkigym.synthesis import synthesize_torch_to_nkigym

MEAN_RELATIVE_LATENCY_TARGET = 0.9


def test_nkigym_ladders(cpu_hosts: tuple[str, ...], trn2_hosts: tuple[str, ...], tmp_path: Path) -> None:
    """Replay every recorded ladder and check correctness and mean relative latency."""
    workloads = [
        (f"{workload_type}_{workload_index}", workload)
        for workload_type, grouped_workloads in NAKB_WORKLOADS.items()
        for workload_index, workload in enumerate(grouped_workloads)
    ]
    unknown_ladders = sorted(set(BEST_NKIGYM_LADDERS) - {name for name, _ in workloads})
    assert not unknown_ladders, f"best NKIGym ladders reference unknown workloads: {', '.join(unknown_ladders)}"
    configured_seed = os.environ.get("NKIGYM_NAKB_VALIDATION_SEED")
    seed = int(configured_seed) if configured_seed is not None else random.SystemRandom().randrange(1 << 63)
    relative_latencies: list[float] = []
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
                    validation=accuracy_validation(workload),
                )
            )
        if simulation_cases:
            completed = batch_simulate_fp32(
                hosts=list(cpu_hosts),
                cases=simulation_cases,
                atol=workload["accuracy"].atol,
                rtol=workload["accuracy"].rtol,
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
        validate_nakb_outputs(metrics.outputs, expected, inputs, accuracy_validation(workload)[1])

        relative_latencies.append(metrics.latency_ms / workload["nakb_latency_ms"])
        print(
            f"{workload_name}: nkigym_latency_ms={metrics.latency_ms:.9f} "
            f"nakb_latency_ms={workload['nakb_latency_ms']:.9f}",
            flush=True,
        )

    mean_relative_latency = fmean(relative_latencies)
    print(
        f"mean_relative_latency={mean_relative_latency:.6f} "
        f"mean_latency_reduction_percent={100.0 * (1.0 - mean_relative_latency):.6f}",
        flush=True,
    )
    assert (
        mean_relative_latency <= MEAN_RELATIVE_LATENCY_TARGET
    ), f"mean relative latency {mean_relative_latency:.6f} exceeds target {MEAN_RELATIVE_LATENCY_TARGET:.6f}"
