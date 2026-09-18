"""Random-rollout correctness coverage for selected NAKB workload types.

Each workload type contributes its first registered configuration. This keeps
operation coverage broad without repeating the expensive rollout across every
shape and configuration variant. ``hf_ffn`` remains covered by synthesis and
hardware profiling tests but is excluded here because its instruction-level CPU
simulation cannot complete within this test's fixed timeout.
"""

from __future__ import annotations

import multiprocessing
import os
import random
import time
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from math import ceil
from multiprocessing.synchronize import Event

import pytest

from benchmark import NAKB_WORKLOADS, Workload, accuracy_validation
from nkigym.codegen import render
from nkigym.ir import KernelIR, build_initial_ir
from nkigym.profile import FP32SimulationCase, batch_simulate_fp32
from nkigym.synthesis import SynthesizedKernel, synthesize_torch_to_nkigym
from nkigym.transforms import public_transforms

ROLLOUT_STEPS = 500
TRANSFORMS_PER_SIMULATION = 50
SIMULATIONS_PER_ROLLOUT = ROLLOUT_STEPS // TRANSFORMS_PER_SIMULATION
TEST_TIMEOUT_SECONDS = 600
LARGE_INPUT_BYTES = 1 << 30
TRANSFORMS = public_transforms()
ROLLOUT_WORKLOADS: dict[str, Workload] = {
    f"{workload_type}_0": workloads[0]
    for workload_type, workloads in NAKB_WORKLOADS.items()
    if workload_type != "hf_ffn"
}
pytestmark = pytest.mark.timeout(TEST_TIMEOUT_SECONDS)

_SimulationWorkload = tuple[str, list[FP32SimulationCase]]
_ROLLOUT_CONTEXT = multiprocessing.get_context("spawn")
_CANCEL_ROLLOUTS = _ROLLOUT_CONTEXT.Event()


def _initialize_rollout_worker(cancel: Event) -> None:
    """Share the controller's cancellation event with each fresh worker."""
    global _CANCEL_ROLLOUTS
    _CANCEL_ROLLOUTS = cancel


def _rollout(name: str, kernel: SynthesizedKernel, seed: int, deadline: float) -> Iterator[tuple[str, KernelIR]]:
    """Keep one seeded rollout and all of its transform analysis in one process."""
    rng = random.Random(seed)
    prefix = f"{name} seed {seed}"
    state = build_initial_ir(kernel.function, kernel.input_specs)
    yield (f"{prefix} step 0", state)
    for step in range(1, ROLLOUT_STEPS + 1):
        if _CANCEL_ROLLOUTS.is_set():
            raise RuntimeError(f"{prefix} was cancelled")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{prefix} exceeded the shared rollout deadline")
        actions = [(transform, option) for transform in TRANSFORMS for option in transform.analyze(state)]
        if not actions:
            raise AssertionError(f"{prefix} terminated after {step - 1} steps")
        action = rng.choice(actions)
        state = action[0].apply(state, action[1])
        label = f"{prefix} step {step}: {type(action[0]).__name__} {action[1]!r}"
        yield (label, state)


def _generate_rollout_cases(name: str, seed: int, deadline: float) -> list[FP32SimulationCase]:
    """Render every fiftieth state from one complete rollout."""
    workload = ROLLOUT_WORKLOADS[name]
    kernel = synthesize_torch_to_nkigym(workload["torch_ref"], workload["input_specs"])
    reference_inputs = workload["input_generator"](workload["input_specs"], seed)
    inputs = kernel.adapt_inputs({name: value.copy() for name, value in reference_inputs.items()})
    expected = kernel.adapt_output(
        workload["torch_ref"](**{name: value.copy() for name, value in reference_inputs.items()})
    )
    generated_name = f"nki_{kernel.function.__name__}"
    print(f"{name} simulation_interval={TRANSFORMS_PER_SIMULATION}", flush=True)
    cases = [
        FP32SimulationCase(
            label=label,
            kernel=render(state),
            func_name=generated_name,
            inputs=inputs,
            expected=expected,
            validation=accuracy_validation(workload),
        )
        for step, (label, state) in enumerate(_rollout(name, kernel, seed, deadline))
        if step > 0 and step % TRANSFORMS_PER_SIMULATION == 0
    ]
    return cases


def _simulate_groups(
    cpu_hosts: tuple[str, ...], generated: dict[str, list[FP32SimulationCase]], deadline: float
) -> dict[str, int | Exception]:
    """Simulate compatible workload groups across the available CPU hosts."""
    groups: dict[bool, list[_SimulationWorkload]] = {}
    for name, cases in generated.items():
        if any(case.validation is None for case in cases):
            raise ValueError(f"{name}: rollout cases must carry their copied NAKB validation criteria")
        input_bytes = sum(value.nbytes for value in cases[0].inputs.values())
        groups.setdefault(input_bytes > LARGE_INPUT_BYTES, []).append((name, cases))
    results: dict[str, int | Exception] = {}
    for large_inputs, workloads in groups.items():
        cases = [case for _name, workload_cases in workloads for case in workload_cases]
        hosts = [cpu_hosts[0]] if large_inputs else list(cpu_hosts)
        timeout_s = max(1, ceil(deadline - time.monotonic()))
        print(
            f"validating {len(cases)} cases from {len(workloads)} workloads across {len(hosts)} hosts "
            f"using per-case NAKB criteria, input_bytes_over_1_gib={str(large_inputs).lower()}",
            flush=True,
        )
        try:
            completed = batch_simulate_fp32(hosts=hosts, cases=cases, atol=0.0, rtol=0.0, timeout_s=timeout_s)
            if completed != len(cases):
                raise RuntimeError(f"simulation completed {completed} of {len(cases)} cases")
            for name, workload_cases in workloads:
                results[name] = len(workload_cases)
        except Exception as error:
            for name, _workload_cases in workloads:
                results[name] = error
    return results


@pytest.fixture(scope="module")
def rollout_results(cpu_hosts: tuple[str, ...]) -> dict[str, int | Exception]:
    """Run complete rollouts in parallel processes, then simulate global batches."""
    _CANCEL_ROLLOUTS.clear()
    deadline = time.monotonic() + TEST_TIMEOUT_SECONDS
    results: dict[str, int | Exception] = {}
    generated: dict[str, list[FP32SimulationCase]] = {}
    cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    rollout_workers = min(len(ROLLOUT_WORKLOADS), cpu_count)
    print(f"concurrent_rollouts={rollout_workers}", flush=True)
    print(f"simulation_hosts={','.join(cpu_hosts)}", flush=True)
    rollout_executor = ProcessPoolExecutor(
        max_workers=rollout_workers,
        mp_context=_ROLLOUT_CONTEXT,
        initializer=_initialize_rollout_worker,
        initargs=(_CANCEL_ROLLOUTS,),
    )
    try:
        futures: dict[Future[list[FP32SimulationCase]], str] = {}
        for name in ROLLOUT_WORKLOADS:
            seed = random.SystemRandom().randrange(1 << 63)
            print(f"{name} seed {seed}", flush=True)
            futures[rollout_executor.submit(_generate_rollout_cases, name, seed, deadline)] = name
        for future in as_completed(futures, timeout=max(1, deadline - time.monotonic())):
            name = futures.pop(future)
            try:
                cases = future.result()
                if len(cases) != SIMULATIONS_PER_ROLLOUT:
                    raise AssertionError(f"{name}: generated {len(cases)} cases, expected {SIMULATIONS_PER_ROLLOUT}")
                generated[name] = cases
            except Exception as error:
                results[name] = error
    except BaseException:
        _CANCEL_ROLLOUTS.set()
        raise
    finally:
        cancelled = _CANCEL_ROLLOUTS.is_set()
        rollout_executor.shutdown(wait=not cancelled, cancel_futures=cancelled)
    results.update(_simulate_groups(cpu_hosts, generated, deadline))
    return results


@pytest.mark.parametrize("workload_name", [pytest.param(name, id=name) for name in ROLLOUT_WORKLOADS])
def test_one_random_rollout_per_selected_workload_type_preserves_every_fiftieth_kernel(
    workload_name: str, rollout_results: dict[str, int | Exception]
) -> None:
    """Apply 500 random transforms and CPU-simulate every fiftieth state."""
    result = rollout_results[workload_name]
    if isinstance(result, Exception):
        raise result
    assert result == SIMULATIONS_PER_ROLLOUT
