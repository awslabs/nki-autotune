"""Random-rollout correctness coverage for every discovered workload."""

from __future__ import annotations

import os
import random
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import cast

from config import SIMULATION_HOSTS

from kernel_library import WORKLOADS
from nkigym.codegen import render
from nkigym.environment import KernelMDP
from nkigym.ir import KernelIR
from nkigym.profile import FP32SimulationCase, batch_simulate_fp32
from nkigym.synthesis import SynthesizedKernel, synthesize_numpy_to_nkigym
from nkigym.transforms import TransformOption, public_transforms

ROLLOUT_STEPS = 500
TRANSFORMS_PER_SIMULATION = 10
ANALYZER_WORKERS = 4
SIMULATION_WORKERS_PER_HOST = 32
BATCH_SIMULATION_TIMEOUT_SECONDS = 7200
TRANSFORMS = public_transforms()
AnalysisResult = tuple[int, tuple[TransformOption, ...], float]


@dataclass(frozen=True)
class RolloutTask:
    """Pair one discovered workload with its rollout seed."""

    name: str
    seed: int


def _run_seed() -> int:
    """Return a fresh run seed."""
    return random.SystemRandom().randrange(1 << 63)


def _analyze_transform_group(state: KernelIR, indices: tuple[int, ...]) -> tuple[AnalysisResult, ...]:
    """Analyze one transform group and retain per-transform timing."""
    results: list[AnalysisResult] = []
    for index in indices:
        started = time.perf_counter()
        options = cast(tuple[TransformOption, ...], tuple(TRANSFORMS[index].analyze(state)))
        results.append((index, options, time.perf_counter() - started))
    return tuple(results)


def _analyzer_ready() -> None:
    """Start analyzer processes before the renderer thread is created."""


def _analysis_groups(weights: list[float], worker_count: int) -> tuple[tuple[int, ...], ...]:
    """Greedily balance transform analyzers by their latest measured costs."""
    groups: list[list[int]] = [[] for _worker in range(worker_count)]
    loads = [0.0 for _worker in range(worker_count)]
    for index in sorted(range(len(weights)), key=lambda item: (-weights[item], item)):
        worker = min(range(worker_count), key=lambda item: (loads[item], item))
        groups[worker].append(index)
        loads[worker] += weights[index]
    return tuple(tuple(sorted(group)) for group in groups)


def _parallel_rollout(
    name: str, kernel: SynthesizedKernel, seed: int, executor: ProcessPoolExecutor
) -> Iterator[tuple[str, KernelIR]]:
    """Yield one seeded rollout while analyzing registered transforms in parallel."""
    environment = KernelMDP(kernel.function, kernel.input_specs, transforms=TRANSFORMS)
    rng = random.Random(seed)
    prefix = f"{name} seed {seed}"
    state = environment.reset()
    weights = [1.0 for _transform in TRANSFORMS]
    yield (f"{prefix} step 0", state)
    for step in range(1, ROLLOUT_STEPS + 1):
        groups = _analysis_groups(weights, ANALYZER_WORKERS)
        futures = [executor.submit(_analyze_transform_group, state, group) for group in groups]
        analyzed = tuple(result for future in futures for result in future.result())
        options_by_index: dict[int, tuple[TransformOption, ...]] = {}
        for index, options, elapsed_s in analyzed:
            options_by_index[index] = options
            weights[index] = elapsed_s
        actions = [
            (TRANSFORMS[index], option) for index in range(len(TRANSFORMS)) for option in options_by_index[index]
        ]
        if not actions:
            raise AssertionError(f"{prefix} terminated after {step - 1} steps")
        action = rng.choice(actions)
        state = environment.step(state, action)
        label = f"{prefix} step {step}: {type(action[0]).__name__} {action[1]!r}"
        yield (label, state)


def _generate_rollout_cases(task: RolloutTask) -> list[FP32SimulationCase]:
    """Render every tenth state in one complete rollout."""
    workload = WORKLOADS[task.name]
    kernel = synthesize_numpy_to_nkigym(workload["numpy_ref"], workload["input_specs"])
    reference_inputs = workload["input_generator"](workload["input_specs"], task.seed)
    inputs = kernel.adapt_inputs({name: value.copy() for name, value in reference_inputs.items()})
    expected = kernel.adapt_output(
        workload["numpy_ref"](**{name: value.copy() for name, value in reference_inputs.items()})
    )
    generated_name = f"nki_{kernel.function.__name__}"
    worker_count = min(ANALYZER_WORKERS, len(TRANSFORMS))
    with ProcessPoolExecutor(max_workers=worker_count) as analyzer:
        analyzer.submit(_analyzer_ready).result()
        cases = [
            FP32SimulationCase(
                label=label, kernel=render(state), func_name=generated_name, inputs=inputs, expected=expected
            )
            for step, (label, state) in enumerate(_parallel_rollout(task.name, kernel, task.seed, analyzer))
            if step > 0 and step % TRANSFORMS_PER_SIMULATION == 0
        ]
    return cases


def _simulate_cases(hosts: list[str], cases: list[FP32SimulationCase], atol: float, rtol: float) -> int:
    """Shard one tolerance group across hosts and validate every case."""
    shards = [cases[index :: len(hosts)] for index in range(len(hosts))]
    assignments = [(host, shard) for host, shard in zip(hosts, shards, strict=True) if shard]
    with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
        futures = [
            executor.submit(
                batch_simulate_fp32,
                hosts=[host],
                cases=shard,
                atol=atol,
                rtol=rtol,
                timeout_s=BATCH_SIMULATION_TIMEOUT_SECONDS,
                workers_per_host=SIMULATION_WORKERS_PER_HOST,
            )
            for host, shard in assignments
        ]
        completed = sum(future.result() for future in futures)
    return completed


def test_one_random_rollout_per_workload_preserves_every_tenth_kernel() -> None:
    """Apply 500 random transforms and CPU-simulate every tenth state."""
    run_seed = _run_seed()
    rng = random.Random(run_seed)
    print(f"random_rollout_seed={run_seed}", flush=True)
    tasks: list[RolloutTask] = []
    for name in WORKLOADS:
        seed = rng.randrange(1 << 63)
        print(f"{name} seed {seed}", flush=True)
        tasks.append(RolloutTask(name=name, seed=seed))
    worker_count = min(len(tasks), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        case_groups = list(executor.map(_generate_rollout_cases, tasks, chunksize=1))
    cases_by_tolerance: dict[tuple[float, float], list[FP32SimulationCase]] = {}
    for task, cases in zip(tasks, case_groups, strict=True):
        assert len(cases) == ROLLOUT_STEPS // TRANSFORMS_PER_SIMULATION
        workload = WORKLOADS[task.name]
        cases_by_tolerance.setdefault((workload["atol"], workload["rtol"]), []).extend(cases)
    hosts: list[str] = list(SIMULATION_HOSTS)
    print(f"simulation_hosts={','.join(hosts)}", flush=True)
    completed = 0
    for (atol, rtol), cases in cases_by_tolerance.items():
        print(f"validating {len(cases)} cases with atol={atol:g}, rtol={rtol:g}", flush=True)
        completed += _simulate_cases(hosts, cases, atol, rtol)
    assert completed == len(tasks) * (ROLLOUT_STEPS // TRANSFORMS_PER_SIMULATION)
