"""Combinatorial random-rollout correctness coverage."""

from __future__ import annotations

import os
import random
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import cast

import numpy as np
from config import SIMULATION_HOSTS

from kernel_library import Workload
from kernel_library.attention_q16384_kv16384_d128 import HEAD_DIM
from kernel_library.attention_q16384_kv16384_d128 import WORKLOAD as ATTENTION_WORKLOAD
from kernel_library.attention_q16384_kv16384_d128 import make_input_specs as make_attention_input_specs
from kernel_library.matmul_lhs_rhs_m2048_k2048_n2048 import WORKLOAD as MATMUL_LHS_WORKLOAD
from kernel_library.matmul_lhs_t_rhs_m2048_k2048_n2048 import WORKLOAD as MATMUL_LHS_T_WORKLOAD
from kernel_library.rmsnorm_matmul_m2048_k2048_n2048 import WORKLOAD as RMSNORM_MATMUL_WORKLOAD
from kernel_library.rmsnorm_matmul_m2048_k2048_n2048 import make_f_nkigym as make_rmsnorm_matmul_f_nkigym
from nkigym.codegen import render
from nkigym.environment import KernelMDP
from nkigym.ir import KernelIR
from nkigym.ops.matmul import NKIMatmul
from nkigym.profile import FP32SimulationCase, batch_simulate_fp32
from nkigym.transforms import TransformOption, public_transforms

ROLLOUT_STEPS = 500
SHAPES_PER_WORKLOAD = 5
ANALYZER_WORKERS = 4
SIMULATION_WORKERS_PER_HOST = 32
BATCH_SIMULATION_TIMEOUT_SECONDS = 7200
TILE_SIZE = NKIMatmul.MIN_TILE_SIZE["K"]
MAX_EXTENT_TILES = 8
MAX_MATMUL_TILE_VOLUME = 60
MAX_ATTENTION_TILE_AREA = 20
TRANSFORMS = public_transforms()
AnalysisResult = tuple[int, tuple[TransformOption, ...], float]


@dataclass(frozen=True)
class RolloutWorkload:
    """Randomized shape metadata paired with one library workload."""

    family: str
    dimensions: tuple[int, ...]
    name: str
    definition: Workload


@dataclass(frozen=True)
class RolloutTask:
    """Serializable description of one independently generated rollout."""

    family: str
    dimensions: tuple[int, ...]
    seed: int


def _lhs_t_workload(k: int, m: int, n: int) -> RolloutWorkload:
    """Build one pretransposed matmul workload."""
    definition = Workload(
        input_specs={"lhs_T": ((k, m), "bfloat16"), "rhs": ((k, n), "bfloat16")},
        f_numpy=MATMUL_LHS_T_WORKLOAD.f_numpy,
        f_nkigym=MATMUL_LHS_T_WORKLOAD.f_nkigym,
        input_generator=MATMUL_LHS_T_WORKLOAD.input_generator,
        atol=MATMUL_LHS_T_WORKLOAD.atol,
        rtol=MATMUL_LHS_T_WORKLOAD.rtol,
    )
    return RolloutWorkload(
        family="matmul_lhs_t",
        dimensions=(k, m, n),
        name=f"random_matmul_lhsT_rhs_k{k}_m{m}_n{n}",
        definition=definition,
    )


def _lhs_workload(k: int, m: int, n: int) -> RolloutWorkload:
    """Build one row-major matmul workload."""
    definition = Workload(
        input_specs={"lhs": ((m, k), "bfloat16"), "rhs": ((k, n), "bfloat16")},
        f_numpy=MATMUL_LHS_WORKLOAD.f_numpy,
        f_nkigym=MATMUL_LHS_WORKLOAD.f_nkigym,
        input_generator=MATMUL_LHS_WORKLOAD.input_generator,
        atol=MATMUL_LHS_WORKLOAD.atol,
        rtol=MATMUL_LHS_WORKLOAD.rtol,
    )
    return RolloutWorkload(
        family="matmul_lhs", dimensions=(k, m, n), name=f"random_matmul_lhs_rhs_k{k}_m{m}_n{n}", definition=definition
    )


def _attention_workload(query_length: int, sequence_length: int) -> RolloutWorkload:
    """Build one attention workload."""
    definition = Workload(
        input_specs=make_attention_input_specs(query_length, sequence_length, HEAD_DIM),
        f_numpy=ATTENTION_WORKLOAD.f_numpy,
        f_nkigym=ATTENTION_WORKLOAD.f_nkigym,
        input_generator=ATTENTION_WORKLOAD.input_generator,
        atol=ATTENTION_WORKLOAD.atol,
        rtol=ATTENTION_WORKLOAD.rtol,
    )
    return RolloutWorkload(
        family="attention",
        dimensions=(query_length, sequence_length),
        name=f"random_attention_q{query_length}_s{sequence_length}",
        definition=definition,
    )


def _rmsnorm_matmul_workload(k: int, m: int, n: int) -> RolloutWorkload:
    """Build one RMSNorm+matmul workload."""
    definition = Workload(
        input_specs={"lhs": ((m, k), "bfloat16"), "rhs": ((k, n), "bfloat16")},
        f_numpy=RMSNORM_MATMUL_WORKLOAD.f_numpy,
        f_nkigym=make_rmsnorm_matmul_f_nkigym(k),
        input_generator=RMSNORM_MATMUL_WORKLOAD.input_generator,
        atol=RMSNORM_MATMUL_WORKLOAD.atol,
        rtol=RMSNORM_MATMUL_WORKLOAD.rtol,
    )
    return RolloutWorkload(
        family="rmsnorm_matmul",
        dimensions=(k, m, n),
        name=f"random_rmsnorm_matmul_k{k}_m{m}_n{n}",
        definition=definition,
    )


def _run_seed() -> int:
    """Return a fresh run seed."""
    seed = random.SystemRandom().randrange(1 << 63)
    return seed


def _matmul_n_tile_factors() -> tuple[int, ...]:
    """Return N factors accepted by the canonical matmul tile contract."""
    maximum = NKIMatmul.MAX_TILE_SIZE["N"]
    if maximum is None:
        factors = tuple(range(1, MAX_EXTENT_TILES + 1))
    else:
        factors = tuple(
            factor
            for factor in range(1, MAX_EXTENT_TILES + 1)
            if (factor * TILE_SIZE) % min(factor * TILE_SIZE, maximum) == 0
        )
    return factors


def _random_matmul_shapes(rng: random.Random, count: int) -> tuple[tuple[int, int, int], ...]:
    """Sample unique aligned K, M, N shapes within the existing cost envelope."""
    shapes: list[tuple[int, int, int]] = []
    n_factors = _matmul_n_tile_factors()
    while len(shapes) < count:
        factors = (rng.randint(1, MAX_EXTENT_TILES), rng.randint(1, MAX_EXTENT_TILES), rng.choice(n_factors))
        shape = (factors[0] * TILE_SIZE, factors[1] * TILE_SIZE, factors[2] * TILE_SIZE)
        if factors[0] * factors[1] * factors[2] <= MAX_MATMUL_TILE_VOLUME and shape not in shapes:
            shapes.append(shape)
    return tuple(shapes)


def _random_attention_shapes(rng: random.Random, count: int) -> tuple[tuple[int, int], ...]:
    """Sample unique aligned query and sequence lengths within the cost envelope."""
    shapes: list[tuple[int, int]] = []
    sequence_factors = _matmul_n_tile_factors()
    while len(shapes) < count:
        factors = (rng.randint(1, MAX_EXTENT_TILES), rng.choice(sequence_factors))
        shape = (factors[0] * TILE_SIZE, factors[1] * TILE_SIZE)
        if factors[0] * factors[1] <= MAX_ATTENTION_TILE_AREA and shape not in shapes:
            shapes.append(shape)
    return tuple(shapes)


def _random_workloads(rng: random.Random) -> tuple[RolloutWorkload, ...]:
    """Build five randomized shapes for each of the four workload families."""
    lhs_t_shapes = _random_matmul_shapes(rng, SHAPES_PER_WORKLOAD)
    lhs_shapes = _random_matmul_shapes(rng, SHAPES_PER_WORKLOAD)
    attention_shapes = _random_attention_shapes(rng, SHAPES_PER_WORKLOAD)
    rmsnorm_shapes = _random_matmul_shapes(rng, SHAPES_PER_WORKLOAD)
    workloads = (
        *(_lhs_t_workload(k, m, n) for k, m, n in lhs_t_shapes),
        *(_lhs_workload(k, m, n) for k, m, n in lhs_shapes),
        *(_attention_workload(query_length, sequence_length) for query_length, sequence_length in attention_shapes),
        *(_rmsnorm_matmul_workload(k, m, n) for k, m, n in rmsnorm_shapes),
    )
    return workloads


def _simulation_hosts() -> list[str]:
    """Return the configured CPU simulation host pool."""
    return list(SIMULATION_HOSTS)


def _rollout(workload: RolloutWorkload, seed: int) -> Iterator[tuple[str, KernelIR]]:
    """Yield the initial state and every action selected by ``seed``."""
    definition = workload.definition
    environment = KernelMDP(definition.f_nkigym, definition.input_specs, transforms=TRANSFORMS)
    rng = random.Random(seed)
    prefix = f"{workload.name} seed {seed}"
    state = environment.reset()
    yield (f"{prefix} step 0", state)
    for step in range(1, ROLLOUT_STEPS + 1):
        actions = environment.legal_actions(state)
        if not actions:
            raise AssertionError(f"{prefix} terminated after {step - 1} steps")
        action = rng.choice(actions)
        state = environment.step(state, action)
        label = f"{prefix} step {step}: {type(action[0]).__name__} {action[1]!r}"
        yield (label, state)


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
    workload: RolloutWorkload, seed: int, executor: ProcessPoolExecutor
) -> Iterator[tuple[str, KernelIR]]:
    """Yield the exact seeded rollout while analyzing transforms in parallel."""
    definition = workload.definition
    environment = KernelMDP(definition.f_nkigym, definition.input_specs, transforms=TRANSFORMS)
    rng = random.Random(seed)
    prefix = f"{workload.name} seed {seed}"
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


def _simulation_cases(
    workload: RolloutWorkload, seed: int, states: Iterator[tuple[str, KernelIR]]
) -> list[FP32SimulationCase]:
    """Render every rollout state into a remotely executable validation case."""
    definition = workload.definition
    inputs = definition.generate_inputs(seed)
    expected = np.asarray(definition.f_numpy(**inputs))
    generated_name = f"nki_{definition.f_nkigym.__name__}"
    cases = [
        FP32SimulationCase(
            label=label, kernel=render(state), func_name=generated_name, inputs=inputs, expected=expected
        )
        for label, state in states
    ]
    return cases


def _workload_for_task(task: RolloutTask) -> RolloutWorkload:
    """Rebuild one workload from its process-safe descriptor."""
    if task.family == "matmul_lhs_t":
        k, m, n = task.dimensions
        workload = _lhs_t_workload(k, m, n)
    elif task.family == "matmul_lhs":
        k, m, n = task.dimensions
        workload = _lhs_workload(k, m, n)
    elif task.family == "attention":
        query_length, sequence_length = task.dimensions
        workload = _attention_workload(query_length, sequence_length)
    elif task.family == "rmsnorm_matmul":
        k, m, n = task.dimensions
        workload = _rmsnorm_matmul_workload(k, m, n)
    else:
        raise ValueError(f"unknown rollout workload family {task.family!r}")
    return workload


def _generate_rollout_cases(task: RolloutTask) -> list[FP32SimulationCase]:
    """Generate and render every state for one process-isolated rollout."""
    workload = _workload_for_task(task)
    definition = workload.definition
    inputs = definition.generate_inputs(task.seed)
    expected = np.asarray(definition.f_numpy(**inputs))
    generated_name = f"nki_{definition.f_nkigym.__name__}"
    worker_count = min(ANALYZER_WORKERS, len(TRANSFORMS))
    with ProcessPoolExecutor(max_workers=worker_count) as analyzer:
        analyzer.submit(_analyzer_ready).result()
        with ThreadPoolExecutor(max_workers=1) as renderer:
            pending = [
                (label, renderer.submit(render, state))
                for label, state in _parallel_rollout(workload, task.seed, analyzer)
            ]
            cases = [
                FP32SimulationCase(
                    label=label, kernel=source.result(), func_name=generated_name, inputs=inputs, expected=expected
                )
                for label, source in pending
            ]
    return cases


def test_one_random_rollout_per_randomized_shape_preserves_every_generated_kernel() -> None:
    """One 500-step rollout per newly sampled shape preserves every generated kernel."""
    run_seed = _run_seed()
    rng = random.Random(run_seed)
    print(f"random_rollout_seed={run_seed}", flush=True)
    workloads = _random_workloads(rng)
    tasks: list[RolloutTask] = []
    for workload in workloads:
        seed = rng.randrange(1 << 63)
        print(f"{workload.name} seed {seed}", flush=True)
        tasks.append(RolloutTask(family=workload.family, dimensions=workload.dimensions, seed=seed))
    worker_count = min(len(tasks), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        case_groups = list(executor.map(_generate_rollout_cases, tasks, chunksize=1))
    cases_by_tolerance: dict[tuple[float, float], list[FP32SimulationCase]] = {}
    for task, workload_cases in zip(tasks, case_groups, strict=True):
        assert len(workload_cases) == ROLLOUT_STEPS + 1
        definition = _workload_for_task(task).definition
        cases_by_tolerance.setdefault((definition.atol, definition.rtol), []).extend(workload_cases)
    hosts = _simulation_hosts()
    print(f"simulation_hosts={','.join(hosts)}", flush=True)
    completed = 0
    for (atol, rtol), cases in cases_by_tolerance.items():
        print(f"validating {len(cases)} cases with atol={atol:g}, rtol={rtol:g}", flush=True)
        completed += batch_simulate_fp32(
            hosts=hosts,
            cases=cases,
            atol=atol,
            rtol=rtol,
            timeout_s=BATCH_SIMULATION_TIMEOUT_SECONDS,
            workers_per_host=SIMULATION_WORKERS_PER_HOST,
        )
    assert completed == len(workloads) * (ROLLOUT_STEPS + 1)
