"""Random-rollout correctness coverage for every discovered workload."""

from __future__ import annotations

import multiprocessing
import os
import random
import time
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from math import ceil
from types import TracebackType
from typing import Any, cast

import pytest
from config import SIMULATION_HOSTS

from kernel_library import WORKLOADS
from nkigym.codegen import render
from nkigym.ir import KernelIR, build_initial_ir
from nkigym.profile import FP32SimulationCase, batch_simulate_fp32
from nkigym.synthesis import SynthesizedKernel, synthesize_numpy_to_nkigym
from nkigym.transforms import Transform, TransformOption, public_transforms

ROLLOUT_STEPS = 500
TRANSFORMS_PER_SIMULATION = 10
BATCH_SIMULATION_TIMEOUT_SECONDS = 7200
TRANSFORMS = public_transforms()

_AnalysisResult = tuple[int, tuple[TransformOption, ...], float]
_IndexedTransform = tuple[int, Transform[Any]]


def _analyzer_ready() -> None:
    """Provide a picklable task used to start analyzer workers eagerly."""


def _analyze_transform_group(state: KernelIR, transforms: tuple[_IndexedTransform, ...]) -> tuple[_AnalysisResult, ...]:
    """Analyze one group and return options plus per-transform elapsed time."""
    results: list[_AnalysisResult] = []
    for index, transform in transforms:
        started = time.perf_counter()
        options = cast(tuple[TransformOption, ...], tuple(transform.analyze(state)))
        results.append((index, options, time.perf_counter() - started))
    return tuple(results)


def _analysis_groups(weights: list[float], worker_limit: int) -> tuple[tuple[int, ...], ...]:
    """Balance transforms using enough groups to approach the longest-task bound."""
    groups: tuple[tuple[int, ...], ...] = ()
    if weights:
        useful_workers = min(worker_limit, len(weights), ceil(sum(weights) / max(weights)) + 1)
        mutable_groups: list[list[int]] = [[] for _worker in range(useful_workers)]
        loads = [0.0 for _worker in range(useful_workers)]
        for index in sorted(range(len(weights)), key=lambda item: (-weights[item], item)):
            worker = min(range(useful_workers), key=lambda item: (loads[item], item))
            mutable_groups[worker].append(index)
            loads[worker] += weights[index]
        groups = tuple(tuple(sorted(group)) for group in mutable_groups)
    return groups


class _ParallelLegalityAnalyzer:
    """Keep one local process pool across states and adapt transform grouping."""

    def __init__(self, transforms: list[Transform[Any]]) -> None:
        """Detect usable local CPUs and initialize transform cost estimates."""
        self.transforms = transforms
        cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
        self.max_workers = min(cpu_count, len(transforms))
        self._weights = [1.0 for _transform in transforms]
        self._executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> _ParallelLegalityAnalyzer:
        """Start the persistent analyzer process pool."""
        self._executor = ProcessPoolExecutor(
            max_workers=self.max_workers, mp_context=multiprocessing.get_context("fork")
        )
        ready = [self._executor.submit(_analyzer_ready) for _worker in range(self.max_workers)]
        for future in ready:
            future.result()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop every analyzer process."""
        if self._executor is None:
            raise RuntimeError("parallel legality analyzer was not started")
        self._executor.shutdown()
        self._executor = None

    def legal_actions(self, state: KernelIR) -> list[tuple[Transform[Any], TransformOption]]:
        """Return ordered legal actions after adaptively grouped analysis."""
        if self._executor is None:
            raise RuntimeError("parallel legality analyzer must be used as a context manager")
        groups = _analysis_groups(self._weights, self.max_workers)
        futures = [
            self._executor.submit(
                _analyze_transform_group, state, tuple((index, self.transforms[index]) for index in group)
            )
            for group in groups
        ]
        options_by_index: dict[int, tuple[TransformOption, ...]] = {}
        for future in futures:
            for index, options, elapsed_s in future.result():
                options_by_index[index] = options
                self._weights[index] = elapsed_s
        actions = [
            (transform, option) for index, transform in enumerate(self.transforms) for option in options_by_index[index]
        ]
        return actions


def _rollout(
    name: str, kernel: SynthesizedKernel, seed: int, analyzer: _ParallelLegalityAnalyzer
) -> Iterator[tuple[str, KernelIR]]:
    """Yield one seeded rollout with parallel transform legality analysis."""
    rng = random.Random(seed)
    prefix = f"{name} seed {seed}"
    state = build_initial_ir(kernel.function, kernel.input_specs)
    yield (f"{prefix} step 0", state)
    for step in range(1, ROLLOUT_STEPS + 1):
        actions = analyzer.legal_actions(state)
        if not actions:
            raise AssertionError(f"{prefix} terminated after {step - 1} steps")
        action = rng.choice(actions)
        state = action[0].apply(state, action[1])
        label = f"{prefix} step {step}: {type(action[0]).__name__} {action[1]!r}"
        yield (label, state)


def _generate_rollout_cases(name: str, seed: int, analyzer: _ParallelLegalityAnalyzer) -> list[FP32SimulationCase]:
    """Render every tenth state in one complete rollout."""
    workload = WORKLOADS[name]
    kernel = synthesize_numpy_to_nkigym(workload["numpy_ref"], workload["input_specs"])
    reference_inputs = workload["input_generator"](workload["input_specs"], seed)
    inputs = kernel.adapt_inputs({name: value.copy() for name, value in reference_inputs.items()})
    expected = kernel.adapt_output(
        workload["numpy_ref"](**{name: value.copy() for name, value in reference_inputs.items()})
    )
    generated_name = f"nki_{kernel.function.__name__}"
    cases = [
        FP32SimulationCase(
            label=label, kernel=render(state), func_name=generated_name, inputs=inputs, expected=expected
        )
        for step, (label, state) in enumerate(_rollout(name, kernel, seed, analyzer))
        if step > 0 and step % TRANSFORMS_PER_SIMULATION == 0
    ]
    return cases


@pytest.fixture(scope="module")
def rollout_results() -> dict[str, int | Exception]:
    """Generate workloads serially while one remote simulation batch runs ahead."""
    results: dict[str, int | Exception] = {}
    futures: dict[str, Future[int]] = {}
    with _ParallelLegalityAnalyzer(TRANSFORMS) as analyzer, ThreadPoolExecutor(max_workers=1) as executor:
        print(f"legality_analyzer_workers={analyzer.max_workers}", flush=True)
        for name, workload in WORKLOADS.items():
            seed = random.SystemRandom().randrange(1 << 63)
            print(f"{name} seed {seed}", flush=True)
            try:
                cases = _generate_rollout_cases(name, seed, analyzer)
                expected_case_count = ROLLOUT_STEPS // TRANSFORMS_PER_SIMULATION
                if len(cases) != expected_case_count:
                    raise AssertionError(f"{name}: generated {len(cases)} cases, expected {expected_case_count}")
                hosts: list[str] = list(SIMULATION_HOSTS)
                print(f"simulation_hosts={','.join(hosts)}", flush=True)
                print(
                    f"validating {len(cases)} cases with atol={workload['atol']:g}, rtol={workload['rtol']:g}",
                    flush=True,
                )
                futures[name] = executor.submit(
                    batch_simulate_fp32,
                    hosts=hosts,
                    cases=cases,
                    atol=workload["atol"],
                    rtol=workload["rtol"],
                    timeout_s=BATCH_SIMULATION_TIMEOUT_SECONDS,
                )
            except Exception as error:
                results[name] = error
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except Exception as error:
                results[name] = error
    return results


@pytest.mark.parametrize("workload_name", [pytest.param(name, id=name) for name in WORKLOADS])
def test_one_random_rollout_per_workload_preserves_every_tenth_kernel(
    workload_name: str, rollout_results: dict[str, int | Exception]
) -> None:
    """Apply 500 random transforms and CPU-simulate every tenth state."""
    expected_case_count = ROLLOUT_STEPS // TRANSFORMS_PER_SIMULATION
    result = rollout_results[workload_name]
    if isinstance(result, Exception):
        raise result
    assert result == expected_case_count
