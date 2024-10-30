# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple, Any
import multiprocessing, warnings, pickle
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from time import perf_counter

import neuronxcc.nki as nki
from neuronxcc.starfish.penguin.targets.nki.TraceKernel import (
    BenchmarkKernel,
    decltensor,
)


class AutotuneKernel(BenchmarkKernel):
    """
    Compile and benchmark NKI kernel on NeuronDevice.
    """

    def __init__(
        self,
        configs: List[dict] = None,
        warmup=2,
        iters=5,
        max_workers=None,
        benchmark_machines=None,
        **kwargs,
    ):
        super().__init__(warmup=warmup, iters=iters, **kwargs)
        self.design_space = configs
        self.max_workers = max_workers
        self.results = []
        self.benchmark_machines = (
            benchmark_machines if benchmark_machines is not None else ["localhost"]
        )  # a list of instance used for remote benchmark, if it is none, use the current machine itself

    def create_tensor_ptr(self, ctx, shape, dtype, name="", annotation=None):
        return decltensor(shape, dtype)

    def __call__(self, *args, **kwargs):
        device_locks = {
            machine: multiprocessing.Manager().Lock()
            for machine in self.benchmark_machines
        }

        args = [
            self._translate_param(ctx=None, o=a, name="", annotation=None) for a in args
        ]
        kwargs = {
            k: self._translate_param(ctx=None, o=v, name=k, annotation=None)
            for k, v in kwargs.items()
        }
        self.results = {
            "timestamps": [],
            "configs": [],
            "latencies": [],
        }

        with ProcessPoolExecutor(max_workers=self.max_workers) as e:
            all_instances = []
            # TODO Evenly assign the benchmarking jobs to instances, can improve to dynamic allocation in the future
            for config, machine in zip(
                self.design_space,
                self.benchmark_machines
                * (len(self.design_space) // len(self.benchmark_machines) + 1),
            ):
                future = e.submit(
                    test_design,
                    func=self.func,
                    grid=self.grid,
                    args=args,
                    kwargs=kwargs,
                    configs=config,
                    device_lock=device_locks[machine],
                    warmup=self.warmup,
                    iters=self.iters,
                    device_count=self.device_count,
                    benchmark_machine=machine,
                )
                all_instances.append((future, config))

            total_configs = len(all_instances)
            start = perf_counter()
            for i in tqdm(range(total_configs), desc="Benchmarking configurations"):
                future, config = all_instances[i]
                try:
                    latency_us = future.result()
                except Exception as e:
                    latency_us = float("inf")
                    warnings.warn(
                        f"Warning: failed for config {config}, reason: {e} ({type(e)})",
                        category=RuntimeWarning,
                        stacklevel=2,
                    )
                elapsed = perf_counter() - start
                self.results["timestamps"].append(elapsed)
                self.results["configs"].append(config)
                self.results["latencies"].append(latency_us)

        assert self.results["latencies"], "No configs tested"
        best_config, best_latency_us = min(
            zip(self.results["configs"], self.results["latencies"]), key=lambda x: x[1]
        )
        print(f"The best latency is {best_latency_us} us for config {best_config}")

        self.grid = ()
        pickle.dump(self.results, open("benchmark_results.pkl", "wb"))
        return None  # the kernel output means nothing here


def test_design(
    func,
    grid,
    args,
    kwargs,
    configs,
    device_lock,
    warmup,
    iters,
    device_count,
    benchmark_machine=None,
):
    benchmark = BenchmarkKernel(
        func=func,
        grid=grid,
        warmup=warmup,
        iters=iters,
        device_count=device_count,
        device_lock=device_lock,
        benchmark_machine=benchmark_machine,
    )
    benchmark(*args, **configs, **kwargs)
    return benchmark.benchmark_result.nc_latency.get_latency_percentile(99)
