# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List
import multiprocessing, warnings, dill, pickle

multiprocessing.reduction.ForkingPickler.dumps = dill.dumps
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from time import perf_counter
from pprint import pformat
from neuronxcc.starfish.penguin.targets.nki.NumpyKernel import BenchmarkKernel
from neuronxcc.starfish.penguin.targets.nki.TraceKernel import decltensor

from src.visualize import plot_tuning_results


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
        self.perf_results = []
        self.benchmark_machines = (
            benchmark_machines if benchmark_machines is not None else ["localhost"]
        )  # a list of instance used for remote benchmark, if it is none, use the current machine itself

    def create_parameter(self, ctx, shape, dtype, name="", annotation=None):
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
                self.perf_results.append(
                    {
                        "configs": config,
                        "latency": latency_us,
                        "time_elapsed": elapsed,
                    }
                )

        self.grid = ()
        self._post_tuning()
        return None

    def _post_tuning(self):
        assert self.perf_results, "No configs tested"
        best_result = min(
            self.perf_results,
            key=lambda element: element["latency"],
        )
        min_latency = best_result["latency"]
        min_config = best_result["configs"]

        # dump the performance logs
        with open(f"./perf_results.log", "w") as f:
            f.write(pformat(self.perf_results))
            f.write(
                f"\nThe best latency is {min_latency} us for the config {min_config}"
            )
        pickle.dump(self.perf_results, open(f"./perf_results.pkl", "wb"))
        plot_tuning_results(self.perf_results)


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
        kernel_return=True,
    )
    benchmark(*args, **configs, **kwargs)
    return benchmark.benchmark_result.nc_latency.get_latency_percentile(99)
