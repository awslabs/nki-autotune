# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union
import pickle, subprocess
from tqdm import tqdm
from time import perf_counter
from neuronxcc.starfish.penguin.targets.nki.TraceKernel import (
    BaremetalKernel,
    BenchmarkKernel,
)

from src.visualize import plot_tuning_results


class AutotuneKernel(BaremetalKernel):
    """
    Compile and benchmark NKI kernel on NeuronDevice.
    """

    def __init__(
        self,
        warmup=5,
        iters=10,
        device_count=1,
        configs: Union[dict, List[dict]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.warmup = warmup
        self.iters = iters
        self.device_count = device_count
        self.perf_results = []
        self.design_space = configs

    def __call__(self, *args, **kwargs):
        total_configs = len(self.design_space)
        start = perf_counter()
        for i in tqdm(range(total_configs)):
            configs = self.design_space[i]
            latency = self.test_design(configs, args, kwargs)
            time_elapsed = perf_counter() - start
            self.perf_results.append(
                {
                    "configs": configs,
                    "latency": latency,
                    "time_elapsed": time_elapsed,
                }
            )
            print(f"Progress: {i + 1}/{total_configs} configurations tested")
        self.post_tuning()
        return None  # the kernel output means nothing here

    def test_design(self, configs, args, kwargs):
        benchmark = BenchmarkKernel(
            func=self.func,
            warmup=self.warmup,
            iters=self.iters,
            device_count=self.device_count,
        )
        combined_kwargs = {**configs, **kwargs}
        try:
            benchmark(*args, **combined_kwargs)
            latency = (
                benchmark.benchmark_result.nc_latency.get_latency_percentile(99)
                / 1000.0
            )
        except AssertionError as e:
            print(f"Warning: failed for config {configs}, reason: {e}")
            latency = float("inf")
        except subprocess.CalledProcessError as e:
            print(
                f"Warning: Will fail if not running on a neuron device, failed for config {configs}, reason: {e}"
            )
            latency = float("inf")
        return latency

    def post_tuning(self):
        assert self.perf_results, "No configs tested"
        best_result = min(
            self.perf_results,
            key=lambda element: element["latency"],
        )
        min_latency = best_result["latency"]
        min_config = best_result["configs"]
        print(f"The best latency is {min_latency} for config {min_config}")

        # dump the performance logs
        from pprint import pformat

        with open(f"./perf_results.txt", "w") as f:
            f.write(pformat(self.perf_results))
        pickle.dump(self.perf_results, open(f"./perf_results.pkl", "wb"))
        plot_tuning_results(self.perf_results)
