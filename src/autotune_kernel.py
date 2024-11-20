# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple, Any, Union
import multiprocessing, warnings, pickle, subprocess
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from time import perf_counter

import neuronxcc.nki as nki
from neuronxcc.starfish.penguin.targets.nki.TraceKernel import (
    BaremetalKernel,
    BenchmarkKernel,
)


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
        self.perf_dict = {}
        self.design_space = self._generate_design_space(configs)

    def __call__(self, *args, **kwargs):
        total_configs = len(self.design_space)
        for i in tqdm(range(total_configs)):
            configs = self.design_space[i]
            perf = self.test_design(configs, args, kwargs)
            self.perf_dict[str(configs)] = perf
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
            perf = (
                benchmark.benchmark_result.nc_latency.get_latency_percentile(99)
                / 1000.0
            )
        except AssertionError as e:
            print(f"Warning: failed for config {configs}, reason: {e}")
            perf = float("inf")
        except subprocess.CalledProcessError as e:
            print(
                f"Warning: Will fail if not running on a neuron device, failed for config {configs}, reason: {e}"
            )
            perf = float("inf")
        return perf

    def _generate_design_space(self, configs) -> List:
        if isinstance(configs, dict):
            # user use numerical method to define the design space
            param_specs = configs

            def generate_values(spec):
                method, *args = spec
                if method == "linear":
                    start, end, step = args
                    return range(start, end + 1, step)
                elif method == "power_of_2":
                    start, end = args
                    return (2**i for i in range(start, end + 1))
                elif method == "categorical":
                    return args[0]
                else:
                    raise ValueError(f"Unknown method: {method}")

            def generate_configs(params):
                if not params:
                    result_configs.append(current.copy())
                    return
                key, spec = params[0]
                for value in generate_values(spec):
                    current[key] = value
                    generate_configs(params[1:])

            result_configs = []
            current = {}
            generate_configs(list(param_specs.items()))
            return result_configs
        elif isinstance(configs, List):  # user provide a list of designs
            return configs
        else:
            raise NotImplementedError(f"Unsupported configs type {type(configs)}")

    def post_tuning(self):
        assert len(self.perf_dict) > 0, "No configs tested"
        min_key = min(
            self.perf_dict, key=lambda configs_str: self.perf_dict[configs_str]
        )
        min_value = self.perf_dict[min_key]
        print(f"The best perf is {min_value} for config {min_key}")

        # dump the performance dict
        from pprint import pformat

        with open(f"./perf_dict.txt", "w") as f:
            f.write(pformat(self.perf_dict))
