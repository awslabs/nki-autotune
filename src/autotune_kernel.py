# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Callable, Tuple, Dict
import multiprocessing, warnings, dill, pickle, math, os, shutil
from itertools import product

multiprocessing.reduction.ForkingPickler.dumps = dill.dumps
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from time import perf_counter
from pprint import pformat

from src.benchmark import test_design
from src.visualize import plot_tuning_results

from neuronxcc.nki.compile import GenericKernel


class Autotune:
    """
    Compile and benchmark NKI kernel on NeuronDevice.
    """

    def __init__(
        self,
        kernel: GenericKernel,
        max_configs: int | None = None,
        warmup: int = 2,
        iters: int = 5,
        pruning_func: Callable | None = None,
        benchmark_machines=None,
        cache_dir="./autotune_cache",
    ):
        self.kernel = kernel
        self.max_configs = max_configs
        self.warmup = warmup
        self.iters = iters
        self.pruning_func = pruning_func
        self.benchmark_machines = benchmark_machines if benchmark_machines is not None else ["localhost"]
        self.perf_results = []
        self.cache_dir = cache_dir
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir)

    def _prune(self, configs: List[dict], args: Tuple, kwargs: Dict) -> List[dict]:
        # Pruning func should throw a fail if the inputs are illegal
        valid_configs = []
        for config in configs:
            arg_shapes = [arg.tensor_shape for arg in args]
            try:
                if self.pruning_func is not None:
                    self.pruning_func(*arg_shapes, **config, **kwargs)
                valid_configs.append(config)
            except Exception as e:
                warnings.warn(
                    f"Warning: invalid config {config}, reason: {e} ({type(e)})", category=RuntimeWarning, stacklevel=2
                )
            if len(valid_configs) == self.max_configs:
                break
        return valid_configs

    def __call__(self, configs: List[dict], *args, **kwargs):
        valid_configs = self._prune(configs, args, kwargs)
        device_locks = {machine: multiprocessing.Manager().Lock() for machine in self.benchmark_machines}

        benchmark_machines = self.benchmark_machines * math.ceil(len(valid_configs) // len(self.benchmark_machines))

        start = perf_counter()
        for config, machine in tqdm(
            zip(valid_configs, benchmark_machines), total=len(valid_configs), desc="Benchmarking configs"
        ):
            try:
                latency_us = test_design(
                    func=self.kernel,
                    args=args,
                    kwargs=kwargs,
                    configs=config,
                    device_lock=device_locks[machine],
                    warmup=self.warmup,
                    iters=self.iters,
                    cache_dir=self.cache_dir,
                    benchmark_machine=machine,
                )
            except Exception as e:
                latency_us = float("inf")
                warnings.warn(
                    f"Warning: failed for config {config}, kernel {self.kernel.func_name} reason: {e} ({type(e)})",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
            elapsed = perf_counter() - start
            self.perf_results.append({"configs": config, "latency": latency_us, "time_elapsed": elapsed})

            self._post_tuning(*args)
        return None

    def _post_tuning(self, *args):
        assert self.perf_results, "No configs tested"
        best_result = min(self.perf_results, key=lambda element: element["latency"])
        min_latency = best_result["latency"]
        min_config = best_result["configs"]

        # Dump the performance logs
        with open(f"{self.cache_dir}/tune.log", "w") as f:
            f.write(pformat(self.perf_results))
            f.write(f"\nAutotune for inputs {[arg.tensor_shape for arg in args]}")
            f.write(f"\nThe best latency is {min_latency} us for the config {min_config}")
        pickle.dump(self.perf_results, open(f"{self.cache_dir}/tune.pkl", "wb"))
        plot_tuning_results(self.perf_results, self.cache_dir)
