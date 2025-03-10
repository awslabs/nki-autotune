# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Callable, Tuple, Dict
import multiprocessing, warnings, dill, pickle, math, os, shutil
from itertools import product

multiprocessing.reduction.ForkingPickler.dumps = dill.dumps
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from time import perf_counter
from pprint import pformat

from src.benchmark import profile_kernel
from src.visualize import plot_tuning_results

from neuronxcc.nki.compile import GenericKernel


class Autotune:
    """
    Compile and benchmark NKI kernel on NeuronDevice.
    """

    def __init__(
        self,
        kernel: GenericKernel,
        configs: List[dict],
        max_configs: int | None = None,
        warmup: int = 2,
        iters: int = 5,
        pruning_func: Callable | None = None,
        benchmark_machines=None,
        cache_dir="./autotune_cache",
    ):
        self.kernel = kernel
        self.configs = configs
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

    def _prune(self, args: Tuple, kwargs: Dict) -> List[dict]:
        # Pruning func should throw a fail if the inputs are illegal
        valid_configs = []
        for config in self.configs:
            arg_shapes = [arg.tensor_shape for arg in args]
            try:
                if self.pruning_func is not None:
                    self.pruning_func(*arg_shapes, **config, **kwargs)
                valid_configs.append(config)
            except Exception as e:
                warnings.warn(
                    f"Warning: invalid config {config}, reason: {e} ({type(e)})", category=RuntimeWarning, stacklevel=2
                )
            if self.max_configs and len(valid_configs) == self.max_configs:
                break
        return valid_configs

    def __call__(self, *args, **kwargs):
        start = perf_counter()
        valid_configs = self._prune(args, kwargs)
        device_locks = {machine: multiprocessing.Manager().Lock() for machine in self.benchmark_machines}

        benchmark_machines = self.benchmark_machines * math.ceil(len(valid_configs) // len(self.benchmark_machines))

        max_workers = min(len(valid_configs), os.cpu_count() - 1)
        futures_to_config = {}
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for config, machine in zip(valid_configs, benchmark_machines):
                future = pool.submit(
                    profile_kernel,
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
                futures_to_config[future] = config

        for future in tqdm(
            as_completed(futures_to_config), total=len(futures_to_config), desc="Benchmarking configurations"
        ):
            config = futures_to_config[future]
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
