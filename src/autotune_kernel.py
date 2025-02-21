# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List
import multiprocessing, warnings, dill, pickle, math, os, shutil
from itertools import product

multiprocessing.reduction.ForkingPickler.dumps = dill.dumps
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from time import perf_counter
from pprint import pformat

from src.benchmark import test_design
from src.visualize import plot_tuning_results


class Autotune:
    """
    Compile and benchmark NKI kernel on NeuronDevice.
    """

    def __init__(
        self,
        configs: List[dict],
        warmup: int = 2,
        iters: int = 5,
        pruning_func=None,
        benchmark_machines=None,
        cache_dir="./autotune_cache",
        **kwargs,
    ):
        self.configs = configs
        self.warmup = warmup
        self.iters = iters
        self.kwargs = kwargs
        self.pruning_func = pruning_func
        self.benchmark_machines = (
            benchmark_machines if benchmark_machines is not None else ["localhost"]
        )
        self.perf_results = []
        self.cache_dir = cache_dir
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir)

    def __call__(self, *args, **kwargs):
        device_locks = {
            machine: multiprocessing.Manager().Lock()
            for machine in self.benchmark_machines
        }

        benchmark_machines = self.benchmark_machines * math.ceil(
            len(self.configs) // len(self.benchmark_machines)
        )

        start = perf_counter()
        for config, machine in tqdm(
            zip(self.configs, benchmark_machines),
            total=len(self.configs),
            desc="Benchmarking configs",
        ):
            kernel = config.pop("kernel", None)
            try:
                latency_us = test_design(
                    func=kernel,
                    args=args,
                    kwargs=kwargs,
                    configs=config,
                    pruning_func=self.pruning_func,
                    device_lock=device_locks[machine],
                    warmup=self.warmup,
                    iters=self.iters,
                    cache_dir=self.cache_dir,
                    benchmark_machine=machine,
                )
            except Exception as e:
                latency_us = float("inf")
                warnings.warn(
                    f"Warning: failed for config {config}, kernel {kernel.func_name} reason: {e} ({type(e)})",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
            elapsed = perf_counter() - start
            config["kernel_name"] = kernel.func_name
            self.perf_results.append(
                {
                    "configs": config,
                    "latency": latency_us,
                    "time_elapsed": elapsed,
                }
            )

            self._post_tuning(*args)
        return None

    def _post_tuning(self, *args):
        assert self.perf_results, "No configs tested"
        best_result = min(
            self.perf_results,
            key=lambda element: element["latency"],
        )
        min_latency = best_result["latency"]
        min_config = best_result["configs"]

        # Dump the performance logs
        with open(f"{self.cache_dir}/tune.log", "w") as f:
            f.write(pformat(self.perf_results))
            f.write(f"\nAutotune for inputs {[arg.tensor_shape for arg in args]}")
            f.write(
                f"\nThe best latency is {min_latency} us for the config {min_config}"
            )
        pickle.dump(self.perf_results, open(f"{self.cache_dir}/tune.pkl", "wb"))
        plot_tuning_results(self.perf_results, self.cache_dir)
