# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List
import multiprocessing, warnings, dill, pickle, math

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
        func,
        configs: List[dict],
        warmup: int = 2,
        iters: int = 5,
        max_workers: int | None = None,
        benchmark_machines=None,
        **kwargs,
    ):
        self.func = func
        self.configs = configs
        self.warmup = warmup
        self.iters = iters
        self.max_workers = max_workers
        self.benchmark_machines = (
            benchmark_machines if benchmark_machines is not None else ["localhost"]
        )  # a list of instance used for remote benchmark, if it is none, use the current machine itself
        self.perf_results = []

    def __call__(self, *args, **kwargs):
        device_locks = {
            machine: multiprocessing.Manager().Lock()
            for machine in self.benchmark_machines
        }

        benchmark_machines = self.benchmark_machines * math.ceil(
            len(self.configs) // len(self.benchmark_machines)
        )

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            all_instances = []
            for config, machine in zip(
                self.configs,
                benchmark_machines,
            ):
                future = executor.submit(
                    test_design,
                    func=self.func,
                    args=args,
                    kwargs=kwargs,
                    configs=config,
                    device_lock=device_locks[machine],
                    warmup=self.warmup,
                    iters=self.iters,
                    benchmark_machine=machine,
                )
                all_instances.append((future, config))

            start = perf_counter()
            for i in tqdm(
                range(len(all_instances)), desc="Benchmarking configurations"
            ):
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

        # dump the performance logs
        with open(f"./perf_results.log", "w") as f:
            f.write(pformat(self.perf_results))
            f.write(f"\nAutotune for inputs {[arg.tensor_shape for arg in args]}")
            f.write(
                f"\nThe best latency is {min_latency} us for the config {min_config}"
            )
        pickle.dump(self.perf_results, open(f"./perf_results.pkl", "wb"))
        plot_tuning_results(self.perf_results)
