# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Callable, Tuple, Dict
import multiprocessing, warnings, dill, pickle, math, os, shutil, subprocess
from itertools import product

multiprocessing.reduction.ForkingPickler.dumps = dill.dumps
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from time import perf_counter
from pprint import pformat

from src.benchmark import profile_kernel
from src.visualize import plot_tuning_results
from src.cache.directories import TUNED_NKI_CACHE_DIR, split_file_info, convert_tensor_shapes

from neuronxcc.nki.compile import GenericKernel


class Autotune:
    """
    Compile and benchmark NKI kernel on NeuronDevice.
    """

    def __init__(
        self,
        kernel: GenericKernel,
        kernel_args: Tuple,
        configs: List[Dict],
        max_configs: int | None = None,
        warmup: int = 10,
        iters: int = 100,
        pruning_func: Callable | None = None,
        benchmark_machines=None,
        cache_dir: str | None = None,
        trace: bool = False,
    ):
        self.kernel = kernel
        self.kernel_args = kernel_args
        self.configs = configs
        self.max_configs = max_configs
        self.warmup = warmup
        self.iters = iters
        self.pruning_func = pruning_func
        self.benchmark_machines = benchmark_machines if benchmark_machines is not None else ["localhost"]
        self.perf_results = []
        self.cache_dir = self._get_cache_dir(cache_dir)
        self.trace = trace

    def _get_cache_dir(self, cache_dir: str | None = None) -> str:
        if not cache_dir:
            cache_dir = f"{TUNED_NKI_CACHE_DIR}/{self.kernel.func_name}"
        shape_dir = convert_tensor_shapes([str(arg.tensor_shape) for arg in self.kernel_args])
        cache_dir = f"{cache_dir}/{shape_dir}"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
        return cache_dir

    def _prune(self) -> List[dict]:
        """
        Pruning func should throw a fail if the inputs are illegal
        """
        valid_configs = []
        arg_shapes = [arg.tensor_shape for arg in self.kernel_args]
        for config in self.configs:
            try:
                if self.pruning_func is not None:
                    self.pruning_func(*arg_shapes, **config)
                valid_configs.append(config)
            except Exception as e:
                print(f"Prune invalid config {config}, reason: {e} ({type(e)})")
            if self.max_configs and len(valid_configs) == self.max_configs:
                break
        assert valid_configs, f"No valid autotune configs found"
        return valid_configs

    def __call__(self):
        start = perf_counter()
        valid_configs = self._prune()
        device_locks = {machine: multiprocessing.Manager().Lock() for machine in self.benchmark_machines}

        benchmark_machines = self.benchmark_machines * math.ceil(len(valid_configs) // len(self.benchmark_machines))

        max_workers = min(len(valid_configs), os.cpu_count() - 1)
        futures_to_config = {}
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for config, machine in zip(valid_configs, benchmark_machines):
                future = pool.submit(
                    profile_kernel,
                    func=self.kernel,
                    args=self.kernel_args,
                    cache_dir=self.cache_dir,
                    configs=config,
                    warmup=self.warmup,
                    iters=self.iters,
                    device_lock=device_locks[machine],
                    benchmark_machine=machine,
                )
                futures_to_config[future] = config

        neff_files = []
        for future in tqdm(
            as_completed(futures_to_config), total=len(futures_to_config), desc="Benchmarking configurations"
        ):
            config = futures_to_config[future]
            try:
                latency_us, neff_file = future.result()
                neff_files.append(neff_file)
            except Exception as e:
                latency_us = float("inf")
                warnings.warn(
                    f"Warning: failed for config {config}, reason: {e} ({type(e)})",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
            elapsed = perf_counter() - start
            self.perf_results.append({"configs": config, "latency": latency_us, "time_elapsed": elapsed})
            self._post_tuning()
        if self.trace:
            self._trace_neffs(neff_files)
        return None

    def _post_tuning(self):
        assert self.perf_results, "No configs tested"
        best_result = min(self.perf_results, key=lambda element: element["latency"])
        min_latency = best_result["latency"]
        min_config = best_result["configs"]

        # Dump the performance logs
        with open(f"{self.cache_dir}/tune.log", "w") as f:
            f.write(pformat(self.perf_results))
            f.write(f"\nAutotune {self.kernel.func_name} for inputs {[arg.tensor_shape for arg in self.kernel_args]}")
            f.write(f"\nThe best latency is {min_latency} ms for the config {min_config}")
        pickle.dump(self.perf_results, open(f"{self.cache_dir}/tune.pkl", "wb"))
        plot_tuning_results(self.perf_results, self.cache_dir)

    def _trace_neffs(self, neff_files: List[str]):
        for neff_file in neff_files:
            directory, neff_name, file_type = split_file_info(neff_file)
            assert file_type == "neff", f"{neff_file} is not a .neff file."
            ntff_file = f"{directory}/{neff_name}.ntff"
            trace_cmd = f"neuron-profile capture -n {neff_file} --profile-nth-exec={self.iters}"
            subprocess.run(trace_cmd, shell=True)
            shutil.move(f"profile_exec_{self.iters}.ntff", ntff_file)
            upload_command = (
                f'profile-upload -F "neff=@{neff_name}.neff" -F "ntff=@{neff_name}.ntff" -F name={neff_name}'
            )
            with open(f"{self.cache_dir}/upload_profile.log", "a") as f:
                f.write(f"{upload_command}\n")
