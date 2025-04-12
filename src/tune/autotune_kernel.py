# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import os
import shutil
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Tuple

import numpy as np
from neuronxcc.nki.compile import GenericKernel
from tqdm import tqdm

import src.tune.utils as utils
from src.cache.directories import TUNED_CACHE_DIR, get_cache_dir, split_file_info
from src.cache.results import PerformanceMetrics, PerformanceResult
from src.tune.utils import create_and_compile_nki_kernel


class Autotune:
    """
    Compile and benchmark NKI kernel on NeuronDevice.
    """

    def __init__(
        self,
        kernel: GenericKernel,
        kernel_args: Tuple[np.ndarray, ...],
        configs: List[Dict],
        max_configs: int | None = None,
        warmup: int = 10,
        iters: int = 100,
        pruning_func: Callable | None = None,
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
        self.perf_results = PerformanceMetrics()
        if not cache_dir:
            self.cache_dir = get_cache_dir(cache_root_dir=TUNED_CACHE_DIR, kernel=kernel, kernel_args=kernel_args)
        else:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
            self.cache_dir = cache_dir
        self.trace = trace

    def __call__(self):
        os.environ["NEURON_CC_FLAGS"] = "--framework=XLA --target=trn1 --auto-cast=none"
        valid_configs = self._prune()
        num_workers = min(len(valid_configs), os.cpu_count() - 1)

        futures_to_config = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for config in valid_configs:
                utils.set_kernel(self.kernel)
                future = executor.submit(create_and_compile_nki_kernel, self.kernel_args, config, self.cache_dir)
                futures_to_config[future] = config

        for future in tqdm(futures_to_config, total=len(futures_to_config), desc="Compiling kernels"):
            try:
                neff = future.result()
            except Exception as e:
                print(f"Error in worker process: {str(e)} {type(e)}")

        # device_locks = {machine: multiprocessing.Manager().Lock() for machine in self.benchmark_machines}

        # futures_to_config = {}
        # with ProcessPoolExecutor(max_workers=num_workers) as pool:
        #     for config, machine in zip(valid_configs, benchmark_machines):
        #         future = pool.submit(
        #             profile_kernel,
        #             func=self.kernel,
        #             args=self.kernel_args,
        #             cache_dir=self.cache_dir,
        #             configs=config,
        #             warmup=self.warmup,
        #             iters=self.iters,
        #             device_lock=device_locks[machine],
        #             benchmark_machine=machine,
        #         )
        #         futures_to_config[future] = config

        # neff_files = []
        # for future in futures_to_config:
        #     config = futures_to_config[future]
        #     try:
        #         result, neff_file = future.result()
        #         neff_files.append(neff_file)
        #     except Exception as e:
        #         result = PerformanceResult(config, float("inf"), error=e)
        #         warnings.warn(
        #             f"Warning: failed for config {config}, reason: {e} ({type(e)})",
        #             category=RuntimeWarning,
        #             stacklevel=2,
        #         )
        #     self.perf_results.append(result)
        #     self.perf_results.save(cache_dir=self.cache_dir)
        # if self.trace:
        #     self._trace_neffs(neff_files)
        # return None

    def _prune(self) -> List[dict]:
        """
        Pruning func should throw a fail if the inputs are illegal
        """
        valid_configs = []
        invalid_configs = []
        arg_shapes = [arg.shape for arg in self.kernel_args]
        for config in self.configs:
            try:
                if self.pruning_func is not None:
                    self.pruning_func(*arg_shapes, **config)
                valid_configs.append(config)
            except Exception as e:
                invalid_configs.append((config, e))
            if self.max_configs and len(valid_configs) == self.max_configs:
                break
        assert valid_configs, f"No valid autotune configs found"
        return valid_configs

    def _trace_neffs(self, neff_files: List[str]):
        """
        Generate trace profiles for compiled kernel files.

        This method processes each NEFF (Neuron Executable File Format) file by:
        1. Capturing a trace profile using neuron-profile
        2. Moving the resulting trace file (NTFF) to the appropriate location
        3. Creating an upload command for the profile data and logging it

        Args:
            neff_files (List[str]): List of paths to NEFF files to be traced.

        Raises:
            AssertionError: If any of the provided files is not a .neff file.

        Note:
            This method is used when the 'trace' flag is set to True, allowing
            for detailed performance analysis of the compiled kernels.
        """
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
