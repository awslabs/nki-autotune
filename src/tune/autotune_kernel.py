# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Tuple

import numpy as np
from neuronpy.runtime.spike import SpikeExecutor
from neuronxcc.nki.compile import GenericKernel
from tqdm import tqdm

import src.tune.utils as utils
from src.cache.directories import TUNED_CACHE_DIR, get_cache_dir, split_file_info
from src.cache.results import PerformanceMetrics
from src.tune.utils import create_and_compile_nki_kernel, create_spike_kernel


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
        self.results = PerformanceMetrics(sort_key="min_ms")
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

        """
        Parallel NKI compilation
        TODO: support different NKI kernels
        """
        jobs = {}
        for job_id, config in enumerate(valid_configs):
            jobs[job_id] = {"config": config}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for job_id in jobs:
                utils.set_kernel(self.kernel)
                future = executor.submit(
                    create_and_compile_nki_kernel, self.kernel_args, jobs[job_id]["config"], self.cache_dir
                )
                jobs[job_id]["future"] = future
        for job_id in tqdm(jobs, total=len(jobs), desc="Compiling kernels"):
            future = jobs[job_id]["future"]
            try:
                neff = future.result()
                jobs[job_id]["neff"] = neff
            except Exception as e:
                # TODO: pass error to final results
                print(f"Error in NKI compilation: {str(e)} {type(e)}")

        for job_id in jobs:
            jobs[job_id]["spike_kernel"] = create_spike_kernel(
                jobs[job_id]["neff"], self.kernel, self.kernel_args, jobs[job_id]["config"]
            )

        with SpikeExecutor(verbose=0) as spike:
            for job_id in jobs:
                # FIXME: args are used, kwargs are needed to run but not used
                # TODO: pass error to final results
                stats = spike.benchmark(
                    jobs[job_id]["spike_kernel"],
                    *self.kernel_args,
                    **jobs[job_id]["config"],
                    warmup_iterations=self.warmup,
                    benchmark_iterations=self.iters,
                    device_id=0,
                )
                self.results.add_result(config=jobs[job_id]["config"], **stats)

        self.results.save(cache_dir=self.cache_dir)
        if self.trace:
            for job_id in tqdm(jobs, total=len(jobs), desc="Tracing NEFFs"):
                self._trace_neff(jobs[job_id]["neff"])
        return None

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

    def _trace_neff(self, neff_file: str):
        """
        Generate trace profiles for compiled kernel files.

        This method processes each NEFF (Neuron Executable File Format) file by:
        1. Capturing a trace profile using neuron-profile
        2. Moving the resulting trace file (NTFF) to the appropriate location
        3. Creating an upload command for the profile data and logging it

        Args:
            neff_files (str): NEFF file to be traced.

        Raises:
            AssertionError: If any of the provided files is not a .neff file.

        Note:
            This method is used when the 'trace' flag is set to True, allowing
            for detailed performance analysis of the compiled kernels.
        """
        directory, neff_name, file_type = split_file_info(neff_file)
        assert file_type == "neff", f"{neff_file} is not a .neff file."
        ntff_file = f"{directory}/{neff_name}.ntff"
        trace_cmd = f"neuron-profile capture -n {neff_file} --profile-nth-exec={self.iters}"
        subprocess.run(trace_cmd, shell=True)
        shutil.move(f"profile_exec_{self.iters}.ntff", ntff_file)
        upload_command = f'profile-upload -F "neff=@{neff_name}.neff" -F "ntff=@{neff_name}.ntff" -F name={neff_name}'
        with open(f"{self.cache_dir}/upload_profile.log", "a") as f:
            f.write(f"{upload_command}\n")
