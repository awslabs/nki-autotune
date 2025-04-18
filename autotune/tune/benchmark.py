# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Dict

from neuronpy.runtime.spike import SpikeExecutor
from tqdm import tqdm

from autotune.cache.directories import split_file_info
from autotune.cache.results import PerformanceMetrics
from autotune.tune.job import ProfileJob, ProfileJobs
from autotune.tune.utils import compile_kernel, create_spike_kernel


class Benchmark:
    """
    Compile and benchmark NKI kernel on NeuronDevice.
    """

    def __init__(self, jobs: ProfileJobs, cache_dir: str, warmup: int = 10, iters: int = 100, trace: bool = False):
        self.jobs = jobs
        self.warmup = warmup
        self.iters = iters
        self.results = PerformanceMetrics(sort_key="min_ms")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
        self.cache_dir = cache_dir
        self.trace = trace

    def __call__(self):
        num_workers = min(len(self.jobs), os.cpu_count() - 1)

        """
        Parallel compilation
        """
        future_to_job: Dict[Future, ProfileJob] = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for job in self.jobs:
                future = executor.submit(
                    compile_kernel, job.kernel.__name__, job.name, job.kernel_args, job.kwargs, self.cache_dir
                )
                future_to_job[future] = job
        for future in tqdm(future_to_job, total=len(future_to_job), desc="Compiling kernels"):
            job = future_to_job[future]
            try:
                neff = future.result()
                spike_kernel = create_spike_kernel(neff, job.kernel, job.kernel_args, job.kwargs)
                job.add_fields(neff=neff, spike_kernel=spike_kernel)
            except Exception as e:
                # TODO: catch the entire stdout, stderr
                exc_type, exc_value, exc_traceback = sys.exc_info()
                error_string = f"{exc_type.__name__}: {str(e)}\n"
                error_string += "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                job.add_fields(error=error_string)

        with SpikeExecutor(verbose=0) as spike:
            for job in self.jobs:
                if job.spike_kernel:
                    # FIXME: args are used, kwargs are needed to run but not used
                    stats = spike.benchmark(
                        job.spike_kernel,
                        *job.kernel_args,
                        **job.kwargs,
                        warmup_iterations=self.warmup,
                        benchmark_iterations=self.iters,
                        device_id=0,
                    )
                    self.results.add_result(config=job.kwargs, neff=job.neff, **stats)
                else:
                    self.results.add_result(config=job.kwargs, error=job.error, min_ms=float("inf"))

        self.results.save(cache_dir=self.cache_dir)
        if self.trace:
            for job in tqdm(self.jobs, total=len(self.jobs), desc="Tracing NEFFs"):
                if job.neff:
                    self._trace_neff(job.neff)
        return None

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
