# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NKI kernel profiling backend (local, on-box).

Compiles and benchmarks NKI kernels in-process on a Trainium box::

    from autotune.runner.api import profile

    output = profile(
        kernels={"copy_v0.py": job0, "copy_v1.py": job1},
        cache_dir="/home/ubuntu/autotune_cache/copy",
        seed=42,
        neuron_platform_target="trn2",
        collect_detailed_profile=False,
    )
    print(output)

To run on a remote box, drive this through ``transport/kaizen.sh``
(sync code, execute, download artifacts).
"""
