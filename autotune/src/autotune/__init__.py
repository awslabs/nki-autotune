# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NKI kernel profiling backend.

Distributes NKI kernel compilation and benchmarking across remote
Trainium hosts via SSH::

    from autotune.runner.remote import remote_profile

    output = remote_profile(
        kernels={"copy_v0.py": source, "copy_v1.py": source},
        input_specs={"a": ((128, 512), "bfloat16")},
        hosts=["gym-1", "gym-2", "gym-3"],
    )
    print(output)
"""
