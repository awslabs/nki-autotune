"""Profile 100 NKI kernel variants across remote Trainium hosts.

Sends a simple tensor-copy kernel (repeated 100 times as distinct
"variants") to remote workers for compilation and benchmarking.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/remote_profile.py
    python examples/remote_profile.py --cache-dir /home/ubuntu/cache/remote_profile
"""

import logging

from autotune.runner.api import remote_profile
from autotune.runner.types import ProfileConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

KERNEL_SOURCE = """\
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def nki_tensor_copy(a):
    output = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.shared_hbm)
    sbuf_a = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sbuf_a[0:128, 0:512], src=a[0:128, 0:512])
    sbuf_out = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sbuf_out[0:128, 0:512], src=sbuf_a[0:128, 0:512])
    nisa.dma_copy(dst=output[0:128, 0:512], src=sbuf_out[0:128, 0:512])
    return output
"""


def main() -> None:
    """Profile 100 kernel variants and print results."""
    kernels = {f"copy_v{i}.py": KERNEL_SOURCE for i in range(100)}

    output = remote_profile(
        kernels=kernels,
        input_specs={"a": ((128, 512), "bfloat16")},
        hosts=["gym-1", "gym-2", "gym-3", "gym-4", "gym-5"],
        cache_dir="/home/ubuntu/cache/remote_profile_test_5w",
        config=ProfileConfig(warmup=10, iters=100),
    )
    print(output)

    output = remote_profile(
        kernels=kernels,
        input_specs={"a": ((128, 512), "bfloat16")},
        hosts=["gym-1"],
        cache_dir="/home/ubuntu/cache/remote_profile_test_1w",
        config=ProfileConfig(warmup=10, iters=100),
    )
    print(output)


if __name__ == "__main__":
    main()
