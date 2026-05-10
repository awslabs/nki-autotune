"""Compile ``rmsnorm(lhs) @ rhs`` from numpy through the full nkigym pipeline.

Runs the three stages of ``nkigym_compile`` end-to-end:

    1. ``synthesis``        — synthesise ``f_nkigym`` via the Claude
                              Agent SDK; write ``<cache>/f_nkigym.py``.
    2. ``initial_codegen``  — render the canonical eager NKI kernel
                              into ``<cache>/kernel.py`` and validate
                              against the numpy reference through
                              ``nki.simulate``.
    3. ``tune`` (batch)     — enumerate the rewrite pool, sample
                              ``num_kernels`` kernels, render each
                              into ``<cache>/kernel_tuned_NNNN.py``,
                              CPU-sim + HW profile via
                              ``autotune.remote_profile``. Results
                              land in ``<cache>/results.json``.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/rmsnorm_matmul.py
"""

import shutil
from pathlib import Path

import numpy as np

from nkigym import nkigym_compile


def rmsnorm_matmul_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Plain-numpy ``rmsnorm(lhs) @ rhs`` golden."""
    m = np.mean(np.square(lhs), axis=1, keepdims=True)
    rms_inv = 1.0 / np.sqrt(m + 1e-6)
    normed = lhs * rms_inv
    return normed @ rhs


if __name__ == "__main__":
    cache_dir = Path("/home/ubuntu/cache/rmsnorm_matmul_compile")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)

    M, K, N = 2048, 2048, 2048
    INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}

    """Batch tune path: enumerate the rewrite graph from the canonical
    forest, sample 16 distinct kernels, render each, profile on the
    gym via autotune.remote_profile. Adjust ``num_kernels`` / ``hosts``
    as needed; the ``seed`` kwarg controls sampling reproducibility."""
    nkigym_compile(
        f=rmsnorm_matmul_numpy,
        input_specs=INPUT_SPECS,
        cache_dir=cache_dir,
        num_kernels=100,
        hosts=["gym-1", "gym-2", "gym-3"],
        venv_python="/home/ubuntu/venvs/kernel-env/bin/python",
        neuron_platform_target="trn2",
        seed=0,
    )
    print(f"[rmsnorm_matmul] canonical kernel: {cache_dir / 'kernel.py'}")
    print(f"[rmsnorm_matmul] results.json:     {cache_dir / 'results.json'}")
