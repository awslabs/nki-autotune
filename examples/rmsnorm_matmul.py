"""Compile ``rmsnorm(lhs) @ rhs`` from numpy through the full nkigym pipeline.

Drives the staged ``nkigym_compile`` pipeline:

    1. ``"synthesis"`` — synthesise ``f_nkigym`` from the numpy reference
       via the Claude Agent SDK; write ``<cache>/f_nkigym.py``.
    2. ``"initial_codegen"`` — render the canonical eager NKI kernel into
       ``<cache>/kernel.py`` and auto-validate it against the numpy
       reference through ``nki.simulate``.
    3. ``"tune"`` — randomly draw legal fusion atoms from the current
       forest (seeded for reproducibility) and apply them in sequence,
       render the transformed kernel into ``<cache>/kernel_tuned.py``,
       and auto-validate the tuned kernel against numpy.

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

    """The tune stage defaults to a seeded random draw: after each
    apply, it re-enumerates legal fusion atoms from the current forest,
    flips an independent coin on each, and applies the first survivor.
    The loop terminates when no atom survives the coin flip. Change the
    ``seed`` kwarg to sample a different tuning outcome."""
    nkigym_compile(
        rmsnorm_matmul_numpy, INPUT_SPECS, cache_dir, stages=["synthesis", "initial_codegen", "tune"], seed=0
    )
    print(f"[rmsnorm_matmul] canonical kernel: {cache_dir / 'kernel.py'}")
    print(f"[rmsnorm_matmul] tuned kernel:     {cache_dir / 'kernel_tuned.py'}")
