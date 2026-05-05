"""Compile ``rmsnorm(lhs) @ rhs`` from numpy to nkigym.

The user writes the plain-numpy reference and ``INPUT_SPECS``; one
``compile_numpy_to_nkigym`` call drives the translation and returns
validated ``f_nkigym`` source.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/rmsnorm_matmul.py
"""

import shutil
from pathlib import Path

import numpy as np

from nkigym.synthesis import compile_numpy_to_nkigym


def rmsnorm_matmul_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Plain-numpy golden for the nkipy baremetal_jit baseline."""
    m = np.mean(np.square(lhs), axis=1, keepdims=True)
    rms_inv = 1.0 / np.sqrt(m + 1e-6)
    normed = lhs * rms_inv
    output = normed @ rhs
    return output


if __name__ == "__main__":
    cache_dir = Path("/home/ubuntu/cache/rmsnorm_matmul_compile")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)

    M, K, N = 2048, 2048, 2048
    INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}

    source = compile_numpy_to_nkigym(rmsnorm_matmul_numpy, INPUT_SPECS)
    (cache_dir / "f_nkigym.py").write_text(source)
