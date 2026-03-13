"""Interpret an NKIKernel with numpy: simulate(kernel, kwargs) -> np.ndarray."""

from typing import Any

import numpy as np

from nkigym.codegen.types import NKIKernel


def simulate(kernel: NKIKernel, kwargs: dict[str, np.ndarray]) -> np.ndarray:
    """Execute an NKIKernel with numpy, returning the output array.

    Args:
        kernel: The NKI kernel to interpret.
        kwargs: Input arrays keyed by parameter name.

    Returns:
        The output numpy array.
    """
    env: dict[str, Any] = {k: np.asarray(v, dtype=np.float64) for k, v in kwargs.items()}
    env["output"] = np.zeros(kernel.output_shape, dtype=np.float64)
    for block in kernel.blocks:
        for stmt in block.body:
            stmt.simulate(env)
    return env["output"]
