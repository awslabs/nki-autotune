"""Compute theoretical MAC count from a math function."""

from collections.abc import Callable

import numpy as np

from nkigym.kernel_ir import build_ir


def compute_mac_count(func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> int:
    """Total MAC count across every ``NKIMatmul`` op in the baseline IR."""
    ir = build_ir(func, input_specs)
    total = 0
    for op in ir.ops:
        if op.kind != "NKIMatmul":
            continue
        axis_map = op.axis_map
        total += (
            ir.dimensions[axis_map["K"]].dim_size
            * ir.dimensions[axis_map["M"]].dim_size
            * ir.dimensions[axis_map["N"]].dim_size
        )
    return total
