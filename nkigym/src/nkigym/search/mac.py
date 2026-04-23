"""Compute theoretical MAC count from a math function."""

from collections.abc import Callable

import numpy as np

from nkigym.kernel_ir import build_initial
from nkigym.ops.matmul import NKIMatmul


def compute_mac_count(func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> int:
    """Total MAC count across every ``NKIMatmul`` op in the pre-rewrite IR."""
    ir, _graph = build_initial(func, input_specs)
    total = 0
    for op in ir.op_inputs:
        if not isinstance(op, NKIMatmul):
            continue
        axis_map = ir.op_axis_map[op]
        total += (
            ir.dimensions[axis_map["K"]].dim_size
            * ir.dimensions[axis_map["M"]].dim_size
            * ir.dimensions[axis_map["N"]].dim_size
        )
    return total
