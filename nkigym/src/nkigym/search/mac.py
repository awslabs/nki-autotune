"""Compute theoretical MAC count from a math function.

MACs are a property of the math function (matmul dim sizes ×
number of matmuls), not the lowering. Computed from the
pre-rewrite analysis so it remains stable even when
``apply_online_fusion`` replaces matmul nodes with composite ops
that absorb them.
"""

from collections.abc import Callable

import numpy as np

from nkigym.kernel_ir.dim_analysis import analyze_dims
from nkigym.kernel_ir.op_graph import build_op_graph
from nkigym.ops.matmul import NKIMatmul


def compute_mac_count(func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> int:
    """Total MAC count across every ``NKIMatmul`` op in the pre-rewrite IR.

    Runs ``analyze_dims`` / ``build_op_graph`` directly so the
    result is immune to IR rewrites (e.g. online fusion) that
    absorb matmul nodes into composite ops.
    """
    da = analyze_dims(func, input_specs)
    graph = build_op_graph(func, input_specs)
    total = 0
    for op_idx, op_cls in enumerate(graph.op_classes):
        if op_cls is not NKIMatmul:
            continue
        axis_map = da.per_op_axis_maps[op_idx]
        total += da.dims[axis_map["K"]].dim_size * da.dims[axis_map["M"]].dim_size * da.dims[axis_map["N"]].dim_size
    return total
