"""Compute theoretical MAC count from a KernelIR."""

from nkigym.kernel_ir import KernelIR
from nkigym.ops.matmul import NKIMatmul


def compute_mac_count(ir: KernelIR) -> int:
    """Total MAC count across every ``NKIMatmul`` op in the IR.

    Per op MACs are the product of the concrete ``K``, ``M``, ``N``
    dim sizes resolved via ``per_op_axis_maps``. MACs are a property
    of the math function (dim sizes × number of matmuls), not the
    lowering — identical across all tiling/fusion variants.
    """
    da = ir.dim_analysis
    total = 0
    for op_idx, op_cls in enumerate(ir.op_graph.op_classes):
        if op_cls is not NKIMatmul:
            continue
        axis_map = da.per_op_axis_maps[op_idx]
        total += da.dims[axis_map["K"]].dim_size * da.dims[axis_map["M"]].dim_size * da.dims[axis_map["N"]].dim_size
    return total
