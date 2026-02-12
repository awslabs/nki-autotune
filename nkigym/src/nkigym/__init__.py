"""NKI Gym - Tunable kernel environment for AWS Trainium hardware.

Pipeline: workload spec -> tiling -> transforms -> lower to NKI

Subpackages:
    ops: Operator definitions and registry (NKIOp, NKIMatmul, OP_REGISTRY)
    tiling: Dimension analysis and tiled code generation
    transforms: Optimization passes on the tiled IR (e.g., data reuse)
    codegen: Lowering from nkigym IR to target kernel code (e.g., NKI)
    utils: Code generation helpers and logging
    ir: Conversion between callable, source, and IR representations
"""

from nkigym.ops import (
    ALLOC_F32_OP,
    ALLOC_F64_OP,
    ALLOC_OPS,
    LOAD_OP,
    NC_MATMUL_OP,
    STORE_OP,
    AllocOp,
    LoadOp,
    StoreOp,
    ndarray,
)

nc_matmul = NC_MATMUL_OP

__all__ = [
    "nc_matmul",
    "ndarray",
    "LoadOp",
    "StoreOp",
    "AllocOp",
    "LOAD_OP",
    "STORE_OP",
    "ALLOC_F32_OP",
    "ALLOC_F64_OP",
    "ALLOC_OPS",
    "NC_MATMUL_OP",
]
