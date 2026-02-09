"""NKI Gym - Tunable kernel environment for AWS Trainium hardware.

Pipeline: workload spec -> tiling -> transforms -> lower to NKI

Subpackages:
    ops: Operator definitions and registry (NKIOp, NKIMatmul, OP_REGISTRY)
    tiling: Dimension analysis and tiled code generation
    transforms: Optimization passes on the tiled IR (e.g., data reuse)
    lower: Lowering from nkigym IR to target kernel code (e.g., NKI)
    utils: Code generation helpers and logging
"""

from nkigym.ops import NKIMatmul, ndarray

nc_matmul: NKIMatmul = NKIMatmul()

__all__ = ["nc_matmul", "ndarray"]
