"""NKI Gym - Tunable kernel environment for AWS Trainium hardware.

Pipeline: workload spec -> tiling -> transforms -> lower to NKI

Subpackages:
    ops: Operator definitions and registry (GymOp, MatmulOp, etc.)
    tiling: Dimension analysis and tiled code generation
    transforms: Optimization passes on the tiled IR (e.g., data reuse)
    codegen: Lowering from nkigym IR to target kernel code (e.g., NKI)
    utils: Code generation helpers and logging
    ir: Conversion between callable, source, and IR representations
"""

from nkigym.ops import ActivationOp, GymOp, MatmulOp, NcTransposeOp, TensorScalarOp, TensorTensorOp

nc_matmul = MatmulOp()
nc_transpose = NcTransposeOp()
activation = ActivationOp()
tensor_tensor = TensorTensorOp()
tensor_scalar = TensorScalarOp()

__all__ = [
    "GymOp",
    "MatmulOp",
    "NcTransposeOp",
    "ActivationOp",
    "TensorTensorOp",
    "TensorScalarOp",
    "nc_matmul",
    "nc_transpose",
    "activation",
    "tensor_tensor",
    "tensor_scalar",
]
