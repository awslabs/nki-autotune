"""NKI operator definitions.

Importing each op module triggers GymOp auto-registration.
"""

from nkigym.ops.activation import ActivationOp
from nkigym.ops.base import GymOp, Tensor
from nkigym.ops.matmul import MatmulOp
from nkigym.ops.nc_transpose import NcTransposeOp
from nkigym.ops.tensor_scalar import TensorScalarOp
from nkigym.ops.tensor_tensor import TensorTensorOp
from nkigym.ops.tiling_ops import AllocateOp, IndexingOp, LoadOp, StoreOp

__all__ = [
    "GymOp",
    "Tensor",
    "MatmulOp",
    "NcTransposeOp",
    "ActivationOp",
    "TensorTensorOp",
    "TensorScalarOp",
    "AllocateOp",
    "LoadOp",
    "IndexingOp",
    "StoreOp",
]
