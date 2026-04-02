"""NKI operator definitions --- importing triggers auto-registration."""

from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_1d import NKIActivation1D
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.add import NKIAdd
from nkigym.ops.affine_select import NKIAffineSelect
from nkigym.ops.base import NKIOp
from nkigym.ops.dma_copy import NKIDmaCopy
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.multiply import NKIMultiply
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.tensor_scalar_const import NKITensorScalarConst
from nkigym.ops.transpose import NKITranspose

__all__ = [
    "NKIOp",
    "NKIAffineSelect",
    "NKIMatmul",
    "NKIActivation",
    "NKIActivation1D",
    "NKIActivationReduce",
    "NKIAdd",
    "NKIDmaCopy",
    "NKIMultiply",
    "NKITensorCopy",
    "NKITensorReduce",
    "NKITensorScalar",
    "NKITensorScalarConst",
    "NKITranspose",
]
