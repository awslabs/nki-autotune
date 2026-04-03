"""NKI operator definitions --- importing triggers auto-registration."""

from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.affine_select import NKIAffineSelect
from nkigym.ops.base import NKIOp
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose

__all__ = [
    "NKIOp",
    "NKIActivation",
    "NKIActivationReduce",
    "NKIAffineSelect",
    "NKIMatmul",
    "NKITensorReduce",
    "NKITensorScalar",
    "NKITranspose",
]
