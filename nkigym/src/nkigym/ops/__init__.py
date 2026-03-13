"""NKI operator definitions.

Importing each op module triggers NKIOp auto-registration.
"""

from nkigym.ops.activation import NKIActivation
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import NKIOp
from nkigym.ops.dma_copy import NKIDmaCopy
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_copy import NKITensorCopy

__all__ = ["NKIOp", "NKIMatmul", "NKIActivation", "NKIAlloc", "NKIDmaCopy", "NKITensorCopy"]
