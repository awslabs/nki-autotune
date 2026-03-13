"""Transform passes for NKI Gym.

Transforms are atomic rewrites on NKIKernel. Each transform is
independent and self-contained: it inspects the kernel, finds optimization
opportunities, and applies them one at a time.

Every transform follows the analyze/apply protocol defined by the
``NKITransform`` base class:

1. ``analyze(kernel)`` — inspect the kernel and return a list of TransformOptions.
2. ``apply(kernel, option)`` — apply a single option, returning a new kernel.
"""

from nkigym.transforms.base import NKITransform, StmtRef, TransformOption
from nkigym.transforms.data_reuse import DataReuseTransform
from nkigym.transforms.operand_merge import OperandMergeTransform

__all__ = ["DataReuseTransform", "NKITransform", "OperandMergeTransform", "StmtRef", "TransformOption"]
