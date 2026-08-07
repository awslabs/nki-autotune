"""Rewrite transforms over :class:`nkigym.ir.KernelIR`."""

from typing import Any

from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption
from nkigym.transforms.batch_permutation import BatchPermutation, BatchPermutationOption
from nkigym.transforms.buffer_compaction import BufferCompaction, BufferCompactionOption
from nkigym.transforms.buffer_layout import BufferLayout, BufferLayoutOption
from nkigym.transforms.buffer_placement import BufferPlacement, BufferPlacementOption
from nkigym.transforms.code_motion import CodeMotion, CodeMotionOption
from nkigym.transforms.common_subexpression_elimination import (
    CommonSubexpressionElimination,
    CommonSubexpressionEliminationOption,
)
from nkigym.transforms.copy_propagation import CopyPropagation, CopyPropagationOption
from nkigym.transforms.decompose_broadcast_subtract import DecomposeBroadcastSubtract, DecomposeBroadcastSubtractOption
from nkigym.transforms.eliminate_identity_initializer import (
    EliminateIdentityInitializer,
    EliminateIdentityInitializerOption,
)
from nkigym.transforms.fuse import Fuse, FuseOption
from nkigym.transforms.fuse_pointwise import FusePointwise, FusePointwiseOption
from nkigym.transforms.online_fusion import OnlineFusion, OnlineFusionOption
from nkigym.transforms.reorder import Reorder, ReorderOption
from nkigym.transforms.rfactor import RFactor, RFactorOption
from nkigym.transforms.software_pipeline import SoftwarePipeline, SoftwarePipelineOption
from nkigym.transforms.split import Split, SplitOption
from nkigym.transforms.transpose_pair import (
    CancelTransposePairOption,
    InsertTransposePairOption,
    TransposePair,
    TransposePairOption,
)
from nkigym.transforms.transpose_through_load import TransposeThroughLoad, TransposeThroughLoadOption
from nkigym.transforms.transpose_through_matmul import TransposeThroughMatmul, TransposeThroughMatmulOption
from nkigym.transforms.transpose_through_tensor_copy import TransposeThroughTensorCopy, TransposeThroughTensorCopyOption

PUBLIC_TRANSFORM_TYPES: tuple[type[Transform[Any]], ...] = (
    BatchPermutation,
    BufferPlacement,
    BufferCompaction,
    BufferLayout,
    CodeMotion,
    CommonSubexpressionElimination,
    CopyPropagation,
    DecomposeBroadcastSubtract,
    EliminateIdentityInitializer,
    Fuse,
    FusePointwise,
    OnlineFusion,
    Reorder,
    RFactor,
    SoftwarePipeline,
    Split,
    TransposePair,
    TransposeThroughLoad,
    TransposeThroughMatmul,
    TransposeThroughTensorCopy,
)


def public_transforms() -> list[Transform[Any]]:
    """Instantiate the complete public transform action namespace."""
    return [transform_type() for transform_type in PUBLIC_TRANSFORM_TYPES]


__all__ = [
    "BatchPermutation",
    "BatchPermutationOption",
    "BufferCompaction",
    "BufferCompactionOption",
    "BufferLayout",
    "BufferLayoutOption",
    "BufferPlacement",
    "BufferPlacementOption",
    "CodeMotion",
    "CodeMotionOption",
    "CommonSubexpressionElimination",
    "CommonSubexpressionEliminationOption",
    "CopyPropagation",
    "CopyPropagationOption",
    "DecomposeBroadcastSubtract",
    "DecomposeBroadcastSubtractOption",
    "EliminateIdentityInitializer",
    "EliminateIdentityInitializerOption",
    "CancelTransposePairOption",
    "Fuse",
    "FuseOption",
    "FusePointwise",
    "FusePointwiseOption",
    "InsertTransposePairOption",
    "OnlineFusion",
    "OnlineFusionOption",
    "PUBLIC_TRANSFORM_TYPES",
    "Reorder",
    "ReorderOption",
    "RFactor",
    "RFactorOption",
    "SoftwarePipeline",
    "SoftwarePipelineOption",
    "Split",
    "SplitOption",
    "Transform",
    "TransformLegalityError",
    "TransformOption",
    "TransposePair",
    "TransposePairOption",
    "TransposeThroughLoad",
    "TransposeThroughLoadOption",
    "TransposeThroughMatmul",
    "TransposeThroughMatmulOption",
    "TransposeThroughTensorCopy",
    "TransposeThroughTensorCopyOption",
    "public_transforms",
]
