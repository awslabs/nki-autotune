"""Rewrite transforms over :class:`nkigym.ir.KernelIR`."""

from importlib import import_module
from typing import Any, cast

from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption

_PUBLIC_TRANSFORMS = (
    ("batch_permutation", "BatchPermutation"),
    ("buffer_placement", "BufferPlacement"),
    ("buffer_compaction", "BufferCompaction"),
    ("buffer_layout", "BufferLayout"),
    ("code_motion", "CodeMotion"),
    ("common_subexpression_elimination", "CommonSubexpressionElimination"),
    ("copy_propagation", "CopyPropagation"),
    ("decompose_broadcast_subtract", "DecomposeBroadcastSubtract"),
    ("eliminate_identity_initializer", "EliminateIdentityInitializer"),
    ("fuse", "Fuse"),
    ("fuse_pointwise", "FusePointwise"),
    ("online_fusion", "OnlineFusion"),
    ("reorder", "Reorder"),
    ("rfactor", "RFactor"),
    ("software_pipeline", "SoftwarePipeline"),
    ("split", "Split"),
    ("transpose_pair", "TransposePair"),
    ("transpose_through_load", "TransposeThroughLoad"),
    ("transpose_through_matmul", "TransposeThroughMatmul"),
    ("transpose_through_tensor_copy", "TransposeThroughTensorCopy"),
)
_EXPORT_MODULES = {
    "BatchPermutation": "batch_permutation",
    "BatchPermutationOption": "batch_permutation",
    "BufferCompaction": "buffer_compaction",
    "BufferCompactionOption": "buffer_compaction",
    "BufferLayout": "buffer_layout",
    "BufferLayoutOption": "buffer_layout",
    "BufferPlacement": "buffer_placement",
    "BufferPlacementOption": "buffer_placement",
    "CancelTransposePairOption": "transpose_pair",
    "CodeMotion": "code_motion",
    "CodeMotionOption": "code_motion",
    "CommonSubexpressionElimination": "common_subexpression_elimination",
    "CommonSubexpressionEliminationOption": "common_subexpression_elimination",
    "CopyPropagation": "copy_propagation",
    "CopyPropagationOption": "copy_propagation",
    "DecomposeBroadcastSubtract": "decompose_broadcast_subtract",
    "DecomposeBroadcastSubtractOption": "decompose_broadcast_subtract",
    "EliminateIdentityInitializer": "eliminate_identity_initializer",
    "EliminateIdentityInitializerOption": "eliminate_identity_initializer",
    "Fuse": "fuse",
    "FuseOption": "fuse",
    "FusePointwise": "fuse_pointwise",
    "FusePointwiseOption": "fuse_pointwise",
    "InsertTransposePairOption": "transpose_pair",
    "OnlineFusion": "online_fusion",
    "OnlineFusionOption": "online_fusion",
    "Reorder": "reorder",
    "ReorderOption": "reorder",
    "RFactor": "rfactor",
    "RFactorOption": "rfactor",
    "SoftwarePipeline": "software_pipeline",
    "SoftwarePipelineOption": "software_pipeline",
    "Split": "split",
    "SplitOption": "split",
    "TransposePair": "transpose_pair",
    "TransposePairOption": "transpose_pair",
    "TransposeThroughLoad": "transpose_through_load",
    "TransposeThroughLoadOption": "transpose_through_load",
    "TransposeThroughMatmul": "transpose_through_matmul",
    "TransposeThroughMatmulOption": "transpose_through_matmul",
    "TransposeThroughTensorCopy": "transpose_through_tensor_copy",
    "TransposeThroughTensorCopyOption": "transpose_through_tensor_copy",
}

BatchPermutation: type[Transform[Any]]
BatchPermutationOption: type[TransformOption]
BufferCompaction: type[Transform[Any]]
BufferCompactionOption: type[TransformOption]
BufferLayout: type[Transform[Any]]
BufferLayoutOption: type[TransformOption]
BufferPlacement: type[Transform[Any]]
BufferPlacementOption: type[TransformOption]
CancelTransposePairOption: type[TransformOption]
CodeMotion: type[Transform[Any]]
CodeMotionOption: type[TransformOption]
CommonSubexpressionElimination: type[Transform[Any]]
CommonSubexpressionEliminationOption: type[TransformOption]
CopyPropagation: type[Transform[Any]]
CopyPropagationOption: type[TransformOption]
DecomposeBroadcastSubtract: type[Transform[Any]]
DecomposeBroadcastSubtractOption: type[TransformOption]
EliminateIdentityInitializer: type[Transform[Any]]
EliminateIdentityInitializerOption: type[TransformOption]
Fuse: type[Transform[Any]]
FuseOption: type[TransformOption]
FusePointwise: type[Transform[Any]]
FusePointwiseOption: type[TransformOption]
InsertTransposePairOption: type[TransformOption]
OnlineFusion: type[Transform[Any]]
OnlineFusionOption: type[TransformOption]
PUBLIC_TRANSFORM_TYPES: tuple[type[Transform[Any]], ...]
Reorder: type[Transform[Any]]
ReorderOption: type[TransformOption]
RFactor: type[Transform[Any]]
RFactorOption: type[TransformOption]
SoftwarePipeline: type[Transform[Any]]
SoftwarePipelineOption: type[TransformOption]
Split: type[Transform[Any]]
SplitOption: type[TransformOption]
TransposePair: type[Transform[Any]]
TransposePairOption: type[TransformOption]
TransposeThroughLoad: type[Transform[Any]]
TransposeThroughLoadOption: type[TransformOption]
TransposeThroughMatmul: type[Transform[Any]]
TransposeThroughMatmulOption: type[TransformOption]
TransposeThroughTensorCopy: type[Transform[Any]]
TransposeThroughTensorCopyOption: type[TransformOption]


def _load_export(name: str) -> object:
    """Load and cache one public transform symbol."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def _public_transform_types() -> tuple[type[Transform[Any]], ...]:
    """Load the ordered public transform type registry."""
    return tuple(
        (cast(type[Transform[Any]], _load_export(class_name)) for _module_name, class_name in _PUBLIC_TRANSFORMS)
    )


def public_transforms() -> list[Transform[Any]]:
    """Instantiate the complete public transform action namespace."""
    return [transform_type() for transform_type in _public_transform_types()]


def __getattr__(name: str) -> object:
    """Resolve transform exports without eagerly importing every implementation."""
    if name == "PUBLIC_TRANSFORM_TYPES":
        value: object = _public_transform_types()
        globals()[name] = value
    else:
        value = _load_export(name)
    return value


__all__ = [
    "BatchPermutation",
    "BatchPermutationOption",
    "BufferCompaction",
    "BufferCompactionOption",
    "BufferLayout",
    "BufferLayoutOption",
    "BufferPlacement",
    "BufferPlacementOption",
    "CancelTransposePairOption",
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
