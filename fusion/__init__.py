"""
FusionChain: Universal kernel fusion implementation based on the MegaFuse paper.

This package provides an extensible framework for fusing blocking operators
with accumulation operations, enabling concurrent execution while maintaining
mathematical equivalence.
"""

from fusion.fusion_chain import FusionChain, RMSNormMatmul
from fusion.fusion_typing import FusionConfig, FusionState
from fusion.operators import AccumulationOperator, BlockingOperator, create_rmsnorm_matmul
from fusion.tensors import Tensor

__all__ = [
    "FusionChain",
    "RMSNormMatmul",
    "Tensor",
    "BlockingOperator",
    "AccumulationOperator",
    "create_rmsnorm_matmul",
    "FusionConfig",
    "FusionState",
]

__version__ = "1.0.0"
