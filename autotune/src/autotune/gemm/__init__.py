"""GEMM (General Matrix Multiplication) package for NKI autotune.

Provides template-based kernel generation, configuration management,
validation, and tile overlap utilities.
"""

from autotune.gemm.config import GEMMConfig, sample_gemm_configs
from autotune.gemm.kernels import MetaGEMM, lhs_rhs_meta_gemm, lhsT_rhs_meta_gemm
from autotune.gemm.utils import calculate_tile_overlap_ranges
from autotune.gemm.validation import gemm_correctness_check, lhs_rhs_gemm_golden, lhsT_rhs_gemm_golden

__all__ = [
    "GEMMConfig",
    "sample_gemm_configs",
    "lhsT_rhs_meta_gemm",
    "lhs_rhs_meta_gemm",
    "MetaGEMM",
    "gemm_correctness_check",
    "lhsT_rhs_gemm_golden",
    "lhs_rhs_gemm_golden",
    "calculate_tile_overlap_ranges",
]
