# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GEMM (General Matrix Multiplication) package for NKI autotune.

This package contains all GEMM-related functionality including:
- Kernel implementations
- Configuration management
- Validation and correctness checking
- Utility functions
"""

from autotune.gemm.config import GEMMConfig, sample_gemm_configs
from autotune.gemm.kernels import MetaGEMM, lhs_rhs_meta_gemm, lhsT_rhs_meta_gemm
from autotune.gemm.utils import calculate_tile_overlap_ranges
from autotune.gemm.validation import GEMMCorrectness

__all__ = [
    # Configuration
    "GEMMConfig",
    "sample_gemm_configs",
    # Kernels
    "lhsT_rhs_meta_gemm",
    "lhs_rhs_meta_gemm",
    "MetaGEMM",
    # Validation
    "GEMMCorrectness",
    # Utils
    "calculate_tile_overlap_ranges",
]
