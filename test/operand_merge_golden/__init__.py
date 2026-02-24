"""Operand merge golden test data package."""

from typing import NamedTuple

from operand_merge_golden.accumulation import (  # noqa: F401
    ACCUMULATION_BLOCKS_AFTER_1_MERGE,
    ACCUMULATION_BLOCKS_AFTER_2_MERGES,
    ACCUMULATION_BLOCKS_AFTER_3_MERGES,
    ACCUMULATION_BLOCKS_PROGRAM,
)
from operand_merge_golden.basic import (  # noqa: F401
    AFTER_ADJACENT_LOADS_2X,
    AFTER_ADJACENT_LOADS_2X_PARTIAL,
    BEFORE_ADJACENT_LOADS_2X,
    BEFORE_DIFFERENT_PARTITION_SLICES,
    BEFORE_DIFFERENT_SOURCE_TENSORS,
    BEFORE_NO_ADJACENT_LOADS,
    BEFORE_SINGLE_SUBGRAPH,
)
from operand_merge_golden.corner_cases import (  # noqa: F401
    BEFORE_ALREADY_MERGED,
    BEFORE_HETEROGENEOUS_OPS,
    BEFORE_SINGLE_ACTIVATION,
)
from operand_merge_golden.elementwise import (  # noqa: F401
    AFTER_ACTIVATION_2X,
    AFTER_TENSOR_SCALAR_2X,
    AFTER_TENSOR_TENSOR_2X,
    AFTER_TENSOR_TENSOR_DIFF_OPS,
    BEFORE_ACTIVATION_2X,
    BEFORE_TENSOR_SCALAR_2X,
    BEFORE_TENSOR_TENSOR_2X,
    BEFORE_TENSOR_TENSOR_DIFF_OPS,
)
from operand_merge_golden.limits import (  # noqa: F401
    AFTER_EXCEEDS_N_LIMIT,
    AFTER_EXCEEDS_N_LIMIT_PARTIAL,
    AFTER_N_AT_LIMIT,
    BEFORE_MATMUL_EXCEEDS_N_LIMIT,
    BEFORE_MATMUL_N_AT_LIMIT,
)
from operand_merge_golden.limits_m import (  # noqa: F401
    AFTER_M_DIM_MERGE,
    AFTER_M_EXCEEDS_LIMIT,
    BEFORE_MATMUL_M_DIM_MERGE,
    BEFORE_MATMUL_M_EXCEEDS_LIMIT,
)
from operand_merge_golden.post_reuse import (  # noqa: F401
    AFTER_POST_REUSE_1X2,
    AFTER_POST_REUSE_1X4,
    AFTER_POST_REUSE_2X2,
    AFTER_POST_REUSE_2X2_PARTIAL,
    BEFORE_MATMUL_POST_REUSE_1X2,
    BEFORE_MATMUL_POST_REUSE_1X4,
    BEFORE_MATMUL_POST_REUSE_2X2,
)
from operand_merge_golden.scaling import AFTER_ADJACENT_4X, AFTER_ADJACENT_4X_PARTIAL, BEFORE_ADJACENT_4X  # noqa: F401
from operand_merge_golden.structured import (  # noqa: F401
    DEPENDENCY_BLOCKS_AFTER_1_MERGE,
    DEPENDENCY_BLOCKS_AFTER_2_MERGES,
    DEPENDENCY_BLOCKS_PROGRAM,
)

from nkigym.ir import GymProgram


class OperandMergeCase(NamedTuple):
    """A single operand merge test case.

    Attributes:
        id: Test case identifier for pytest parametrize.
        before: Pre-merge GymProgram.
        merge_count: Number of merge iterations to apply.
        after: Expected post-merge GymProgram.
    """

    id: str
    before: GymProgram
    merge_count: int
    after: GymProgram


CORNER_CASES: list[OperandMergeCase] = [
    OperandMergeCase("no_adjacent_loads", BEFORE_NO_ADJACENT_LOADS, 0, BEFORE_NO_ADJACENT_LOADS),
    OperandMergeCase("single_subgraph_no_merge", BEFORE_SINGLE_SUBGRAPH, 0, BEFORE_SINGLE_SUBGRAPH),
    OperandMergeCase("different_source_tensors", BEFORE_DIFFERENT_SOURCE_TENSORS, 0, BEFORE_DIFFERENT_SOURCE_TENSORS),
    OperandMergeCase(
        "different_partition_slices", BEFORE_DIFFERENT_PARTITION_SLICES, 0, BEFORE_DIFFERENT_PARTITION_SLICES
    ),
    OperandMergeCase("single_activation", BEFORE_SINGLE_ACTIVATION, 0, BEFORE_SINGLE_ACTIVATION),
    OperandMergeCase("heterogeneous_ops", BEFORE_HETEROGENEOUS_OPS, 0, BEFORE_HETEROGENEOUS_OPS),
    OperandMergeCase("already_merged", BEFORE_ALREADY_MERGED, 0, BEFORE_ALREADY_MERGED),
    OperandMergeCase(
        "dependency_blocks_exhausted", DEPENDENCY_BLOCKS_AFTER_2_MERGES, 0, DEPENDENCY_BLOCKS_AFTER_2_MERGES
    ),
    OperandMergeCase(
        "accumulation_blocks_exhausted", ACCUMULATION_BLOCKS_AFTER_3_MERGES, 0, ACCUMULATION_BLOCKS_AFTER_3_MERGES
    ),
]

CASES: list[OperandMergeCase] = [
    OperandMergeCase("adjacent_loads_2x_partial", BEFORE_ADJACENT_LOADS_2X, 1, AFTER_ADJACENT_LOADS_2X_PARTIAL),
    OperandMergeCase("adjacent_loads_2x", BEFORE_ADJACENT_LOADS_2X, 3, AFTER_ADJACENT_LOADS_2X),
    OperandMergeCase("adjacent_4x_partial", BEFORE_ADJACENT_4X, 3, AFTER_ADJACENT_4X_PARTIAL),
    OperandMergeCase("adjacent_4x", BEFORE_ADJACENT_4X, 9, AFTER_ADJACENT_4X),
    OperandMergeCase("post_reuse_1x2", BEFORE_MATMUL_POST_REUSE_1X2, 3, AFTER_POST_REUSE_1X2),
    OperandMergeCase("post_reuse_1x4", BEFORE_MATMUL_POST_REUSE_1X4, 9, AFTER_POST_REUSE_1X4),
    OperandMergeCase("post_reuse_2x2_partial", BEFORE_MATMUL_POST_REUSE_2X2, 2, AFTER_POST_REUSE_2X2_PARTIAL),
    OperandMergeCase("post_reuse_2x2", BEFORE_MATMUL_POST_REUSE_2X2, 6, AFTER_POST_REUSE_2X2),
    OperandMergeCase("n_at_limit", BEFORE_MATMUL_N_AT_LIMIT, 3, AFTER_N_AT_LIMIT),
    OperandMergeCase("exceeds_n_limit_partial", BEFORE_MATMUL_EXCEEDS_N_LIMIT, 4, AFTER_EXCEEDS_N_LIMIT_PARTIAL),
    OperandMergeCase("exceeds_n_limit", BEFORE_MATMUL_EXCEEDS_N_LIMIT, 10, AFTER_EXCEEDS_N_LIMIT),
    OperandMergeCase("m_dim_merge", BEFORE_MATMUL_M_DIM_MERGE, 3, AFTER_M_DIM_MERGE),
    OperandMergeCase("m_exceeds_limit", BEFORE_MATMUL_M_EXCEEDS_LIMIT, 1, AFTER_M_EXCEEDS_LIMIT),
    OperandMergeCase("tensor_tensor_2x", BEFORE_TENSOR_TENSOR_2X, 1, AFTER_TENSOR_TENSOR_2X),
    OperandMergeCase("tensor_tensor_diff_ops", BEFORE_TENSOR_TENSOR_DIFF_OPS, 1, AFTER_TENSOR_TENSOR_DIFF_OPS),
    OperandMergeCase("activation_2x", BEFORE_ACTIVATION_2X, 3, AFTER_ACTIVATION_2X),
    OperandMergeCase("tensor_scalar_2x", BEFORE_TENSOR_SCALAR_2X, 3, AFTER_TENSOR_SCALAR_2X),
    OperandMergeCase("dependency_blocks_load_merge", DEPENDENCY_BLOCKS_PROGRAM, 1, DEPENDENCY_BLOCKS_AFTER_1_MERGE),
    OperandMergeCase(
        "dependency_blocks_compute_merge", DEPENDENCY_BLOCKS_AFTER_1_MERGE, 1, DEPENDENCY_BLOCKS_AFTER_2_MERGES
    ),
    OperandMergeCase(
        "accumulation_blocks_load_merge_1", ACCUMULATION_BLOCKS_PROGRAM, 1, ACCUMULATION_BLOCKS_AFTER_1_MERGE
    ),
    OperandMergeCase(
        "accumulation_blocks_load_merge_2", ACCUMULATION_BLOCKS_AFTER_1_MERGE, 1, ACCUMULATION_BLOCKS_AFTER_2_MERGES
    ),
    OperandMergeCase(
        "accumulation_blocks_compute_merge", ACCUMULATION_BLOCKS_AFTER_2_MERGES, 1, ACCUMULATION_BLOCKS_AFTER_3_MERGES
    ),
]
