"""Golden values for data reuse analysis tests.

This module defines pre-tiled fixture functions and their expected reuse groups.
Tests pass these functions directly to analyze_data_reuse, which parses their
source code to identify tensor slices that access identical data.

Naming convention for fixture functions:
- tiled_matmul_{M}x{N}: Single matmul with MxN tile grid
- tiled_double_matmul_{M}x{N}: Double matmul with MxN tile grid

Reuse patterns for matmul C[m,n] = A[m,k] @ B[k,n]:
- A varies with M tiles (row dimension)
- B varies with N tiles (column dimension)
- When M > 1, N = 1: B is shared across all subgraphs
- When M = 1, N > 1: A is shared across all subgraphs
- When M > 1, N > 1: Both A and B have partial sharing
"""

from collections.abc import Callable

import numpy as np


def tiled_matmul_1x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Single subgraph - no reuse possible."""
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output = np.matmul(a_sg0, b_sg0)
    return output


def tiled_matmul_2x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2 subgraphs (2 M tiles) - B is fully shared."""
    output = np.empty((256, 128), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = np.matmul(a_sg0, b_sg0)
    a_sg1 = a[128:256, 0:128]
    b_sg1 = b[0:128, 0:128]
    output[128:256, 0:128] = np.matmul(a_sg1, b_sg1)
    return output


def tiled_matmul_1x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2 subgraphs (2 N tiles) - A is fully shared."""
    output = np.empty((128, 256), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = np.matmul(a_sg0, b_sg0)
    a_sg1 = a[0:128, 0:128]
    b_sg1 = b[0:128, 128:256]
    output[0:128, 128:256] = np.matmul(a_sg1, b_sg1)
    return output


def tiled_matmul_2x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """4 subgraphs (2x2 grid) - both A and B have partial sharing."""
    output = np.empty((256, 256), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = np.matmul(a_sg0, b_sg0)
    a_sg1 = a[0:128, 0:128]
    b_sg1 = b[0:128, 128:256]
    output[0:128, 128:256] = np.matmul(a_sg1, b_sg1)
    a_sg2 = a[128:256, 0:128]
    b_sg2 = b[0:128, 0:128]
    output[128:256, 0:128] = np.matmul(a_sg2, b_sg2)
    a_sg3 = a[128:256, 0:128]
    b_sg3 = b[0:128, 128:256]
    output[128:256, 128:256] = np.matmul(a_sg3, b_sg3)
    return output


def tiled_matmul_4x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """4 subgraphs (4 M tiles) - B is fully shared by all 4."""
    output = np.empty((512, 128), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = np.matmul(a_sg0, b_sg0)
    a_sg1 = a[128:256, 0:128]
    b_sg1 = b[0:128, 0:128]
    output[128:256, 0:128] = np.matmul(a_sg1, b_sg1)
    a_sg2 = a[256:384, 0:128]
    b_sg2 = b[0:128, 0:128]
    output[256:384, 0:128] = np.matmul(a_sg2, b_sg2)
    a_sg3 = a[384:512, 0:128]
    b_sg3 = b[0:128, 0:128]
    output[384:512, 0:128] = np.matmul(a_sg3, b_sg3)
    return output


def tiled_matmul_1x4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """4 subgraphs (4 N tiles) - A is fully shared by all 4."""
    output = np.empty((128, 512), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = np.matmul(a_sg0, b_sg0)
    a_sg1 = a[0:128, 0:128]
    b_sg1 = b[0:128, 128:256]
    output[0:128, 128:256] = np.matmul(a_sg1, b_sg1)
    a_sg2 = a[0:128, 0:128]
    b_sg2 = b[0:128, 256:384]
    output[0:128, 256:384] = np.matmul(a_sg2, b_sg2)
    a_sg3 = a[0:128, 0:128]
    b_sg3 = b[0:128, 384:512]
    output[0:128, 384:512] = np.matmul(a_sg3, b_sg3)
    return output


def tiled_matmul_3x3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """9 subgraphs (3x3 grid) - both A and B have partial sharing across rows/columns.

    A shape: (384, 128) = 3 tiles in M dimension
    B shape: (128, 384) = 3 tiles in N dimension
    Output shape: (384, 384)

    Reuse patterns:
    - A tensors: 3 groups of 3 (same row shares A)
      - (a_sg0, a_sg1, a_sg2) - row 0
      - (a_sg3, a_sg4, a_sg5) - row 1
      - (a_sg6, a_sg7, a_sg8) - row 2
    - B tensors: 3 groups of 3 (same column shares B)
      - (b_sg0, b_sg3, b_sg6) - column 0
      - (b_sg1, b_sg4, b_sg7) - column 1
      - (b_sg2, b_sg5, b_sg8) - column 2
    """
    output = np.empty((384, 384), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = np.matmul(a_sg0, b_sg0)
    a_sg1 = a[0:128, 0:128]
    b_sg1 = b[0:128, 128:256]
    output[0:128, 128:256] = np.matmul(a_sg1, b_sg1)
    a_sg2 = a[0:128, 0:128]
    b_sg2 = b[0:128, 256:384]
    output[0:128, 256:384] = np.matmul(a_sg2, b_sg2)
    a_sg3 = a[128:256, 0:128]
    b_sg3 = b[0:128, 0:128]
    output[128:256, 0:128] = np.matmul(a_sg3, b_sg3)
    a_sg4 = a[128:256, 0:128]
    b_sg4 = b[0:128, 128:256]
    output[128:256, 128:256] = np.matmul(a_sg4, b_sg4)
    a_sg5 = a[128:256, 0:128]
    b_sg5 = b[0:128, 256:384]
    output[128:256, 256:384] = np.matmul(a_sg5, b_sg5)
    a_sg6 = a[256:384, 0:128]
    b_sg6 = b[0:128, 0:128]
    output[256:384, 0:128] = np.matmul(a_sg6, b_sg6)
    a_sg7 = a[256:384, 0:128]
    b_sg7 = b[0:128, 128:256]
    output[256:384, 128:256] = np.matmul(a_sg7, b_sg7)
    a_sg8 = a[256:384, 0:128]
    b_sg8 = b[0:128, 256:384]
    output[256:384, 256:384] = np.matmul(a_sg8, b_sg8)
    return output


def tiled_double_matmul_1x1(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Double matmul, single subgraph - no reuse."""
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    temp_sg0 = np.matmul(a_sg0, b_sg0)
    c_sg0 = c[0:128, 0:128]
    output = np.matmul(temp_sg0, c_sg0)
    return output


def tiled_double_matmul_2x1(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Double matmul, 2 subgraphs - B and C are fully shared."""
    output = np.empty((256, 128), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    temp_sg0 = np.matmul(a_sg0, b_sg0)
    c_sg0 = c[0:128, 0:128]
    output[0:128, 0:128] = np.matmul(temp_sg0, c_sg0)
    a_sg1 = a[128:256, 0:128]
    b_sg1 = b[0:128, 0:128]
    temp_sg1 = np.matmul(a_sg1, b_sg1)
    c_sg1 = c[0:128, 0:128]
    output[128:256, 0:128] = np.matmul(temp_sg1, c_sg1)
    return output


def tiled_double_matmul_2x2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Double matmul, 4 subgraphs - B is fully shared, A and C have partial sharing."""
    output = np.empty((256, 256), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    temp_sg0 = np.matmul(a_sg0, b_sg0)
    c_sg0 = c[0:128, 0:128]
    output[0:128, 0:128] = np.matmul(temp_sg0, c_sg0)
    a_sg1 = a[0:128, 0:128]
    b_sg1 = b[0:128, 0:128]
    temp_sg1 = np.matmul(a_sg1, b_sg1)
    c_sg1 = c[0:128, 128:256]
    output[0:128, 128:256] = np.matmul(temp_sg1, c_sg1)
    a_sg2 = a[128:256, 0:128]
    b_sg2 = b[0:128, 0:128]
    temp_sg2 = np.matmul(a_sg2, b_sg2)
    c_sg2 = c[0:128, 0:128]
    output[128:256, 0:128] = np.matmul(temp_sg2, c_sg2)
    a_sg3 = a[128:256, 0:128]
    b_sg3 = b[0:128, 0:128]
    temp_sg3 = np.matmul(a_sg3, b_sg3)
    c_sg3 = c[0:128, 128:256]
    output[128:256, 128:256] = np.matmul(temp_sg3, c_sg3)
    return output


EXPECTED_REUSE: dict[Callable, list[tuple[str, ...]]] = {
    tiled_matmul_1x1: [],
    tiled_matmul_2x1: [("b_sg0", "b_sg1")],
    tiled_matmul_1x2: [("a_sg0", "a_sg1")],
    tiled_matmul_2x2: [("a_sg0", "a_sg1"), ("a_sg2", "a_sg3"), ("b_sg0", "b_sg2"), ("b_sg1", "b_sg3")],
    tiled_matmul_4x1: [("b_sg0", "b_sg1", "b_sg2", "b_sg3")],
    tiled_matmul_1x4: [("a_sg0", "a_sg1", "a_sg2", "a_sg3")],
    tiled_matmul_3x3: [
        ("a_sg0", "a_sg1", "a_sg2"),
        ("a_sg3", "a_sg4", "a_sg5"),
        ("a_sg6", "a_sg7", "a_sg8"),
        ("b_sg0", "b_sg3", "b_sg6"),
        ("b_sg1", "b_sg4", "b_sg7"),
        ("b_sg2", "b_sg5", "b_sg8"),
    ],
    tiled_double_matmul_1x1: [],
    tiled_double_matmul_2x1: [("b_sg0", "b_sg1"), ("c_sg0", "c_sg1")],
    tiled_double_matmul_2x2: [
        ("a_sg0", "a_sg1"),
        ("a_sg2", "a_sg3"),
        ("b_sg0", "b_sg1", "b_sg2", "b_sg3"),
        ("c_sg0", "c_sg2"),
        ("c_sg1", "c_sg3"),
    ],
}


MERGED_MATMUL_2X1_B = """\
def tiled_matmul_2x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    \"\"\"2 subgraphs (2 M tiles) - B is fully shared.\"\"\"
    output = np.empty((256, 128), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = np.matmul(a_sg0, b_sg0)
    a_sg1 = a[128:256, 0:128]
    output[128:256, 0:128] = np.matmul(a_sg1, b_sg0)
    return output
"""

MERGED_MATMUL_1X2_A = """\
def tiled_matmul_1x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    \"\"\"2 subgraphs (2 N tiles) - A is fully shared.\"\"\"
    output = np.empty((128, 256), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = np.matmul(a_sg0, b_sg0)
    b_sg1 = b[0:128, 128:256]
    output[0:128, 128:256] = np.matmul(a_sg0, b_sg1)
    return output
"""

MERGED_MATMUL_2X2_A01 = """\
def tiled_matmul_2x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    \"\"\"4 subgraphs (2x2 grid) - both A and B have partial sharing.\"\"\"
    output = np.empty((256, 256), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = np.matmul(a_sg0, b_sg0)
    b_sg1 = b[0:128, 128:256]
    output[0:128, 128:256] = np.matmul(a_sg0, b_sg1)
    a_sg2 = a[128:256, 0:128]
    b_sg2 = b[0:128, 0:128]
    output[128:256, 0:128] = np.matmul(a_sg2, b_sg2)
    a_sg3 = a[128:256, 0:128]
    b_sg3 = b[0:128, 128:256]
    output[128:256, 128:256] = np.matmul(a_sg3, b_sg3)
    return output
"""

MERGED_MATMUL_2X2_B02 = """\
def tiled_matmul_2x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    \"\"\"4 subgraphs (2x2 grid) - both A and B have partial sharing.\"\"\"
    output = np.empty((256, 256), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = np.matmul(a_sg0, b_sg0)
    a_sg1 = a[0:128, 0:128]
    b_sg1 = b[0:128, 128:256]
    output[0:128, 128:256] = np.matmul(a_sg1, b_sg1)
    a_sg2 = a[128:256, 0:128]
    output[128:256, 0:128] = np.matmul(a_sg2, b_sg0)
    a_sg3 = a[128:256, 0:128]
    b_sg3 = b[0:128, 128:256]
    output[128:256, 128:256] = np.matmul(a_sg3, b_sg3)
    return output
"""

MERGED_MATMUL_4X1_B = """\
def tiled_matmul_4x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    \"\"\"4 subgraphs (4 M tiles) - B is fully shared by all 4.\"\"\"
    output = np.empty((512, 128), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = np.matmul(a_sg0, b_sg0)
    a_sg1 = a[128:256, 0:128]
    output[128:256, 0:128] = np.matmul(a_sg1, b_sg0)
    a_sg2 = a[256:384, 0:128]
    b_sg2 = b[0:128, 0:128]
    output[256:384, 0:128] = np.matmul(a_sg2, b_sg2)
    a_sg3 = a[384:512, 0:128]
    b_sg3 = b[0:128, 0:128]
    output[384:512, 0:128] = np.matmul(a_sg3, b_sg3)
    return output
"""

MERGED_DOUBLE_MATMUL_2X1_B = """\
def tiled_double_matmul_2x1(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    \"\"\"Double matmul, 2 subgraphs - B and C are fully shared.\"\"\"
    output = np.empty((256, 128), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    temp_sg0 = np.matmul(a_sg0, b_sg0)
    c_sg0 = c[0:128, 0:128]
    output[0:128, 0:128] = np.matmul(temp_sg0, c_sg0)
    a_sg1 = a[128:256, 0:128]
    temp_sg1 = np.matmul(a_sg1, b_sg0)
    c_sg1 = c[0:128, 0:128]
    output[128:256, 0:128] = np.matmul(temp_sg1, c_sg1)
    return output
"""

MERGED_DOUBLE_MATMUL_2X1_C = """\
def tiled_double_matmul_2x1(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    \"\"\"Double matmul, 2 subgraphs - B and C are fully shared.\"\"\"
    output = np.empty((256, 128), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    temp_sg0 = np.matmul(a_sg0, b_sg0)
    c_sg0 = c[0:128, 0:128]
    output[0:128, 0:128] = np.matmul(temp_sg0, c_sg0)
    a_sg1 = a[128:256, 0:128]
    b_sg1 = b[0:128, 0:128]
    temp_sg1 = np.matmul(a_sg1, b_sg1)
    output[128:256, 0:128] = np.matmul(temp_sg1, c_sg0)
    return output
"""

MERGED_MATMUL_4X1_ALL_B = """\
def tiled_matmul_4x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    \"\"\"4 subgraphs (4 M tiles) - B is fully shared by all 4.\"\"\"
    output = np.empty((512, 128), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = np.matmul(a_sg0, b_sg0)
    a_sg1 = a[128:256, 0:128]
    output[128:256, 0:128] = np.matmul(a_sg1, b_sg0)
    a_sg2 = a[256:384, 0:128]
    output[256:384, 0:128] = np.matmul(a_sg2, b_sg0)
    a_sg3 = a[384:512, 0:128]
    output[384:512, 0:128] = np.matmul(a_sg3, b_sg0)
    return output
"""

MERGED_MATMUL_2X2_ALL_GROUPS = """\
def tiled_matmul_2x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    \"\"\"4 subgraphs (2x2 grid) - both A and B have partial sharing.\"\"\"
    output = np.empty((256, 256), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = np.matmul(a_sg0, b_sg0)
    b_sg1 = b[0:128, 128:256]
    output[0:128, 128:256] = np.matmul(a_sg0, b_sg1)
    a_sg2 = a[128:256, 0:128]
    output[128:256, 0:128] = np.matmul(a_sg2, b_sg0)
    output[128:256, 128:256] = np.matmul(a_sg2, b_sg1)
    return output
"""

EXPECTED_MERGE_TRANSFORMS: dict[tuple[Callable, str, str], str] = {
    (tiled_matmul_2x1, "b_sg0", "b_sg1"): MERGED_MATMUL_2X1_B,
    (tiled_matmul_1x2, "a_sg0", "a_sg1"): MERGED_MATMUL_1X2_A,
    (tiled_matmul_2x2, "a_sg0", "a_sg1"): MERGED_MATMUL_2X2_A01,
    (tiled_matmul_2x2, "b_sg0", "b_sg2"): MERGED_MATMUL_2X2_B02,
    (tiled_matmul_4x1, "b_sg0", "b_sg1"): MERGED_MATMUL_4X1_B,
    (tiled_double_matmul_2x1, "b_sg0", "b_sg1"): MERGED_DOUBLE_MATMUL_2X1_B,
    (tiled_double_matmul_2x1, "c_sg0", "c_sg1"): MERGED_DOUBLE_MATMUL_2X1_C,
}


MERGE_ERROR_CASES: list[tuple[Callable, str, str, type, str]] = [
    (tiled_matmul_2x1, "nonexistent_sg0", "b_sg1", ValueError, "not found"),
    (tiled_matmul_2x1, "b_sg0", "nonexistent_sg1", ValueError, "not found"),
    (tiled_matmul_2x1, "a_sg0", "a_sg1", ValueError, "not share identical slices"),
    (tiled_matmul_2x1, "b_sg0", "b_sg0", ValueError, "Cannot merge .* with itself"),
]
