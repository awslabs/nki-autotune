"""Golden values for data reuse tests.

Defines pre-tiled fixture functions, expected merge sources, and unified
test cases for the data reuse analysis and transform pipeline.
"""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

import nkigym


def tiled_matmul_1x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Single subgraph - no reuse possible."""
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output = nkigym.nc_matmul(a_sg0, b_sg0)
    return output


def tiled_matmul_2x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2 subgraphs (2 M tiles) - B is fully shared."""
    output = nkigym.ndarray((256, 128), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = nkigym.nc_matmul(a_sg0, b_sg0)
    a_sg1 = a[128:256, 0:128]
    b_sg1 = b[0:128, 0:128]
    output[128:256, 0:128] = nkigym.nc_matmul(a_sg1, b_sg1)
    return output


def tiled_matmul_1x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2 subgraphs (2 N tiles) - A is fully shared."""
    output = nkigym.ndarray((128, 256), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = nkigym.nc_matmul(a_sg0, b_sg0)
    a_sg1 = a[0:128, 0:128]
    b_sg1 = b[0:128, 128:256]
    output[0:128, 128:256] = nkigym.nc_matmul(a_sg1, b_sg1)
    return output


def tiled_matmul_2x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """4 subgraphs (2x2 grid) - both A and B have partial sharing."""
    output = nkigym.ndarray((256, 256), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = nkigym.nc_matmul(a_sg0, b_sg0)
    a_sg1 = a[0:128, 0:128]
    b_sg1 = b[0:128, 128:256]
    output[0:128, 128:256] = nkigym.nc_matmul(a_sg1, b_sg1)
    a_sg2 = a[128:256, 0:128]
    b_sg2 = b[0:128, 0:128]
    output[128:256, 0:128] = nkigym.nc_matmul(a_sg2, b_sg2)
    a_sg3 = a[128:256, 0:128]
    b_sg3 = b[0:128, 128:256]
    output[128:256, 128:256] = nkigym.nc_matmul(a_sg3, b_sg3)
    return output


def tiled_matmul_4x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """4 subgraphs (4 M tiles) - B is fully shared by all 4."""
    output = nkigym.ndarray((512, 128), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = nkigym.nc_matmul(a_sg0, b_sg0)
    a_sg1 = a[128:256, 0:128]
    b_sg1 = b[0:128, 0:128]
    output[128:256, 0:128] = nkigym.nc_matmul(a_sg1, b_sg1)
    a_sg2 = a[256:384, 0:128]
    b_sg2 = b[0:128, 0:128]
    output[256:384, 0:128] = nkigym.nc_matmul(a_sg2, b_sg2)
    a_sg3 = a[384:512, 0:128]
    b_sg3 = b[0:128, 0:128]
    output[384:512, 0:128] = nkigym.nc_matmul(a_sg3, b_sg3)
    return output


def tiled_double_matmul_1x1(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Double matmul, single subgraph - no reuse."""
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    temp_sg0 = nkigym.nc_matmul(a_sg0, b_sg0)
    c_sg0 = c[0:128, 0:128]
    output = nkigym.nc_matmul(temp_sg0, c_sg0)
    return output


def tiled_double_matmul_2x1(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Double matmul, 2 subgraphs - B and C are fully shared."""
    output = nkigym.ndarray((256, 128), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    temp_sg0 = nkigym.nc_matmul(a_sg0, b_sg0)
    c_sg0 = c[0:128, 0:128]
    output[0:128, 0:128] = nkigym.nc_matmul(temp_sg0, c_sg0)
    a_sg1 = a[128:256, 0:128]
    b_sg1 = b[0:128, 0:128]
    temp_sg1 = nkigym.nc_matmul(a_sg1, b_sg1)
    c_sg1 = c[0:128, 0:128]
    output[128:256, 0:128] = nkigym.nc_matmul(temp_sg1, c_sg1)
    return output


MERGED_MATMUL_2X1_B = """\
import numpy as np
import nkigym
def tiled_matmul_2x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    \"\"\"2 subgraphs (2 M tiles) - B is fully shared.\"\"\"
    output = nkigym.ndarray((256, 128), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = nkigym.nc_matmul(a_sg0, b_sg0)
    a_sg1 = a[128:256, 0:128]
    output[128:256, 0:128] = nkigym.nc_matmul(a_sg1, b_sg0)
    return output
"""

MERGED_MATMUL_1X2_A = """\
import numpy as np
import nkigym
def tiled_matmul_1x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    \"\"\"2 subgraphs (2 N tiles) - A is fully shared.\"\"\"
    output = nkigym.ndarray((128, 256), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = nkigym.nc_matmul(a_sg0, b_sg0)
    b_sg1 = b[0:128, 128:256]
    output[0:128, 128:256] = nkigym.nc_matmul(a_sg0, b_sg1)
    return output
"""

MERGED_MATMUL_2X2_A01 = """\
import numpy as np
import nkigym
def tiled_matmul_2x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    \"\"\"4 subgraphs (2x2 grid) - both A and B have partial sharing.\"\"\"
    output = nkigym.ndarray((256, 256), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = nkigym.nc_matmul(a_sg0, b_sg0)
    b_sg1 = b[0:128, 128:256]
    output[0:128, 128:256] = nkigym.nc_matmul(a_sg0, b_sg1)
    a_sg2 = a[128:256, 0:128]
    b_sg2 = b[0:128, 0:128]
    output[128:256, 0:128] = nkigym.nc_matmul(a_sg2, b_sg2)
    a_sg3 = a[128:256, 0:128]
    b_sg3 = b[0:128, 128:256]
    output[128:256, 128:256] = nkigym.nc_matmul(a_sg3, b_sg3)
    return output
"""

MERGED_MATMUL_2X2_B02 = """\
import numpy as np
import nkigym
def tiled_matmul_2x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    \"\"\"4 subgraphs (2x2 grid) - both A and B have partial sharing.\"\"\"
    output = nkigym.ndarray((256, 256), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = nkigym.nc_matmul(a_sg0, b_sg0)
    a_sg1 = a[0:128, 0:128]
    b_sg1 = b[0:128, 128:256]
    output[0:128, 128:256] = nkigym.nc_matmul(a_sg1, b_sg1)
    a_sg2 = a[128:256, 0:128]
    output[128:256, 0:128] = nkigym.nc_matmul(a_sg2, b_sg0)
    a_sg3 = a[128:256, 0:128]
    b_sg3 = b[0:128, 128:256]
    output[128:256, 128:256] = nkigym.nc_matmul(a_sg3, b_sg3)
    return output
"""

MERGED_MATMUL_4X1_B = """\
import numpy as np
import nkigym
def tiled_matmul_4x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    \"\"\"4 subgraphs (4 M tiles) - B is fully shared by all 4.\"\"\"
    output = nkigym.ndarray((512, 128), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = nkigym.nc_matmul(a_sg0, b_sg0)
    a_sg1 = a[128:256, 0:128]
    output[128:256, 0:128] = nkigym.nc_matmul(a_sg1, b_sg0)
    a_sg2 = a[256:384, 0:128]
    b_sg2 = b[0:128, 0:128]
    output[256:384, 0:128] = nkigym.nc_matmul(a_sg2, b_sg2)
    a_sg3 = a[384:512, 0:128]
    b_sg3 = b[0:128, 0:128]
    output[384:512, 0:128] = nkigym.nc_matmul(a_sg3, b_sg3)
    return output
"""

MERGED_DOUBLE_MATMUL_2X1_B = """\
import numpy as np
import nkigym
def tiled_double_matmul_2x1(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    \"\"\"Double matmul, 2 subgraphs - B and C are fully shared.\"\"\"
    output = nkigym.ndarray((256, 128), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    temp_sg0 = nkigym.nc_matmul(a_sg0, b_sg0)
    c_sg0 = c[0:128, 0:128]
    output[0:128, 0:128] = nkigym.nc_matmul(temp_sg0, c_sg0)
    a_sg1 = a[128:256, 0:128]
    temp_sg1 = nkigym.nc_matmul(a_sg1, b_sg0)
    c_sg1 = c[0:128, 0:128]
    output[128:256, 0:128] = nkigym.nc_matmul(temp_sg1, c_sg1)
    return output
"""

MERGED_DOUBLE_MATMUL_2X1_C = """\
import numpy as np
import nkigym
def tiled_double_matmul_2x1(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    \"\"\"Double matmul, 2 subgraphs - B and C are fully shared.\"\"\"
    output = nkigym.ndarray((256, 128), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    temp_sg0 = nkigym.nc_matmul(a_sg0, b_sg0)
    c_sg0 = c[0:128, 0:128]
    output[0:128, 0:128] = nkigym.nc_matmul(temp_sg0, c_sg0)
    a_sg1 = a[128:256, 0:128]
    b_sg1 = b[0:128, 0:128]
    temp_sg1 = nkigym.nc_matmul(a_sg1, b_sg1)
    output[128:256, 0:128] = nkigym.nc_matmul(temp_sg1, c_sg0)
    return output
"""

MERGED_MATMUL_4X1_ALL_B = """\
import numpy as np
import nkigym
def tiled_matmul_4x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    \"\"\"4 subgraphs (4 M tiles) - B is fully shared by all 4.\"\"\"
    output = nkigym.ndarray((512, 128), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = nkigym.nc_matmul(a_sg0, b_sg0)
    a_sg1 = a[128:256, 0:128]
    output[128:256, 0:128] = nkigym.nc_matmul(a_sg1, b_sg0)
    a_sg2 = a[256:384, 0:128]
    output[256:384, 0:128] = nkigym.nc_matmul(a_sg2, b_sg0)
    a_sg3 = a[384:512, 0:128]
    output[384:512, 0:128] = nkigym.nc_matmul(a_sg3, b_sg0)
    return output
"""

MERGED_MATMUL_2X2_ALL_GROUPS = """\
import numpy as np
import nkigym
def tiled_matmul_2x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    \"\"\"4 subgraphs (2x2 grid) - both A and B have partial sharing.\"\"\"
    output = nkigym.ndarray((256, 256), dtype=np.float32)
    a_sg0 = a[0:128, 0:128]
    b_sg0 = b[0:128, 0:128]
    output[0:128, 0:128] = nkigym.nc_matmul(a_sg0, b_sg0)
    b_sg1 = b[0:128, 128:256]
    output[0:128, 128:256] = nkigym.nc_matmul(a_sg0, b_sg1)
    a_sg2 = a[128:256, 0:128]
    output[128:256, 0:128] = nkigym.nc_matmul(a_sg2, b_sg0)
    output[128:256, 128:256] = nkigym.nc_matmul(a_sg2, b_sg1)
    return output
"""


class DataReuseCase(NamedTuple):
    """A single data reuse test case.

    Attributes:
        id: Test case identifier for pytest parametrize.
        func: Pre-tiled starting function.
        expected_pairs: Expected reuse pairs from analysis.
        merges: Ordered sequence of (tensor_a, tensor_b) merges to apply.
        expected_source: Expected source after all merges, or None if no merges.
        input_shapes: Input array shapes in parameter order for numerical check.
    """

    id: str
    func: Callable
    expected_pairs: list[tuple[str, str]]
    merges: list[tuple[str, str]]
    expected_source: str | None
    input_shapes: list[tuple[int, ...]]


CASES: list[DataReuseCase] = [
    DataReuseCase(
        id="matmul_1x1_no_reuse",
        func=tiled_matmul_1x1,
        expected_pairs=[],
        merges=[],
        expected_source=None,
        input_shapes=[(128, 128), (128, 128)],
    ),
    DataReuseCase(
        id="double_matmul_1x1_no_reuse",
        func=tiled_double_matmul_1x1,
        expected_pairs=[],
        merges=[],
        expected_source=None,
        input_shapes=[(128, 128), (128, 128), (128, 128)],
    ),
    DataReuseCase(
        id="matmul_2x1_merge_b",
        func=tiled_matmul_2x1,
        expected_pairs=[("b_sg0", "b_sg1")],
        merges=[("b_sg0", "b_sg1")],
        expected_source=MERGED_MATMUL_2X1_B,
        input_shapes=[(256, 128), (128, 128)],
    ),
    DataReuseCase(
        id="matmul_1x2_merge_a",
        func=tiled_matmul_1x2,
        expected_pairs=[("a_sg0", "a_sg1")],
        merges=[("a_sg0", "a_sg1")],
        expected_source=MERGED_MATMUL_1X2_A,
        input_shapes=[(128, 128), (128, 256)],
    ),
    DataReuseCase(
        id="matmul_2x2_merge_a01",
        func=tiled_matmul_2x2,
        expected_pairs=[("a_sg0", "a_sg1"), ("a_sg2", "a_sg3"), ("b_sg0", "b_sg2"), ("b_sg1", "b_sg3")],
        merges=[("a_sg0", "a_sg1")],
        expected_source=MERGED_MATMUL_2X2_A01,
        input_shapes=[(256, 128), (128, 256)],
    ),
    DataReuseCase(
        id="matmul_2x2_merge_b02",
        func=tiled_matmul_2x2,
        expected_pairs=[("a_sg0", "a_sg1"), ("a_sg2", "a_sg3"), ("b_sg0", "b_sg2"), ("b_sg1", "b_sg3")],
        merges=[("b_sg0", "b_sg2")],
        expected_source=MERGED_MATMUL_2X2_B02,
        input_shapes=[(256, 128), (128, 256)],
    ),
    DataReuseCase(
        id="matmul_4x1_merge_b01",
        func=tiled_matmul_4x1,
        expected_pairs=[
            ("b_sg0", "b_sg1"),
            ("b_sg0", "b_sg2"),
            ("b_sg0", "b_sg3"),
            ("b_sg1", "b_sg2"),
            ("b_sg1", "b_sg3"),
            ("b_sg2", "b_sg3"),
        ],
        merges=[("b_sg0", "b_sg1")],
        expected_source=MERGED_MATMUL_4X1_B,
        input_shapes=[(512, 128), (128, 128)],
    ),
    DataReuseCase(
        id="double_matmul_2x1_merge_b",
        func=tiled_double_matmul_2x1,
        expected_pairs=[("b_sg0", "b_sg1"), ("c_sg0", "c_sg1")],
        merges=[("b_sg0", "b_sg1")],
        expected_source=MERGED_DOUBLE_MATMUL_2X1_B,
        input_shapes=[(256, 128), (128, 128), (128, 128)],
    ),
    DataReuseCase(
        id="double_matmul_2x1_merge_c",
        func=tiled_double_matmul_2x1,
        expected_pairs=[("b_sg0", "b_sg1"), ("c_sg0", "c_sg1")],
        merges=[("c_sg0", "c_sg1")],
        expected_source=MERGED_DOUBLE_MATMUL_2X1_C,
        input_shapes=[(256, 128), (128, 128), (128, 128)],
    ),
    DataReuseCase(
        id="matmul_4x1_merge_all_b",
        func=tiled_matmul_4x1,
        expected_pairs=[
            ("b_sg0", "b_sg1"),
            ("b_sg0", "b_sg2"),
            ("b_sg0", "b_sg3"),
            ("b_sg1", "b_sg2"),
            ("b_sg1", "b_sg3"),
            ("b_sg2", "b_sg3"),
        ],
        merges=[("b_sg0", "b_sg1"), ("b_sg0", "b_sg2"), ("b_sg0", "b_sg3")],
        expected_source=MERGED_MATMUL_4X1_ALL_B,
        input_shapes=[(512, 128), (128, 128)],
    ),
    DataReuseCase(
        id="matmul_2x2_merge_all_groups",
        func=tiled_matmul_2x2,
        expected_pairs=[("a_sg0", "a_sg1"), ("a_sg2", "a_sg3"), ("b_sg0", "b_sg2"), ("b_sg1", "b_sg3")],
        merges=[("a_sg0", "a_sg1"), ("a_sg2", "a_sg3"), ("b_sg0", "b_sg2"), ("b_sg1", "b_sg3")],
        expected_source=MERGED_MATMUL_2X2_ALL_GROUPS,
        input_shapes=[(256, 128), (128, 256)],
    ),
]
