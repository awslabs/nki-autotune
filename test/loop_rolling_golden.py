"""Golden values for loop rolling tests.

Contains tiled fixtures and expected rolled sources for various matmul
tiling configurations. Used by test_loop_rolling.py for validation.
"""

from collections.abc import Callable

import nkigym
from nkigym.tiling import generate_tiled_function

TILE = 128


def _make_tiled(a_shape: tuple[int, int], b_shape: tuple[int, int]) -> Callable:
    """Generate a tiled matmul function for given shapes.

    Args:
        a_shape: Shape of first input.
        b_shape: Shape of second input.

    Returns:
        Tiled callable with __source__ attribute.
    """

    def matmul(a, b):
        """Compute matrix multiplication."""
        return nkigym.nc_matmul(a, b)

    return generate_tiled_function(matmul, {"a": a_shape, "b": b_shape})


GOLDEN: dict[str, tuple[tuple[int, int], tuple[int, int], str]] = {}
ROLL_ONCE: dict[str, tuple[tuple[int, int], tuple[int, int], str]] = {}


def _g(name: str, a_shape: tuple[int, int], b_shape: tuple[int, int], expected: str) -> None:
    """Register a golden test case for full roll_loops().

    Args:
        name: Test case identifier.
        a_shape: Shape of first input.
        b_shape: Shape of second input.
        expected: Expected rolled source string.
    """
    GOLDEN[name] = (a_shape, b_shape, expected)


def _r(name: str, a_shape: tuple[int, int], b_shape: tuple[int, int], expected: str) -> None:
    """Register a golden test case for single-pass _roll_once().

    Args:
        name: Test case identifier.
        a_shape: Shape of first input.
        b_shape: Shape of second input.
        expected: Expected source after one roll pass.
    """
    ROLL_ONCE[name] = (a_shape, b_shape, expected)


_g(
    "1x1",
    (TILE, TILE),
    (TILE, TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    return output""",
)

_g(
    "2x1",
    (TILE, 2 * TILE),
    (TILE, TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((256, 128), dtype=np.float32)
    for i_0 in range(2):
        tensor_0 = a[0:128, i_0 * 128:(i_0 + 1) * 128]
        tensor_1 = b[0:128, 0:128]
        tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
        output[i_0 * 128:(i_0 + 1) * 128, 0:128] = tensor_2[0:128, 0:128]
    return output""",
)

_g(
    "3x1",
    (TILE, 3 * TILE),
    (TILE, TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((384, 128), dtype=np.float32)
    for i_0 in range(3):
        tensor_0 = a[0:128, i_0 * 128:(i_0 + 1) * 128]
        tensor_1 = b[0:128, 0:128]
        tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
        output[i_0 * 128:(i_0 + 1) * 128, 0:128] = tensor_2[0:128, 0:128]
    return output""",
)

_g(
    "4x1",
    (TILE, 4 * TILE),
    (TILE, TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((512, 128), dtype=np.float32)
    for i_0 in range(4):
        tensor_0 = a[0:128, i_0 * 128:(i_0 + 1) * 128]
        tensor_1 = b[0:128, 0:128]
        tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
        output[i_0 * 128:(i_0 + 1) * 128, 0:128] = tensor_2[0:128, 0:128]
    return output""",
)

_g(
    "1x4",
    (TILE, TILE),
    (TILE, 4 * TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((128, 512), dtype=np.float32)
    for i_0 in range(4):
        tensor_0 = a[0:128, 0:128]
        tensor_1 = b[0:128, i_0 * 128:(i_0 + 1) * 128]
        tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
        output[0:128, i_0 * 128:(i_0 + 1) * 128] = tensor_2[0:128, 0:128]
    return output""",
)

_g(
    "1x5",
    (TILE, TILE),
    (TILE, 5 * TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((128, 640), dtype=np.float32)
    for i_0 in range(5):
        tensor_0 = a[0:128, 0:128]
        tensor_1 = b[0:128, i_0 * 128:(i_0 + 1) * 128]
        tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
        output[0:128, i_0 * 128:(i_0 + 1) * 128] = tensor_2[0:128, 0:128]
    return output""",
)

_g(
    "2x2",
    (TILE, 2 * TILE),
    (TILE, 2 * TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((256, 256), dtype=np.float32)
    for i_0 in range(2):
        for i_1 in range(2):
            tensor_0 = a[0:128, i_0 * 128:(i_0 + 1) * 128]
            tensor_1 = b[0:128, i_1 * 128:(i_1 + 1) * 128]
            tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
            output[i_0 * 128:(i_0 + 1) * 128, i_1 * 128:(i_1 + 1) * 128] = tensor_2[0:128, 0:128]
    return output""",
)

_g(
    "3x5",
    (TILE, 3 * TILE),
    (TILE, 5 * TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((384, 640), dtype=np.float32)
    for i_0 in range(3):
        for i_1 in range(5):
            tensor_0 = a[0:128, i_0 * 128:(i_0 + 1) * 128]
            tensor_1 = b[0:128, i_1 * 128:(i_1 + 1) * 128]
            tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
            output[i_0 * 128:(i_0 + 1) * 128, i_1 * 128:(i_1 + 1) * 128] = tensor_2[0:128, 0:128]
    return output""",
)

_g(
    "4x4",
    (TILE, 4 * TILE),
    (TILE, 4 * TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((512, 512), dtype=np.float32)
    for i_0 in range(4):
        for i_1 in range(4):
            tensor_0 = a[0:128, i_0 * 128:(i_0 + 1) * 128]
            tensor_1 = b[0:128, i_1 * 128:(i_1 + 1) * 128]
            tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
            output[i_0 * 128:(i_0 + 1) * 128, i_1 * 128:(i_1 + 1) * 128] = tensor_2[0:128, 0:128]
    return output""",
)

_g(
    "red2",
    (2 * TILE, TILE),
    (2 * TILE, TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = a[128:256, 0:128]
    tensor_4 = b[128:256, 0:128]
    tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    return output""",
)

_g(
    "red4",
    (4 * TILE, TILE),
    (4 * TILE, TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    for i_0 in range(3):
        tensor_3 = a[(i_0 + 1) * 128:(i_0 + 2) * 128, 0:128]
        tensor_4 = b[(i_0 + 1) * 128:(i_0 + 2) * 128, 0:128]
        tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    return output""",
)

_g(
    "red8",
    (8 * TILE, TILE),
    (8 * TILE, TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    for i_0 in range(7):
        tensor_3 = a[(i_0 + 1) * 128:(i_0 + 2) * 128, 0:128]
        tensor_4 = b[(i_0 + 1) * 128:(i_0 + 2) * 128, 0:128]
        tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    return output""",
)

_g(
    "2x2_red2",
    (2 * TILE, 2 * TILE),
    (2 * TILE, 2 * TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((256, 256), dtype=np.float32)
    for i_0 in range(2):
        for i_1 in range(2):
            tensor_0 = a[0:128, i_0 * 128:(i_0 + 1) * 128]
            tensor_1 = b[0:128, i_1 * 128:(i_1 + 1) * 128]
            tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
            tensor_3 = a[128:256, i_0 * 128:(i_0 + 1) * 128]
            tensor_4 = b[128:256, i_1 * 128:(i_1 + 1) * 128]
            tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
            output[i_0 * 128:(i_0 + 1) * 128, i_1 * 128:(i_1 + 1) * 128] = tensor_2[0:128, 0:128]
    return output""",
)

_g(
    "3x5_red2",
    (2 * TILE, 3 * TILE),
    (2 * TILE, 5 * TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((384, 640), dtype=np.float32)
    for i_0 in range(3):
        for i_1 in range(5):
            tensor_0 = a[0:128, i_0 * 128:(i_0 + 1) * 128]
            tensor_1 = b[0:128, i_1 * 128:(i_1 + 1) * 128]
            tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
            tensor_3 = a[128:256, i_0 * 128:(i_0 + 1) * 128]
            tensor_4 = b[128:256, i_1 * 128:(i_1 + 1) * 128]
            tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
            output[i_0 * 128:(i_0 + 1) * 128, i_1 * 128:(i_1 + 1) * 128] = tensor_2[0:128, 0:128]
    return output""",
)

_g(
    "2x3_red3",
    (3 * TILE, 2 * TILE),
    (3 * TILE, 3 * TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((256, 384), dtype=np.float32)
    for i_0 in range(2):
        for i_1 in range(3):
            tensor_0 = a[0:128, i_0 * 128:(i_0 + 1) * 128]
            tensor_1 = b[0:128, i_1 * 128:(i_1 + 1) * 128]
            tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
            for i_2 in range(2):
                tensor_3 = a[(i_2 + 1) * 128:(i_2 + 2) * 128, i_0 * 128:(i_0 + 1) * 128]
                tensor_4 = b[(i_2 + 1) * 128:(i_2 + 2) * 128, i_1 * 128:(i_1 + 1) * 128]
                tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
            output[i_0 * 128:(i_0 + 1) * 128, i_1 * 128:(i_1 + 1) * 128] = tensor_2[0:128, 0:128]
    return output""",
)


_r(
    "1x1",
    (TILE, TILE),
    (TILE, TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    return output""",
)

_r(
    "2x1",
    (TILE, 2 * TILE),
    (TILE, TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((256, 128), dtype=np.float32)
    for i_0 in range(2):
        tensor_0 = a[0:128, i_0 * 128:(i_0 + 1) * 128]
        tensor_1 = b[0:128, 0:128]
        tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
        output[i_0 * 128:(i_0 + 1) * 128, 0:128] = tensor_2[0:128, 0:128]
    return output""",
)

_r(
    "4x1",
    (TILE, 4 * TILE),
    (TILE, TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((512, 128), dtype=np.float32)
    for i_0 in range(4):
        tensor_0 = a[0:128, i_0 * 128:(i_0 + 1) * 128]
        tensor_1 = b[0:128, 0:128]
        tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
        output[i_0 * 128:(i_0 + 1) * 128, 0:128] = tensor_2[0:128, 0:128]
    return output""",
)

_r(
    "2x2",
    (TILE, 2 * TILE),
    (TILE, 2 * TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((256, 256), dtype=np.float32)
    for i_0 in range(2):
        tensor_0 = a[0:128, i_0 * 128:(i_0 + 1) * 128]
        tensor_1 = b[0:128, 0:128]
        tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
        output[i_0 * 128:(i_0 + 1) * 128, 0:128] = tensor_2[0:128, 0:128]
        tensor_3 = a[0:128, i_0 * 128:(i_0 + 1) * 128]
        tensor_4 = b[0:128, 128:256]
        tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
        output[i_0 * 128:(i_0 + 1) * 128, 128:256] = tensor_5[0:128, 0:128]
    return output""",
)

_r(
    "red4",
    (4 * TILE, TILE),
    (4 * TILE, TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    for i_0 in range(3):
        tensor_3 = a[(i_0 + 1) * 128:(i_0 + 2) * 128, 0:128]
        tensor_4 = b[(i_0 + 1) * 128:(i_0 + 2) * 128, 0:128]
        tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    return output""",
)

_r(
    "2x2_red2",
    (2 * TILE, 2 * TILE),
    (2 * TILE, 2 * TILE),
    """\
def tiled_matmul(a, b):
    output = nkigym.ndarray((256, 256), dtype=np.float32)
    for i_0 in range(2):
        tensor_0 = a[0:128, i_0 * 128:(i_0 + 1) * 128]
        tensor_1 = b[0:128, 0:128]
        tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
        tensor_3 = a[128:256, i_0 * 128:(i_0 + 1) * 128]
        tensor_4 = b[128:256, 0:128]
        tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
        output[i_0 * 128:(i_0 + 1) * 128, 0:128] = tensor_2[0:128, 0:128]
        tensor_5 = a[0:128, i_0 * 128:(i_0 + 1) * 128]
        tensor_6 = b[0:128, 128:256]
        tensor_7 = nkigym.nc_matmul(tensor_5[0:128, 0:128], tensor_6[0:128, 0:128])
        tensor_8 = a[128:256, i_0 * 128:(i_0 + 1) * 128]
        tensor_9 = b[128:256, 128:256]
        tensor_7[0:128, 0:128] += nkigym.nc_matmul(tensor_8[0:128, 0:128], tensor_9[0:128, 0:128])
        output[i_0 * 128:(i_0 + 1) * 128, 128:256] = tensor_7[0:128, 0:128]
    return output""",
)
