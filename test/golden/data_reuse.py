"""Golden GymProgram constants for data reuse analysis and transform tests.

Small cases (1x1, 2x1, 1x2) are defined inline. Large cases are imported
from split modules to stay under the 500-line file limit.
"""

from typing import NamedTuple

import numpy as np
from golden.data_reuse_2x2 import AFTER_MATMUL_2X2_FULL, AFTER_MATMUL_2X2_PARTIAL, BEFORE_MATMUL_2X2
from golden.data_reuse_4x1 import AFTER_MATMUL_4X1_FULL, AFTER_MATMUL_4X1_PARTIAL, BEFORE_MATMUL_4X1
from golden.data_reuse_chain import AFTER_DOUBLE_MATMUL_2X1, BEFORE_DOUBLE_MATMUL_2X1

from nkigym.ir import GymProgram, GymStatement, TensorRef
from nkigym.ops import AllocateOp, LoadOp, MatmulOp, StoreOp


class DataReuseCase(NamedTuple):
    """A single data reuse test case.

    Attributes:
        id: Test case identifier for pytest parametrize.
        before: Pre-merge GymProgram.
        expected_pairs: Exact reuse pairs returned by analyze_ir.
        merge_count: Number of iterative merge passes to apply.
        after: Expected post-merge GymProgram.
    """

    id: str
    before: GymProgram
    expected_pairs: list[tuple[str, str]]
    merge_count: int
    after: GymProgram


def _kw(shapes: dict) -> dict:
    """Create zero-filled kwargs from shape dict."""
    return {k: np.zeros(v, dtype=np.float32) for k, v in shapes.items()}


_S = (128, 128)
_ID = ((0, 128), (0, 128))


def _ref(name: str) -> TensorRef:
    """Build a 128x128 identity-sliced TensorRef."""
    return TensorRef(name, _S, _ID)


BEFORE_MATMUL_1X1 = GymProgram(
    "before_matmul_1x1",
    _kw({"a": (128, 128), "b": (128, 128)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", _S, _ID)),
        GymStatement(LoadOp, (("src", TensorRef("a", _S, _ID)),), _ref("tensor_0")),
        GymStatement(LoadOp, (("src", TensorRef("b", _S, _ID)),), _ref("tensor_1")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_0")), ("moving", _ref("tensor_1"))), _ref("tensor_2")),
        GymStatement(
            StoreOp, (("src", _ref("tensor_2")), ("dst", TensorRef("output", _S, _ID))), TensorRef("output", _S, _ID)
        ),
    ),
    "output",
    np.float32,
)


_OUT_2X1 = TensorRef("output", (256, 128), ((0, 256), (0, 128)))

BEFORE_MATMUL_2X1 = GymProgram(
    "before_matmul_2x1",
    _kw({"a": (128, 256), "b": (128, 128)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), _OUT_2X1),
        GymStatement(LoadOp, (("src", TensorRef("a", _S, _ID)),), _ref("tensor_0")),
        GymStatement(LoadOp, (("src", TensorRef("b", _S, _ID)),), _ref("tensor_1")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_0")), ("moving", _ref("tensor_1"))), _ref("tensor_2")),
        GymStatement(
            StoreOp, (("src", _ref("tensor_2")), ("dst", TensorRef("output", _S, _ID))), TensorRef("output", _S, _ID)
        ),
        GymStatement(LoadOp, (("src", TensorRef("a", _S, ((0, 128), (128, 256)))),), _ref("tensor_3")),
        GymStatement(LoadOp, (("src", TensorRef("b", _S, _ID)),), _ref("tensor_4")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_3")), ("moving", _ref("tensor_4"))), _ref("tensor_5")),
        GymStatement(
            StoreOp,
            (("src", _ref("tensor_5")), ("dst", TensorRef("output", _S, ((128, 256), (0, 128))))),
            TensorRef("output", _S, ((128, 256), (0, 128))),
        ),
    ),
    "output",
    np.float32,
)

AFTER_MATMUL_2X1 = GymProgram(
    "after_matmul_2x1",
    _kw({"a": (128, 256), "b": (128, 128)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), _OUT_2X1),
        GymStatement(LoadOp, (("src", TensorRef("a", _S, _ID)),), _ref("tensor_0")),
        GymStatement(LoadOp, (("src", TensorRef("b", _S, _ID)),), _ref("tensor_1")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_0")), ("moving", _ref("tensor_1"))), _ref("tensor_2")),
        GymStatement(
            StoreOp, (("src", _ref("tensor_2")), ("dst", TensorRef("output", _S, _ID))), TensorRef("output", _S, _ID)
        ),
        GymStatement(LoadOp, (("src", TensorRef("a", _S, ((0, 128), (128, 256)))),), _ref("tensor_3")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_3")), ("moving", _ref("tensor_1"))), _ref("tensor_5")),
        GymStatement(
            StoreOp,
            (("src", _ref("tensor_5")), ("dst", TensorRef("output", _S, ((128, 256), (0, 128))))),
            TensorRef("output", _S, ((128, 256), (0, 128))),
        ),
    ),
    "output",
    np.float32,
)


_OUT_1X2 = TensorRef("output", (128, 256), ((0, 128), (0, 256)))

BEFORE_MATMUL_1X2 = GymProgram(
    "before_matmul_1x2",
    _kw({"a": (128, 128), "b": (128, 256)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), _OUT_1X2),
        GymStatement(LoadOp, (("src", TensorRef("a", _S, _ID)),), _ref("tensor_0")),
        GymStatement(LoadOp, (("src", TensorRef("b", _S, _ID)),), _ref("tensor_1")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_0")), ("moving", _ref("tensor_1"))), _ref("tensor_2")),
        GymStatement(
            StoreOp, (("src", _ref("tensor_2")), ("dst", TensorRef("output", _S, _ID))), TensorRef("output", _S, _ID)
        ),
        GymStatement(LoadOp, (("src", TensorRef("a", _S, _ID)),), _ref("tensor_3")),
        GymStatement(LoadOp, (("src", TensorRef("b", _S, ((0, 128), (128, 256)))),), _ref("tensor_4")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_3")), ("moving", _ref("tensor_4"))), _ref("tensor_5")),
        GymStatement(
            StoreOp,
            (("src", _ref("tensor_5")), ("dst", TensorRef("output", _S, ((0, 128), (128, 256))))),
            TensorRef("output", _S, ((0, 128), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)

AFTER_MATMUL_1X2 = GymProgram(
    "after_matmul_1x2",
    _kw({"a": (128, 128), "b": (128, 256)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), _OUT_1X2),
        GymStatement(LoadOp, (("src", TensorRef("a", _S, _ID)),), _ref("tensor_0")),
        GymStatement(LoadOp, (("src", TensorRef("b", _S, _ID)),), _ref("tensor_1")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_0")), ("moving", _ref("tensor_1"))), _ref("tensor_2")),
        GymStatement(
            StoreOp, (("src", _ref("tensor_2")), ("dst", TensorRef("output", _S, _ID))), TensorRef("output", _S, _ID)
        ),
        GymStatement(LoadOp, (("src", TensorRef("b", _S, ((0, 128), (128, 256)))),), _ref("tensor_4")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_0")), ("moving", _ref("tensor_4"))), _ref("tensor_5")),
        GymStatement(
            StoreOp,
            (("src", _ref("tensor_5")), ("dst", TensorRef("output", _S, ((0, 128), (128, 256))))),
            TensorRef("output", _S, ((0, 128), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)


CASES: list[DataReuseCase] = [
    DataReuseCase("matmul_1x1_no_reuse", BEFORE_MATMUL_1X1, [], 0, BEFORE_MATMUL_1X1),
    DataReuseCase("matmul_2x1_merge_b", BEFORE_MATMUL_2X1, [("tensor_1", "tensor_4")], 1, AFTER_MATMUL_2X1),
    DataReuseCase("matmul_1x2_merge_a", BEFORE_MATMUL_1X2, [("tensor_0", "tensor_3")], 1, AFTER_MATMUL_1X2),
    DataReuseCase(
        "matmul_2x2_partial",
        BEFORE_MATMUL_2X2,
        [("tensor_0", "tensor_3"), ("tensor_1", "tensor_7"), ("tensor_4", "tensor_10"), ("tensor_6", "tensor_9")],
        1,
        AFTER_MATMUL_2X2_PARTIAL,
    ),
    DataReuseCase(
        "matmul_2x2_full",
        BEFORE_MATMUL_2X2,
        [("tensor_0", "tensor_3"), ("tensor_1", "tensor_7"), ("tensor_4", "tensor_10"), ("tensor_6", "tensor_9")],
        4,
        AFTER_MATMUL_2X2_FULL,
    ),
    DataReuseCase(
        "matmul_4x1_partial",
        BEFORE_MATMUL_4X1,
        [
            ("tensor_1", "tensor_4"),
            ("tensor_1", "tensor_7"),
            ("tensor_1", "tensor_10"),
            ("tensor_4", "tensor_7"),
            ("tensor_4", "tensor_10"),
            ("tensor_7", "tensor_10"),
        ],
        1,
        AFTER_MATMUL_4X1_PARTIAL,
    ),
    DataReuseCase(
        "matmul_4x1_full",
        BEFORE_MATMUL_4X1,
        [
            ("tensor_1", "tensor_4"),
            ("tensor_1", "tensor_7"),
            ("tensor_1", "tensor_10"),
            ("tensor_4", "tensor_7"),
            ("tensor_4", "tensor_10"),
            ("tensor_7", "tensor_10"),
        ],
        3,
        AFTER_MATMUL_4X1_FULL,
    ),
    DataReuseCase(
        "double_matmul_2x1_full",
        BEFORE_DOUBLE_MATMUL_2X1,
        [("tensor_0", "tensor_5"), ("tensor_3", "tensor_8")],
        2,
        AFTER_DOUBLE_MATMUL_2X1,
    ),
]
