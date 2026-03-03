"""Golden GymProgram constants for 2x2 matmul data reuse test cases."""

import numpy as np

from nkigym.ir import GymProgram, GymStatement, TensorRef
from nkigym.ops import AllocateOp, LoadOp, MatmulOp, StoreOp


def _kw(shapes: dict) -> dict:
    """Create zero-filled kwargs from shape dict."""
    return {k: np.zeros(v, dtype=np.float32) for k, v in shapes.items()}


_S = (128, 128)
_ID = ((0, 128), (0, 128))


def _ref(name: str) -> TensorRef:
    """Build a 128x128 identity-sliced TensorRef."""
    return TensorRef(name, _S, _ID)


BEFORE_MATMUL_2X2 = GymProgram(
    "before_matmul_2x2",
    _kw({"a": (128, 256), "b": (128, 256)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (256, 256), ((0, 256), (0, 256)))),
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
        GymStatement(LoadOp, (("src", TensorRef("a", _S, ((0, 128), (128, 256)))),), _ref("tensor_6")),
        GymStatement(LoadOp, (("src", TensorRef("b", _S, _ID)),), _ref("tensor_7")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_6")), ("moving", _ref("tensor_7"))), _ref("tensor_8")),
        GymStatement(
            StoreOp,
            (("src", _ref("tensor_8")), ("dst", TensorRef("output", _S, ((128, 256), (0, 128))))),
            TensorRef("output", _S, ((128, 256), (0, 128))),
        ),
        GymStatement(LoadOp, (("src", TensorRef("a", _S, ((0, 128), (128, 256)))),), _ref("tensor_9")),
        GymStatement(LoadOp, (("src", TensorRef("b", _S, ((0, 128), (128, 256)))),), _ref("tensor_10")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_9")), ("moving", _ref("tensor_10"))), _ref("tensor_11")),
        GymStatement(
            StoreOp,
            (("src", _ref("tensor_11")), ("dst", TensorRef("output", _S, ((128, 256), (128, 256))))),
            TensorRef("output", _S, ((128, 256), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)


AFTER_MATMUL_2X2_PARTIAL = GymProgram(
    "after_matmul_2x2_partial",
    _kw({"a": (128, 256), "b": (128, 256)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (256, 256), ((0, 256), (0, 256)))),
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
        GymStatement(LoadOp, (("src", TensorRef("a", _S, ((0, 128), (128, 256)))),), _ref("tensor_6")),
        GymStatement(LoadOp, (("src", TensorRef("b", _S, _ID)),), _ref("tensor_7")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_6")), ("moving", _ref("tensor_7"))), _ref("tensor_8")),
        GymStatement(
            StoreOp,
            (("src", _ref("tensor_8")), ("dst", TensorRef("output", _S, ((128, 256), (0, 128))))),
            TensorRef("output", _S, ((128, 256), (0, 128))),
        ),
        GymStatement(LoadOp, (("src", TensorRef("a", _S, ((0, 128), (128, 256)))),), _ref("tensor_9")),
        GymStatement(LoadOp, (("src", TensorRef("b", _S, ((0, 128), (128, 256)))),), _ref("tensor_10")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_9")), ("moving", _ref("tensor_10"))), _ref("tensor_11")),
        GymStatement(
            StoreOp,
            (("src", _ref("tensor_11")), ("dst", TensorRef("output", _S, ((128, 256), (128, 256))))),
            TensorRef("output", _S, ((128, 256), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)


AFTER_MATMUL_2X2_FULL = GymProgram(
    "after_matmul_2x2_full",
    _kw({"a": (128, 256), "b": (128, 256)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (256, 256), ((0, 256), (0, 256)))),
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
        GymStatement(LoadOp, (("src", TensorRef("a", _S, ((0, 128), (128, 256)))),), _ref("tensor_6")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_6")), ("moving", _ref("tensor_1"))), _ref("tensor_8")),
        GymStatement(
            StoreOp,
            (("src", _ref("tensor_8")), ("dst", TensorRef("output", _S, ((128, 256), (0, 128))))),
            TensorRef("output", _S, ((128, 256), (0, 128))),
        ),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_6")), ("moving", _ref("tensor_4"))), _ref("tensor_11")),
        GymStatement(
            StoreOp,
            (("src", _ref("tensor_11")), ("dst", TensorRef("output", _S, ((128, 256), (128, 256))))),
            TensorRef("output", _S, ((128, 256), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)
