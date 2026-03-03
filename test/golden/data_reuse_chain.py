"""Golden GymProgram constants for double matmul (chained) data reuse test cases."""

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


_KW = _kw({"a": (128, 128), "b": (128, 256), "c": (128, 128)})
_ALLOC = GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (256, 128), ((0, 256), (0, 128))))
_OUT0 = TensorRef("output", _S, _ID)
_OUT1 = TensorRef("output", _S, ((128, 256), (0, 128)))
_A = TensorRef("a", _S, _ID)
_B0 = TensorRef("b", _S, _ID)
_B1 = TensorRef("b", _S, ((0, 128), (128, 256)))
_C = TensorRef("c", _S, _ID)


BEFORE_DOUBLE_MATMUL_2X1 = GymProgram(
    "before_double_matmul_2x1",
    _KW,
    (
        _ALLOC,
        GymStatement(LoadOp, (("src", _A),), _ref("tensor_0")),
        GymStatement(LoadOp, (("src", _B0),), _ref("tensor_1")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_0")), ("moving", _ref("tensor_1"))), _ref("tensor_2")),
        GymStatement(LoadOp, (("src", _C),), _ref("tensor_3")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_2")), ("moving", _ref("tensor_3"))), _ref("tensor_4")),
        GymStatement(StoreOp, (("src", _ref("tensor_4")), ("dst", _OUT0)), _OUT0),
        GymStatement(LoadOp, (("src", _A),), _ref("tensor_5")),
        GymStatement(LoadOp, (("src", _B1),), _ref("tensor_6")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_5")), ("moving", _ref("tensor_6"))), _ref("tensor_7")),
        GymStatement(LoadOp, (("src", _C),), _ref("tensor_8")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_7")), ("moving", _ref("tensor_8"))), _ref("tensor_9")),
        GymStatement(StoreOp, (("src", _ref("tensor_9")), ("dst", _OUT1)), _OUT1),
    ),
    "output",
    np.float32,
)


AFTER_DOUBLE_MATMUL_2X1 = GymProgram(
    "after_double_matmul_2x1",
    _KW,
    (
        _ALLOC,
        GymStatement(LoadOp, (("src", _A),), _ref("tensor_0")),
        GymStatement(LoadOp, (("src", _B0),), _ref("tensor_1")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_0")), ("moving", _ref("tensor_1"))), _ref("tensor_2")),
        GymStatement(LoadOp, (("src", _C),), _ref("tensor_3")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_2")), ("moving", _ref("tensor_3"))), _ref("tensor_4")),
        GymStatement(StoreOp, (("src", _ref("tensor_4")), ("dst", _OUT0)), _OUT0),
        GymStatement(LoadOp, (("src", _B1),), _ref("tensor_6")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_0")), ("moving", _ref("tensor_6"))), _ref("tensor_7")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_7")), ("moving", _ref("tensor_3"))), _ref("tensor_9")),
        GymStatement(StoreOp, (("src", _ref("tensor_9")), ("dst", _OUT1)), _OUT1),
    ),
    "output",
    np.float32,
)
