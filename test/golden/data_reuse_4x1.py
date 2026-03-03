"""Golden GymProgram constants for 4x1 matmul data reuse test cases."""

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


_KW_4X1 = _kw({"a": (128, 512), "b": (128, 128)})
_OUT_4X1 = TensorRef("output", (512, 128), ((0, 512), (0, 128)))
_ALLOC_4X1 = GymStatement(AllocateOp, (("dtype", np.float32),), _OUT_4X1)

_A0 = TensorRef("a", _S, _ID)
_A1 = TensorRef("a", _S, ((0, 128), (128, 256)))
_A2 = TensorRef("a", _S, ((0, 128), (256, 384)))
_A3 = TensorRef("a", _S, ((0, 128), (384, 512)))
_B0 = TensorRef("b", _S, _ID)
_OUT0 = TensorRef("output", _S, _ID)
_OUT1 = TensorRef("output", _S, ((128, 256), (0, 128)))
_OUT2 = TensorRef("output", _S, ((256, 384), (0, 128)))
_OUT3 = TensorRef("output", _S, ((384, 512), (0, 128)))


BEFORE_MATMUL_4X1 = GymProgram(
    "before_matmul_4x1",
    _KW_4X1,
    (
        _ALLOC_4X1,
        GymStatement(LoadOp, (("src", _A0),), _ref("tensor_0")),
        GymStatement(LoadOp, (("src", _B0),), _ref("tensor_1")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_0")), ("moving", _ref("tensor_1"))), _ref("tensor_2")),
        GymStatement(StoreOp, (("src", _ref("tensor_2")), ("dst", _OUT0)), _OUT0),
        GymStatement(LoadOp, (("src", _A1),), _ref("tensor_3")),
        GymStatement(LoadOp, (("src", _B0),), _ref("tensor_4")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_3")), ("moving", _ref("tensor_4"))), _ref("tensor_5")),
        GymStatement(StoreOp, (("src", _ref("tensor_5")), ("dst", _OUT1)), _OUT1),
        GymStatement(LoadOp, (("src", _A2),), _ref("tensor_6")),
        GymStatement(LoadOp, (("src", _B0),), _ref("tensor_7")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_6")), ("moving", _ref("tensor_7"))), _ref("tensor_8")),
        GymStatement(StoreOp, (("src", _ref("tensor_8")), ("dst", _OUT2)), _OUT2),
        GymStatement(LoadOp, (("src", _A3),), _ref("tensor_9")),
        GymStatement(LoadOp, (("src", _B0),), _ref("tensor_10")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_9")), ("moving", _ref("tensor_10"))), _ref("tensor_11")),
        GymStatement(StoreOp, (("src", _ref("tensor_11")), ("dst", _OUT3)), _OUT3),
    ),
    "output",
    np.float32,
)


AFTER_MATMUL_4X1_PARTIAL = GymProgram(
    "after_matmul_4x1_partial",
    _KW_4X1,
    (
        _ALLOC_4X1,
        GymStatement(LoadOp, (("src", _A0),), _ref("tensor_0")),
        GymStatement(LoadOp, (("src", _B0),), _ref("tensor_1")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_0")), ("moving", _ref("tensor_1"))), _ref("tensor_2")),
        GymStatement(StoreOp, (("src", _ref("tensor_2")), ("dst", _OUT0)), _OUT0),
        GymStatement(LoadOp, (("src", _A1),), _ref("tensor_3")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_3")), ("moving", _ref("tensor_1"))), _ref("tensor_5")),
        GymStatement(StoreOp, (("src", _ref("tensor_5")), ("dst", _OUT1)), _OUT1),
        GymStatement(LoadOp, (("src", _A2),), _ref("tensor_6")),
        GymStatement(LoadOp, (("src", _B0),), _ref("tensor_7")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_6")), ("moving", _ref("tensor_7"))), _ref("tensor_8")),
        GymStatement(StoreOp, (("src", _ref("tensor_8")), ("dst", _OUT2)), _OUT2),
        GymStatement(LoadOp, (("src", _A3),), _ref("tensor_9")),
        GymStatement(LoadOp, (("src", _B0),), _ref("tensor_10")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_9")), ("moving", _ref("tensor_10"))), _ref("tensor_11")),
        GymStatement(StoreOp, (("src", _ref("tensor_11")), ("dst", _OUT3)), _OUT3),
    ),
    "output",
    np.float32,
)


AFTER_MATMUL_4X1_FULL = GymProgram(
    "after_matmul_4x1_full",
    _KW_4X1,
    (
        _ALLOC_4X1,
        GymStatement(LoadOp, (("src", _A0),), _ref("tensor_0")),
        GymStatement(LoadOp, (("src", _B0),), _ref("tensor_1")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_0")), ("moving", _ref("tensor_1"))), _ref("tensor_2")),
        GymStatement(StoreOp, (("src", _ref("tensor_2")), ("dst", _OUT0)), _OUT0),
        GymStatement(LoadOp, (("src", _A1),), _ref("tensor_3")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_3")), ("moving", _ref("tensor_1"))), _ref("tensor_5")),
        GymStatement(StoreOp, (("src", _ref("tensor_5")), ("dst", _OUT1)), _OUT1),
        GymStatement(LoadOp, (("src", _A2),), _ref("tensor_6")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_6")), ("moving", _ref("tensor_1"))), _ref("tensor_8")),
        GymStatement(StoreOp, (("src", _ref("tensor_8")), ("dst", _OUT2)), _OUT2),
        GymStatement(LoadOp, (("src", _A3),), _ref("tensor_9")),
        GymStatement(MatmulOp, (("stationary", _ref("tensor_9")), ("moving", _ref("tensor_1"))), _ref("tensor_11")),
        GymStatement(StoreOp, (("src", _ref("tensor_11")), ("dst", _OUT3)), _OUT3),
    ),
    "output",
    np.float32,
)
