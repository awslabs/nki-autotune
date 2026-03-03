"""Golden programs -- additional no-merge corner cases."""

import numpy as np

from nkigym.ir import GymProgram, GymStatement, TensorRef
from nkigym.ops import ActivationOp, AllocateOp, LoadOp, MatmulOp, StoreOp


def _kw(shapes: dict) -> dict:
    """Create zero-filled kwargs from shape dict."""
    return {k: np.zeros(v, dtype=np.float32) for k, v in shapes.items()}


R = TensorRef

BEFORE_SINGLE_ACTIVATION = GymProgram(
    "tiled_single_activation",
    _kw({"a": (128, 128)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), R("output", (128, 128), ((0, 128), (0, 128)))),
        GymStatement(
            LoadOp,
            (("src", R("a", (128, 128), ((0, 128), (0, 128)))),),
            R("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            ActivationOp,
            (("data", R("tensor_0", (128, 128), ((0, 128), (0, 128)))), ("op", np.tanh)),
            R("tensor_1", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", R("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ("dst", R("output", (128, 128), ((0, 128), (0, 128)))),
            ),
            R("output", (128, 128), ((0, 128), (0, 128))),
        ),
    ),
    "output",
    np.float32,
)

BEFORE_HETEROGENEOUS_OPS = GymProgram(
    "tiled_heterogeneous_ops",
    _kw({"a": (128, 128), "b": (128, 128), "c": (128, 128)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), R("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            LoadOp,
            (("src", R("a", (128, 128), ((0, 128), (0, 128)))),),
            R("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", R("b", (128, 128), ((0, 128), (0, 128)))),),
            R("tensor_1", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", R("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", R("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            R("tensor_2", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", R("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ("dst", R("output", (128, 256), ((0, 128), (0, 128)))),
            ),
            R("output", (128, 256), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", R("c", (128, 128), ((0, 128), (0, 128)))),),
            R("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            ActivationOp,
            (("data", R("tensor_3", (128, 128), ((0, 128), (0, 128)))), ("op", np.tanh)),
            R("tensor_4", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", R("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                ("dst", R("output", (128, 256), ((0, 128), (128, 256)))),
            ),
            R("output", (128, 256), ((0, 128), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)

BEFORE_ALREADY_MERGED = GymProgram(
    "tiled_already_merged",
    _kw({"a": (128, 128), "b": (128, 256)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), R("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            LoadOp,
            (("src", R("a", (128, 128), ((0, 128), (0, 128)))),),
            R("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", R("b", (128, 256), ((0, 128), (0, 256)))),),
            R("tensor_1", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", R("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", R("tensor_1", (128, 256), ((0, 128), (0, 256)))),
            ),
            R("tensor_2", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", R("tensor_2", (128, 256), ((0, 128), (0, 256)))),
                ("dst", R("output", (128, 256), ((0, 128), (0, 256)))),
            ),
            R("output", (128, 256), ((0, 128), (0, 256))),
        ),
    ),
    "output",
    np.float32,
)
