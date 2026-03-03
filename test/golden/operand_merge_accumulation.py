"""Golden programs -- accumulation merge cases."""

import numpy as np

from nkigym.ir import GymProgram, GymStatement, TensorRef
from nkigym.ops import AllocateOp, LoadOp, MatmulOp, StoreOp


def _kw(shapes: dict) -> dict:
    """Create zero-filled kwargs from shape dict."""
    return {k: np.zeros(v, dtype=np.float32) for k, v in shapes.items()}


ACCUMULATION_BLOCKS_PROGRAM = GymProgram(
    "accum_fn",
    _kw({"a": (256, 128), "b": (256, 256)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (256, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (256, 256), ((0, 128), (0, 128)))),),
            TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (256, 128), ((128, 256), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (256, 256), ((128, 256), (0, 128)))),),
            TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                ("acc", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (256, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (256, 256), ((0, 128), (128, 256)))),),
            TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (256, 128), ((128, 256), (0, 128)))),),
            TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (256, 256), ((128, 256), (128, 256)))),),
            TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128)))),
                ("acc", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)

ACCUMULATION_BLOCKS_AFTER_1_MERGE = GymProgram(
    "accum_fn",
    _kw({"a": (256, 128), "b": (256, 256)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (256, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (256, 256), ((0, 128), (0, 256)))),),
            TensorRef("tensor_1", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (256, 128), ((128, 256), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (256, 256), ((128, 256), (0, 128)))),),
            TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                ("acc", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (256, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (128, 256)))),
            ),
            TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (256, 128), ((128, 256), (0, 128)))),),
            TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (256, 256), ((128, 256), (128, 256)))),),
            TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128)))),
                ("acc", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)

ACCUMULATION_BLOCKS_AFTER_2_MERGES = GymProgram(
    "accum_fn",
    _kw({"a": (256, 128), "b": (256, 256)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (256, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (256, 256), ((0, 128), (0, 256)))),),
            TensorRef("tensor_1", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (256, 128), ((128, 256), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (256, 256), ((128, 256), (0, 256)))),),
            TensorRef("tensor_4", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                ("acc", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (256, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (128, 256)))),
            ),
            TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (256, 128), ((128, 256), (0, 128)))),),
            TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (128, 256)))),
                ("acc", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)

ACCUMULATION_BLOCKS_AFTER_3_MERGES = GymProgram(
    "accum_fn",
    _kw({"a": (256, 128), "b": (256, 256)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (256, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (256, 256), ((0, 128), (0, 256)))),),
            TensorRef("tensor_1", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 256), ((0, 128), (0, 256)))),
            ),
            TensorRef("tensor_2", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (256, 128), ((128, 256), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (256, 256), ((128, 256), (0, 256)))),),
            TensorRef("tensor_4", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                ("acc", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (256, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (256, 128), ((128, 256), (0, 128)))),),
            TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (128, 256)))),
                ("acc", TensorRef("tensor_2", (128, 128), ((0, 128), (128, 256)))),
            ),
            TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)
