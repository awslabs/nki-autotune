"""Golden programs -- post-reuse merge cases."""

import numpy as np

from nkigym.ir import GymProgram, GymStatement, TensorRef
from nkigym.ops import AllocateOp, LoadOp, MatmulOp, StoreOp


def _kw(shapes: dict) -> dict:
    """Create zero-filled kwargs from shape dict."""
    return {k: np.zeros(v, dtype=np.float32) for k, v in shapes.items()}


BEFORE_MATMUL_POST_REUSE_1X2 = GymProgram(
    "tiled_matmul_post_reuse_1x2",
    _kw({"a": (128, 128), "b": (128, 256)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 256), ((0, 128), (0, 128)))),),
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
            StoreOp,
            (
                ("src", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 256), ((0, 128), (128, 256)))),),
            TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)

AFTER_POST_REUSE_1X2 = GymProgram(
    "tiled_matmul_post_reuse_1x2",
    _kw({"a": (128, 128), "b": (128, 256)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 256), ((0, 128), (0, 256)))),),
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
            StoreOp,
            (
                ("src", TensorRef("tensor_2", (128, 256), ((0, 128), (0, 256)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (0, 256))),
        ),
    ),
    "output",
    np.float32,
)

BEFORE_MATMUL_POST_REUSE_1X4 = GymProgram(
    "tiled_matmul_post_reuse_1x4",
    _kw({"a": (128, 128), "b": (128, 512)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 512), ((0, 128), (0, 512)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 512), ((0, 128), (0, 128)))),),
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
            StoreOp,
            (
                ("src", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 512), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 512), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 512), ((0, 128), (128, 256)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 512), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (128, 512), ((0, 128), (128, 256))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 512), ((0, 128), (256, 384)))),),
            TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 512), ((0, 128), (256, 384)))),
            ),
            TensorRef("output", (128, 512), ((0, 128), (256, 384))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 512), ((0, 128), (384, 512)))),),
            TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 512), ((0, 128), (384, 512)))),
            ),
            TensorRef("output", (128, 512), ((0, 128), (384, 512))),
        ),
    ),
    "output",
    np.float32,
)

AFTER_POST_REUSE_1X4 = GymProgram(
    "tiled_matmul_post_reuse_1x4",
    _kw({"a": (128, 128), "b": (128, 512)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 512), ((0, 128), (0, 512)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 512), ((0, 128), (0, 512)))),),
            TensorRef("tensor_1", (128, 512), ((0, 128), (0, 512))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 512), ((0, 128), (0, 512)))),
            ),
            TensorRef("tensor_2", (128, 512), ((0, 128), (0, 512))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_2", (128, 512), ((0, 128), (0, 512)))),
                ("dst", TensorRef("output", (128, 512), ((0, 128), (0, 512)))),
            ),
            TensorRef("output", (128, 512), ((0, 128), (0, 512))),
        ),
    ),
    "output",
    np.float32,
)

BEFORE_MATMUL_POST_REUSE_2X2 = GymProgram(
    "tiled_matmul_post_reuse_2x2",
    _kw({"a": (128, 256), "b": (128, 256)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (256, 256), ((0, 256), (0, 256)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 256), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 256), ((0, 128), (0, 128)))),),
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
            StoreOp,
            (
                ("src", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (256, 256), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (256, 256), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 256), ((0, 128), (128, 256)))),),
            TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (256, 256), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (256, 256), ((0, 128), (128, 256))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 256), ((0, 128), (128, 256)))),),
            TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (256, 256), ((128, 256), (0, 128)))),
            ),
            TensorRef("output", (256, 256), ((128, 256), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (256, 256), ((128, 256), (128, 256)))),
            ),
            TensorRef("output", (256, 256), ((128, 256), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)

AFTER_POST_REUSE_2X2_PARTIAL = GymProgram(
    "tiled_matmul_post_reuse_2x2",
    _kw({"a": (128, 256), "b": (128, 256)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (256, 256), ((0, 256), (0, 256)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 256), ((0, 128), (0, 256)))),),
            TensorRef("tensor_0", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 256), ((0, 128), (0, 256)))),),
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
            StoreOp,
            (
                ("src", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (256, 256), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (256, 256), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (128, 256)))),
            ),
            TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (256, 256), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (256, 256), ((0, 128), (128, 256))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (128, 256)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (256, 256), ((128, 256), (0, 128)))),
            ),
            TensorRef("output", (256, 256), ((128, 256), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (128, 256)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (128, 256)))),
            ),
            TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (256, 256), ((128, 256), (128, 256)))),
            ),
            TensorRef("output", (256, 256), ((128, 256), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)

AFTER_POST_REUSE_2X2 = GymProgram(
    "tiled_matmul_post_reuse_2x2",
    _kw({"a": (128, 256), "b": (128, 256)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (256, 256), ((0, 256), (0, 256)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 256), ((0, 128), (0, 256)))),),
            TensorRef("tensor_0", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 256), ((0, 128), (0, 256)))),),
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
            StoreOp,
            (
                ("src", TensorRef("tensor_2", (128, 256), ((0, 128), (0, 256)))),
                ("dst", TensorRef("output", (256, 256), ((0, 128), (0, 256)))),
            ),
            TensorRef("output", (256, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (128, 256)))),
                ("moving", TensorRef("tensor_1", (128, 256), ((0, 128), (0, 256)))),
            ),
            TensorRef("tensor_8", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_8", (128, 256), ((0, 128), (0, 256)))),
                ("dst", TensorRef("output", (256, 256), ((128, 256), (0, 256)))),
            ),
            TensorRef("output", (256, 256), ((128, 256), (0, 256))),
        ),
    ),
    "output",
    np.float32,
)
