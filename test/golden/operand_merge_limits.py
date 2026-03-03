"""Golden programs -- N-dimension limit merge cases."""

import numpy as np

from nkigym.ir import GymProgram, GymStatement, TensorRef
from nkigym.ops import AllocateOp, LoadOp, MatmulOp, StoreOp


def _kw(shapes: dict) -> dict:
    """Create zero-filled kwargs from shape dict."""
    return {k: np.zeros(v, dtype=np.float32) for k, v in shapes.items()}


BEFORE_MATMUL_N_AT_LIMIT = GymProgram(
    "tiled_matmul_n_at_limit",
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
            (("src", TensorRef("b", (128, 512), ((0, 128), (0, 256)))),),
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
                ("dst", TensorRef("output", (128, 512), ((0, 128), (0, 256)))),
            ),
            TensorRef("output", (128, 512), ((0, 128), (0, 256))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 512), ((0, 128), (256, 512)))),),
            TensorRef("tensor_3", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_3", (128, 256), ((0, 128), (0, 256)))),
            ),
            TensorRef("tensor_4", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_4", (128, 256), ((0, 128), (0, 256)))),
                ("dst", TensorRef("output", (128, 512), ((0, 128), (256, 512)))),
            ),
            TensorRef("output", (128, 512), ((0, 128), (256, 512))),
        ),
    ),
    "output",
    np.float32,
)

AFTER_N_AT_LIMIT = GymProgram(
    "tiled_matmul_n_at_limit",
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

BEFORE_MATMUL_EXCEEDS_N_LIMIT = GymProgram(
    "tiled_matmul_exceeds_n_limit",
    _kw({"a": (128, 128), "b": (128, 640)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 640), ((0, 128), (0, 640)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 640), ((0, 128), (0, 128)))),),
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
                ("dst", TensorRef("output", (128, 640), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 640), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 640), ((0, 128), (128, 256)))),),
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
                ("dst", TensorRef("output", (128, 640), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (128, 640), ((0, 128), (128, 256))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 640), ((0, 128), (256, 384)))),),
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
                ("dst", TensorRef("output", (128, 640), ((0, 128), (256, 384)))),
            ),
            TensorRef("output", (128, 640), ((0, 128), (256, 384))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 640), ((0, 128), (384, 512)))),),
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
                ("dst", TensorRef("output", (128, 640), ((0, 128), (384, 512)))),
            ),
            TensorRef("output", (128, 640), ((0, 128), (384, 512))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 640), ((0, 128), (512, 640)))),),
            TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 640), ((0, 128), (512, 640)))),
            ),
            TensorRef("output", (128, 640), ((0, 128), (512, 640))),
        ),
    ),
    "output",
    np.float32,
)

AFTER_EXCEEDS_N_LIMIT_PARTIAL = GymProgram(
    "tiled_matmul_exceeds_n_limit",
    _kw({"a": (128, 128), "b": (128, 640)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 640), ((0, 128), (0, 640)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 640), ((0, 128), (0, 640)))),),
            TensorRef("tensor_1", (128, 640), ((0, 128), (0, 640))),
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
                ("dst", TensorRef("output", (128, 640), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 640), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (128, 256)))),
            ),
            TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 640), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (128, 640), ((0, 128), (128, 256))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (256, 384)))),
            ),
            TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 640), ((0, 128), (256, 384)))),
            ),
            TensorRef("output", (128, 640), ((0, 128), (256, 384))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (384, 512)))),
            ),
            TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 640), ((0, 128), (384, 512)))),
            ),
            TensorRef("output", (128, 640), ((0, 128), (384, 512))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (512, 640)))),
            ),
            TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 640), ((0, 128), (512, 640)))),
            ),
            TensorRef("output", (128, 640), ((0, 128), (512, 640))),
        ),
    ),
    "output",
    np.float32,
)

AFTER_EXCEEDS_N_LIMIT = GymProgram(
    "tiled_matmul_exceeds_n_limit",
    _kw({"a": (128, 128), "b": (128, 640)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 640), ((0, 128), (0, 640)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 640), ((0, 128), (0, 640)))),),
            TensorRef("tensor_1", (128, 640), ((0, 128), (0, 640))),
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
                ("dst", TensorRef("output", (128, 640), ((0, 128), (0, 512)))),
            ),
            TensorRef("output", (128, 640), ((0, 128), (0, 512))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (512, 640)))),
            ),
            TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 640), ((0, 128), (512, 640)))),
            ),
            TensorRef("output", (128, 640), ((0, 128), (512, 640))),
        ),
    ),
    "output",
    np.float32,
)
