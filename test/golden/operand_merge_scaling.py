"""Golden programs -- 4x adjacent merge cases."""

import numpy as np

from nkigym.ir import GymProgram, GymStatement, TensorRef

BEFORE_ADJACENT_4X = GymProgram(
    "tiled_adjacent_4x",
    ("a", "b"),
    (("a", (128, 128)), ("b", (128, 512))),
    (
        GymStatement("np_empty", (("dtype", np.float32),), TensorRef("output", (128, 512), ((0, 128), (0, 512)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 512), ((0, 128), (0, 128)))),),
            TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "nc_matmul",
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 512), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 512), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 512), ((0, 128), (128, 256)))),),
            TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "nc_matmul",
            (
                ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 512), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (128, 512), ((0, 128), (128, 256))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 512), ((0, 128), (256, 384)))),),
            TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "nc_matmul",
            (
                ("stationary", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 512), ((0, 128), (256, 384)))),
            ),
            TensorRef("output", (128, 512), ((0, 128), (256, 384))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 512), ((0, 128), (384, 512)))),),
            TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "nc_matmul",
            (
                ("stationary", TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 512), ((0, 128), (384, 512)))),
            ),
            TensorRef("output", (128, 512), ((0, 128), (384, 512))),
        ),
    ),
    "output",
    np.float32,
)


AFTER_ADJACENT_4X_PARTIAL = GymProgram(
    "tiled_adjacent_4x",
    ("a", "b"),
    (("a", (128, 128)), ("b", (128, 512))),
    (
        GymStatement("np_empty", (("dtype", np.float32),), TensorRef("output", (128, 512), ((0, 128), (0, 512)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 512), ((0, 128), (0, 512)))),),
            TensorRef("tensor_1", (128, 512), ((0, 128), (0, 512))),
        ),
        GymStatement(
            "nc_matmul",
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 512), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 512), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "nc_matmul",
            (
                ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (128, 256)))),
            ),
            TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 512), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (128, 512), ((0, 128), (128, 256))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "nc_matmul",
            (
                ("stationary", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (256, 384)))),
            ),
            TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 512), ((0, 128), (256, 384)))),
            ),
            TensorRef("output", (128, 512), ((0, 128), (256, 384))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "nc_matmul",
            (
                ("stationary", TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (384, 512)))),
            ),
            TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 512), ((0, 128), (384, 512)))),
            ),
            TensorRef("output", (128, 512), ((0, 128), (384, 512))),
        ),
    ),
    "output",
    np.float32,
)


AFTER_ADJACENT_4X = GymProgram(
    "tiled_adjacent_4x",
    ("a", "b"),
    (("a", (128, 128)), ("b", (128, 512))),
    (
        GymStatement("np_empty", (("dtype", np.float32),), TensorRef("output", (128, 512), ((0, 128), (0, 512)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 512), ((0, 128), (0, 512)))),),
            TensorRef("tensor_1", (128, 512), ((0, 128), (0, 512))),
        ),
        GymStatement(
            "nc_matmul",
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 512), ((0, 128), (0, 512)))),
            ),
            TensorRef("tensor_2", (128, 512), ((0, 128), (0, 512))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_2", (128, 512), ((0, 128), (0, 512)))),
                ("dst", TensorRef("output", (128, 512), ((0, 128), (0, 512)))),
            ),
            TensorRef("output", (128, 512), ((0, 128), (0, 512))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
        ),
    ),
    "output",
    np.float32,
)
