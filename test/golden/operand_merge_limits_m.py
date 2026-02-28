"""Golden programs -- M-dimension limit merge cases."""

import numpy as np

from nkigym.ir import GymProgram, GymStatement, TensorRef

BEFORE_MATMUL_M_DIM_MERGE = GymProgram(
    "tiled_matmul_m_dim_merge",
    ("a", "b"),
    (("a", (256, 128)), ("b", (128, 128))),
    (
        GymStatement("np_empty", (("dtype", np.float32),), TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (256, 128), ((0, 128), (0, 64)))),),
            TensorRef("tensor_0", (128, 64), ((0, 128), (0, 64))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "nc_matmul",
            (
                ("stationary", TensorRef("tensor_0", (128, 64), ((0, 128), (0, 64)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_2", (64, 128), ((0, 64), (0, 128))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_2", (64, 128), ((0, 64), (0, 128)))),
                ("dst", TensorRef("output", (128, 128), ((0, 64), (0, 128)))),
            ),
            TensorRef("output", (128, 128), ((0, 64), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (256, 128), ((0, 128), (64, 128)))),),
            TensorRef("tensor_3", (128, 64), ((0, 128), (0, 64))),
        ),
        GymStatement(
            "nc_matmul",
            (
                ("stationary", TensorRef("tensor_3", (128, 64), ((0, 128), (0, 64)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_4", (64, 128), ((0, 64), (0, 128))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_4", (64, 128), ((0, 64), (0, 128)))),
                ("dst", TensorRef("output", (128, 128), ((64, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 128), ((64, 128), (0, 128))),
        ),
    ),
    "output",
    np.float32,
)


AFTER_M_DIM_MERGE = GymProgram(
    "tiled_matmul_m_dim_merge",
    ("a", "b"),
    (("a", (256, 128)), ("b", (128, 128))),
    (
        GymStatement("np_empty", (("dtype", np.float32),), TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (256, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
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
                ("dst", TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 128), ((0, 128), (0, 128))),
        ),
    ),
    "output",
    np.float32,
)


BEFORE_MATMUL_M_EXCEEDS_LIMIT = GymProgram(
    "tiled_matmul_m_exceeds_limit",
    ("a", "b"),
    (("a", (128, 192)), ("b", (128, 128))),
    (
        GymStatement("np_empty", (("dtype", np.float32),), TensorRef("output", (192, 128), ((0, 192), (0, 128)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 192), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
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
                ("dst", TensorRef("output", (192, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (192, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 192), ((0, 128), (128, 192)))),),
            TensorRef("tensor_3", (128, 64), ((0, 128), (0, 64))),
        ),
        GymStatement(
            "nc_matmul",
            (
                ("stationary", TensorRef("tensor_3", (128, 64), ((0, 128), (0, 64)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_4", (64, 128), ((0, 64), (0, 128))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_4", (64, 128), ((0, 64), (0, 128)))),
                ("dst", TensorRef("output", (192, 128), ((128, 192), (0, 128)))),
            ),
            TensorRef("output", (192, 128), ((128, 192), (0, 128))),
        ),
    ),
    "output",
    np.float32,
)


AFTER_M_EXCEEDS_LIMIT = GymProgram(
    "tiled_matmul_m_exceeds_limit",
    ("a", "b"),
    (("a", (128, 192)), ("b", (128, 128))),
    (
        GymStatement("np_empty", (("dtype", np.float32),), TensorRef("output", (192, 128), ((0, 192), (0, 128)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 192), ((0, 128), (0, 192)))),),
            TensorRef("tensor_0", (128, 192), ((0, 128), (0, 192))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
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
                ("dst", TensorRef("output", (192, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (192, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "nc_matmul",
            (
                ("stationary", TensorRef("tensor_0", (128, 64), ((0, 128), (128, 192)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_4", (64, 128), ((0, 64), (0, 128))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_4", (64, 128), ((0, 64), (0, 128)))),
                ("dst", TensorRef("output", (192, 128), ((128, 192), (0, 128)))),
            ),
            TensorRef("output", (192, 128), ((128, 192), (0, 128))),
        ),
    ),
    "output",
    np.float32,
)
