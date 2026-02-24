"""Golden programs -- additional no-merge corner cases."""

import numpy as np

from nkigym.ir import GymProgram, GymStatement, TensorRef

R = TensorRef

BEFORE_SINGLE_ACTIVATION = GymProgram(
    "tiled_single_activation",
    ("a",),
    (("a", (128, 128)),),
    (
        GymStatement("np_empty", (("dtype", "np.float32"),), R("output", (128, 128), ((0, 128), (0, 128)))),
        GymStatement(
            "np_slice",
            (("src", R("a", (128, 128), ((0, 128), (0, 128)))),),
            R("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "activation",
            (("data", R("tensor_0", (128, 128), ((0, 128), (0, 128)))), ("op", "np.tanh")),
            R("tensor_1", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_store",
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
    ("a", "b", "c"),
    (("a", (128, 128)), ("b", (128, 128)), ("c", (128, 128))),
    (
        GymStatement("np_empty", (("dtype", "np.float32"),), R("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            "np_slice",
            (("src", R("a", (128, 128), ((0, 128), (0, 128)))),),
            R("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", R("b", (128, 128), ((0, 128), (0, 128)))),),
            R("tensor_1", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "nc_matmul",
            (
                ("stationary", R("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", R("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            R("tensor_2", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", R("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ("dst", R("output", (128, 256), ((0, 128), (0, 128)))),
            ),
            R("output", (128, 256), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", R("c", (128, 128), ((0, 128), (0, 128)))),),
            R("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "activation",
            (("data", R("tensor_3", (128, 128), ((0, 128), (0, 128)))), ("op", "np.tanh")),
            R("tensor_4", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_store",
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
    ("a", "b"),
    (("a", (128, 128)), ("b", (128, 256))),
    (
        GymStatement("np_empty", (("dtype", "np.float32"),), R("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            "np_slice",
            (("src", R("a", (128, 128), ((0, 128), (0, 128)))),),
            R("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", R("b", (128, 256), ((0, 128), (0, 256)))),),
            R("tensor_1", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            "nc_matmul",
            (
                ("stationary", R("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", R("tensor_1", (128, 256), ((0, 128), (0, 256)))),
            ),
            R("tensor_2", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            "np_store",
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
