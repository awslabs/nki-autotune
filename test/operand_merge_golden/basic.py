"""Golden programs -- basic and adjacent merge cases."""

import numpy as np

from nkigym.ir import GymProgram, GymStatement, TensorRef

BEFORE_NO_ADJACENT_LOADS = GymProgram(
    "tiled_no_adjacent_loads",
    ("a", "b"),
    (("a", (128, 128)), ("b", (128, 384))),
    (
        GymStatement("np_empty", (("dtype", "np.float32"),), TensorRef("output", (128, 384), ((0, 128), (0, 384)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 384), ((0, 128), (0, 128)))),),
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
                ("dst", TensorRef("output", (128, 384), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 384), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 384), ((0, 128), (256, 384)))),),
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
                ("dst", TensorRef("output", (128, 384), ((0, 128), (256, 384)))),
            ),
            TensorRef("output", (128, 384), ((0, 128), (256, 384))),
        ),
    ),
    "output",
    np.float32,
)


BEFORE_SINGLE_SUBGRAPH = GymProgram(
    "tiled_single_subgraph",
    ("a", "b"),
    (("a", (128, 128)), ("b", (128, 128))),
    (
        GymStatement("np_empty", (("dtype", "np.float32"),), TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
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


BEFORE_DIFFERENT_SOURCE_TENSORS = GymProgram(
    "tiled_different_source_tensors",
    ("a", "b", "c"),
    (("a", (128, 128)), ("b", (128, 128)), ("c", (128, 128))),
    (
        GymStatement("np_empty", (("dtype", "np.float32"),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
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
                ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("c", (128, 128), ((0, 128), (0, 128)))),),
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
                ("dst", TensorRef("output", (128, 256), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)


BEFORE_DIFFERENT_PARTITION_SLICES = GymProgram(
    "tiled_different_partition_slices",
    ("a", "b"),
    (("a", (256, 128)), ("b", (128, 128))),
    (
        GymStatement("np_empty", (("dtype", "np.float32"),), TensorRef("output", (256, 128), ((0, 256), (0, 128)))),
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
                ("dst", TensorRef("output", (256, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (256, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (256, 128), ((128, 256), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
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
                ("dst", TensorRef("output", (256, 128), ((128, 256), (0, 128)))),
            ),
            TensorRef("output", (256, 128), ((128, 256), (0, 128))),
        ),
    ),
    "output",
    np.float32,
)


BEFORE_ADJACENT_LOADS_2X = GymProgram(
    "tiled_adjacent_loads_2x",
    ("a", "b"),
    (("a", (128, 128)), ("b", (128, 256))),
    (
        GymStatement("np_empty", (("dtype", "np.float32"),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 256), ((0, 128), (0, 128)))),),
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
                ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 256), ((0, 128), (128, 256)))),),
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
                ("dst", TensorRef("output", (128, 256), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)


AFTER_ADJACENT_LOADS_2X_PARTIAL = GymProgram(
    "tiled_adjacent_loads_2x",
    ("a", "b"),
    (("a", (128, 128)), ("b", (128, 256))),
    (
        GymStatement("np_empty", (("dtype", "np.float32"),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 256), ((0, 128), (0, 256)))),),
            TensorRef("tensor_1", (128, 256), ((0, 128), (0, 256))),
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
                ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (0, 128))),
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
                ("dst", TensorRef("output", (128, 256), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)


AFTER_ADJACENT_LOADS_2X = GymProgram(
    "tiled_adjacent_loads_2x",
    ("a", "b"),
    (("a", (128, 128)), ("b", (128, 256))),
    (
        GymStatement("np_empty", (("dtype", "np.float32"),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 256), ((0, 128), (0, 256)))),),
            TensorRef("tensor_1", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            "nc_matmul",
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 256), ((0, 128), (0, 256)))),
            ),
            TensorRef("tensor_2", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_2", (128, 256), ((0, 128), (0, 256)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
    ),
    "output",
    np.float32,
)
