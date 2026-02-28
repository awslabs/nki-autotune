"""Golden programs -- element-wise op merge cases."""

import numpy as np

from nkigym.ir import GymProgram, GymStatement, TensorRef

BEFORE_TENSOR_TENSOR_2X = GymProgram(
    "tiled_tensor_tensor_2x",
    ("a", "b"),
    (("a", (128, 256)), ("b", (128, 128))),
    (
        GymStatement("np_empty", (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 256), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "tensor_tensor",
            (
                ("data1", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("data2", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ("op", np.add),
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
            (("src", TensorRef("a", (128, 256), ((0, 128), (128, 256)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "tensor_tensor",
            (
                ("data1", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ("data2", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ("op", np.add),
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


AFTER_TENSOR_TENSOR_2X = GymProgram(
    "tiled_tensor_tensor_2x",
    ("a", "b"),
    (("a", (128, 256)), ("b", (128, 128))),
    (
        GymStatement("np_empty", (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 256), ((0, 128), (0, 256)))),),
            TensorRef("tensor_0", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "tensor_tensor",
            (
                ("data1", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("data2", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ("op", np.add),
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
            "tensor_tensor",
            (
                ("data1", TensorRef("tensor_0", (128, 128), ((0, 128), (128, 256)))),
                ("data2", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ("op", np.add),
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


BEFORE_TENSOR_TENSOR_DIFF_OPS = GymProgram(
    "tiled_tensor_tensor_diff_ops",
    ("a", "b"),
    (("a", (128, 256)), ("b", (128, 128))),
    (
        GymStatement("np_empty", (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 256), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "tensor_tensor",
            (
                ("data1", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("data2", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ("op", np.add),
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
            (("src", TensorRef("a", (128, 256), ((0, 128), (128, 256)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "tensor_tensor",
            (
                ("data1", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ("data2", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ("op", np.multiply),
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


AFTER_TENSOR_TENSOR_DIFF_OPS = GymProgram(
    "tiled_tensor_tensor_diff_ops",
    ("a", "b"),
    (("a", (128, 256)), ("b", (128, 128))),
    (
        GymStatement("np_empty", (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 256), ((0, 128), (0, 256)))),),
            TensorRef("tensor_0", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "tensor_tensor",
            (
                ("data1", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("data2", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ("op", np.add),
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
            "tensor_tensor",
            (
                ("data1", TensorRef("tensor_0", (128, 128), ((0, 128), (128, 256)))),
                ("data2", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ("op", np.multiply),
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


BEFORE_ACTIVATION_2X = GymProgram(
    "tiled_activation_2x",
    ("a",),
    (("a", (128, 256)),),
    (
        GymStatement("np_empty", (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 256), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "activation",
            (("data", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))), ("op", np.tanh)),
            TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 256), ((0, 128), (128, 256)))),),
            TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "activation",
            (("data", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))), ("op", np.tanh)),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)


AFTER_ACTIVATION_2X = GymProgram(
    "tiled_activation_2x",
    ("a",),
    (("a", (128, 256)),),
    (
        GymStatement("np_empty", (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 256), ((0, 128), (0, 256)))),),
            TensorRef("tensor_0", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            "activation",
            (("data", TensorRef("tensor_0", (128, 256), ((0, 128), (0, 256)))), ("op", np.tanh)),
            TensorRef("tensor_1", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_1", (128, 256), ((0, 128), (0, 256)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (0, 256))),
        ),
    ),
    "output",
    np.float32,
)


BEFORE_TENSOR_SCALAR_2X = GymProgram(
    "tiled_tensor_scalar_2x",
    ("a",),
    (("a", (128, 256)),),
    (
        GymStatement("np_empty", (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 256), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "tensor_scalar",
            (
                ("data", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("op0", np.multiply),
                ("operand0", 2.0),
            ),
            TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 256), ((0, 128), (128, 256)))),),
            TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "tensor_scalar",
            (
                ("data", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ("op0", np.multiply),
                ("operand0", 2.0),
            ),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)


AFTER_TENSOR_SCALAR_2X = GymProgram(
    "tiled_tensor_scalar_2x",
    ("a",),
    (("a", (128, 256)),),
    (
        GymStatement("np_empty", (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            "np_slice",
            (("src", TensorRef("a", (128, 256), ((0, 128), (0, 256)))),),
            TensorRef("tensor_0", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            "tensor_scalar",
            (
                ("data", TensorRef("tensor_0", (128, 256), ((0, 128), (0, 256)))),
                ("op0", np.multiply),
                ("operand0", 2.0),
            ),
            TensorRef("tensor_1", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            "np_store",
            (
                ("src", TensorRef("tensor_1", (128, 256), ((0, 128), (0, 256)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (0, 256))),
        ),
    ),
    "output",
    np.float32,
)
