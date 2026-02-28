"""Golden test data for IR tests: expected GymProgram outputs and parametrized cases."""

import numpy as np
import pytest

from nkigym.ir import GymProgram, GymStatement, TensorRef

SOURCE_TO_PROGRAM_CASES = [
    pytest.param(
        "def matmul(a, b):\n    return nkigym.nc_matmul(a, b)\n",
        {"a": (128, 128), "b": (128, 128)},
        np.float32,
        GymProgram(
            name="matmul",
            params=("a", "b"),
            input_shapes=(("a", (128, 128)), ("b", (128, 128))),
            stmts=(
                GymStatement(
                    "nc_matmul",
                    (
                        ("stationary", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),
                        ("moving", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),
                    ),
                    TensorRef("_return", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
            return_var="_return",
            output_dtype=np.float32,
        ),
        id="matmul_128x128",
    ),
    pytest.param(
        "def matmul(a, b):\n    return nkigym.nc_matmul(a, b)\n",
        {"a": (256, 128), "b": (256, 512)},
        np.float32,
        GymProgram(
            name="matmul",
            params=("a", "b"),
            input_shapes=(("a", (256, 128)), ("b", (256, 512))),
            stmts=(
                GymStatement(
                    "nc_matmul",
                    (
                        ("stationary", TensorRef("a", (256, 128), ((0, 256), (0, 128)))),
                        ("moving", TensorRef("b", (256, 512), ((0, 256), (0, 512)))),
                    ),
                    TensorRef("_return", (128, 512), ((0, 128), (0, 512))),
                ),
            ),
            return_var="_return",
            output_dtype=np.float32,
        ),
        id="matmul_nonsquare",
    ),
    pytest.param(
        "def double_matmul(a, b, c):\n    return nkigym.nc_matmul(nkigym.nc_matmul(a, b), c)\n",
        {"a": (128, 128), "b": (128, 128), "c": (128, 128)},
        np.float32,
        GymProgram(
            name="double_matmul",
            params=("a", "b", "c"),
            input_shapes=(("a", (128, 128)), ("b", (128, 128)), ("c", (128, 128))),
            stmts=(
                GymStatement(
                    "nc_matmul",
                    (
                        ("stationary", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),
                        ("moving", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),
                    ),
                    TensorRef("_nested_0", (128, 128), ((0, 128), (0, 128))),
                ),
                GymStatement(
                    "nc_matmul",
                    (
                        ("stationary", TensorRef("_nested_0", (128, 128), ((0, 128), (0, 128)))),
                        ("moving", TensorRef("c", (128, 128), ((0, 128), (0, 128)))),
                    ),
                    TensorRef("_return", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
            return_var="_return",
            output_dtype=np.float32,
        ),
        id="double_matmul",
    ),
    pytest.param(
        "def tensor_tensor(a, b):\n    return nkigym.tensor_tensor(a, b, op=np.add)\n",
        {"a": (128, 128), "b": (128, 128)},
        np.float32,
        GymProgram(
            name="tensor_tensor",
            params=("a", "b"),
            input_shapes=(("a", (128, 128)), ("b", (128, 128))),
            stmts=(
                GymStatement(
                    "tensor_tensor",
                    (
                        ("data1", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),
                        ("data2", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),
                        ("op", np.add),
                    ),
                    TensorRef("_return", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
            return_var="_return",
            output_dtype=np.float32,
        ),
        id="tensor_tensor",
    ),
    pytest.param(
        "def activation(a):\n    return nkigym.activation(a, op=np.tanh)\n",
        {"a": (128, 128)},
        np.float32,
        GymProgram(
            name="activation",
            params=("a",),
            input_shapes=(("a", (128, 128)),),
            stmts=(
                GymStatement(
                    "activation",
                    (("data", TensorRef("a", (128, 128), ((0, 128), (0, 128)))), ("op", np.tanh)),
                    TensorRef("_return", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
            return_var="_return",
            output_dtype=np.float32,
        ),
        id="activation",
    ),
    pytest.param(
        "def assigned(a, b):\n    result = nkigym.nc_matmul(a, b)\n    return result\n",
        {"a": (128, 128), "b": (128, 128)},
        np.float32,
        GymProgram(
            name="assigned",
            params=("a", "b"),
            input_shapes=(("a", (128, 128)), ("b", (128, 128))),
            stmts=(
                GymStatement(
                    "nc_matmul",
                    (
                        ("stationary", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),
                        ("moving", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),
                    ),
                    TensorRef("result", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
            return_var="result",
            output_dtype=np.float32,
        ),
        id="assigned_var",
    ),
    pytest.param(
        "def transpose(a):\n    return nkigym.nc_transpose(a)\n",
        {"a": (128, 64)},
        np.float32,
        GymProgram(
            name="transpose",
            params=("a",),
            input_shapes=(("a", (128, 64)),),
            stmts=(
                GymStatement(
                    "nc_transpose",
                    (("data", TensorRef("a", (128, 64), ((0, 128), (0, 64)))),),
                    TensorRef("_return", (64, 128), ((0, 64), (0, 128))),
                ),
            ),
            return_var="_return",
            output_dtype=np.float32,
        ),
        id="transpose",
    ),
    pytest.param(
        "def tensor_scalar(a, b):\n    return nkigym.tensor_scalar(a, b, op=np.multiply)\n",
        {"a": (128, 128), "b": (128, 1)},
        np.float32,
        GymProgram(
            name="tensor_scalar",
            params=("a", "b"),
            input_shapes=(("a", (128, 128)), ("b", (128, 1))),
            stmts=(
                GymStatement(
                    "tensor_scalar",
                    (
                        ("data", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),
                        ("operand0", TensorRef("b", (128, 1), ((0, 128), (0, 1)))),
                        ("op", np.multiply),
                    ),
                    TensorRef("_return", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
            return_var="_return",
            output_dtype=np.float32,
        ),
        id="tensor_scalar",
    ),
    pytest.param(
        "def exp(a):\n    return nkigym.activation(a, op=np.exp)\n",
        {"a": (128, 128)},
        np.float32,
        GymProgram(
            name="exp",
            params=("a",),
            input_shapes=(("a", (128, 128)),),
            stmts=(
                GymStatement(
                    "activation",
                    (("data", TensorRef("a", (128, 128), ((0, 128), (0, 128)))), ("op", np.exp)),
                    TensorRef("_return", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
            return_var="_return",
            output_dtype=np.float32,
        ),
        id="activation_exp",
    ),
    pytest.param(
        "def multiply(a, b):\n    return nkigym.tensor_tensor(a, b, op=np.multiply)\n",
        {"a": (128, 128), "b": (128, 128)},
        np.float32,
        GymProgram(
            name="multiply",
            params=("a", "b"),
            input_shapes=(("a", (128, 128)), ("b", (128, 128))),
            stmts=(
                GymStatement(
                    "tensor_tensor",
                    (
                        ("data1", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),
                        ("data2", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),
                        ("op", np.multiply),
                    ),
                    TensorRef("_return", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
            return_var="_return",
            output_dtype=np.float32,
        ),
        id="tensor_tensor_multiply",
    ),
    pytest.param(
        "def acc_matmul(a, b, c):\n    return nkigym.nc_matmul(a, b, acc=c)\n",
        {"a": (128, 128), "b": (128, 128), "c": (128, 128)},
        np.float32,
        GymProgram(
            name="acc_matmul",
            params=("a", "b", "c"),
            input_shapes=(("a", (128, 128)), ("b", (128, 128)), ("c", (128, 128))),
            stmts=(
                GymStatement(
                    "nc_matmul",
                    (
                        ("stationary", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),
                        ("moving", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),
                        ("acc", TensorRef("c", (128, 128), ((0, 128), (0, 128)))),
                    ),
                    TensorRef("_return", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
            return_var="_return",
            output_dtype=np.float32,
        ),
        id="acc_matmul",
    ),
]

PROGRAM_TO_SOURCE_CASES = [
    pytest.param(
        GymProgram(
            name="matmul",
            params=("a", "b"),
            input_shapes=(("a", (128, 128)), ("b", (128, 128))),
            stmts=(
                GymStatement(
                    "np_empty", (("dtype", np.float32),), TensorRef("output", (128, 128), ((0, 128), (0, 128)))
                ),
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
            return_var="output",
            output_dtype=np.float32,
        ),
        (
            "import numpy as np\n"
            "import nkigym\n"
            "def matmul(a, b):\n"
            "    output = np.empty((128, 128), dtype=np.float32)\n"
            "    tensor_0 = a[0:128, 0:128]\n"
            "    tensor_1 = b[0:128, 0:128]\n"
            "    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])\n"
            "    output[0:128, 0:128] = tensor_2[0:128, 0:128]\n"
            "    return output\n"
            "\n"
        ),
        {"a": (128, 128), "b": (128, 128)},
        lambda a, b: a.T @ b,
        id="matmul",
    ),
    pytest.param(
        GymProgram(
            name="copy_tile",
            params=("a",),
            input_shapes=(("a", (128, 128)),),
            stmts=(
                GymStatement(
                    "np_empty", (("dtype", np.float32),), TensorRef("output", (128, 128), ((0, 128), (0, 128)))
                ),
                GymStatement(
                    "np_slice",
                    (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
                    TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                ),
                GymStatement(
                    "np_store",
                    (
                        ("src", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                        ("dst", TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
                    ),
                    TensorRef("output", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
            return_var="output",
            output_dtype=np.float32,
        ),
        (
            "import numpy as np\n"
            "import nkigym\n"
            "def copy_tile(a):\n"
            "    output = np.empty((128, 128), dtype=np.float32)\n"
            "    tensor_0 = a[0:128, 0:128]\n"
            "    output[0:128, 0:128] = tensor_0[0:128, 0:128]\n"
            "    return output\n"
            "\n"
        ),
        {"a": (128, 128)},
        lambda a: a,
        id="copy_tile",
    ),
    pytest.param(
        GymProgram(
            name="acc_matmul",
            params=("a", "b"),
            input_shapes=(("a", (256, 128)), ("b", (256, 128))),
            stmts=(
                GymStatement(
                    "np_empty", (("dtype", np.float32),), TensorRef("output", (128, 128), ((0, 128), (0, 128)))
                ),
                GymStatement(
                    "np_slice",
                    (("src", TensorRef("a", (256, 128), ((0, 128), (0, 128)))),),
                    TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                ),
                GymStatement(
                    "np_slice",
                    (("src", TensorRef("b", (256, 128), ((0, 128), (0, 128)))),),
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
                    "np_slice",
                    (("src", TensorRef("a", (256, 128), ((128, 256), (0, 128)))),),
                    TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
                ),
                GymStatement(
                    "np_slice",
                    (("src", TensorRef("b", (256, 128), ((128, 256), (0, 128)))),),
                    TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
                ),
                GymStatement(
                    "nc_matmul",
                    (
                        ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                        ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                        ("acc", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                    ),
                    TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
                ),
                GymStatement(
                    "np_store",
                    (
                        ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                        ("dst", TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
                    ),
                    TensorRef("output", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
            return_var="output",
            output_dtype=np.float32,
        ),
        (
            "import numpy as np\n"
            "import nkigym\n"
            "def acc_matmul(a, b):\n"
            "    output = np.empty((128, 128), dtype=np.float32)\n"
            "    tensor_0 = a[0:128, 0:128]\n"
            "    tensor_1 = b[0:128, 0:128]\n"
            "    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])\n"
            "    tensor_3 = a[128:256, 0:128]\n"
            "    tensor_4 = b[128:256, 0:128]\n"
            "    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128], acc=tensor_2[0:128, 0:128])\n"
            "    output[0:128, 0:128] = tensor_5[0:128, 0:128]\n"
            "    return output\n"
            "\n"
        ),
        {"a": (256, 128), "b": (256, 128)},
        lambda a, b: a.T @ b,
        id="acc_matmul",
    ),
    pytest.param(
        GymProgram(
            name="add_tiles",
            params=("a", "b"),
            input_shapes=(("a", (128, 128)), ("b", (128, 128))),
            stmts=(
                GymStatement(
                    "np_empty", (("dtype", np.float32),), TensorRef("output", (128, 128), ((0, 128), (0, 128)))
                ),
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
                        ("dst", TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
                    ),
                    TensorRef("output", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
            return_var="output",
            output_dtype=np.float32,
        ),
        (
            "import numpy as np\n"
            "import nkigym\n"
            "def add_tiles(a, b):\n"
            "    output = np.empty((128, 128), dtype=np.float32)\n"
            "    tensor_0 = a[0:128, 0:128]\n"
            "    tensor_1 = b[0:128, 0:128]\n"
            "    tensor_2 = nkigym.tensor_tensor(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128], op=np.add)\n"
            "    output[0:128, 0:128] = tensor_2[0:128, 0:128]\n"
            "    return output\n"
            "\n"
        ),
        {"a": (128, 128), "b": (128, 128)},
        lambda a, b: a + b,
        id="add_tiles",
    ),
    pytest.param(
        GymProgram("identity", ("a",), (("a", (128, 128)),), (), "a", np.float32),
        ("import numpy as np\n" "import nkigym\n" "def identity(a):\n" "    return a\n" "\n"),
        {"a": (128, 128)},
        lambda a: a.copy(),
        id="passthrough",
    ),
    pytest.param(
        GymProgram(
            name="transpose_tile",
            params=("a",),
            input_shapes=(("a", (128, 64)),),
            stmts=(
                GymStatement("np_empty", (("dtype", np.float32),), TensorRef("output", (64, 128), ((0, 64), (0, 128)))),
                GymStatement(
                    "np_slice",
                    (("src", TensorRef("a", (128, 64), ((0, 128), (0, 64)))),),
                    TensorRef("tensor_0", (128, 64), ((0, 128), (0, 64))),
                ),
                GymStatement(
                    "nc_transpose",
                    (("data", TensorRef("tensor_0", (128, 64), ((0, 128), (0, 64)))),),
                    TensorRef("tensor_1", (64, 128), ((0, 64), (0, 128))),
                ),
                GymStatement(
                    "np_store",
                    (
                        ("src", TensorRef("tensor_1", (64, 128), ((0, 64), (0, 128)))),
                        ("dst", TensorRef("output", (64, 128), ((0, 64), (0, 128)))),
                    ),
                    TensorRef("output", (64, 128), ((0, 64), (0, 128))),
                ),
            ),
            return_var="output",
            output_dtype=np.float32,
        ),
        (
            "import numpy as np\n"
            "import nkigym\n"
            "def transpose_tile(a):\n"
            "    output = np.empty((64, 128), dtype=np.float32)\n"
            "    tensor_0 = a[0:128, 0:64]\n"
            "    tensor_1 = nkigym.nc_transpose(tensor_0[0:128, 0:64])\n"
            "    output[0:64, 0:128] = tensor_1[0:64, 0:128]\n"
            "    return output\n"
            "\n"
        ),
        {"a": (128, 64)},
        lambda a: a.T,
        id="transpose_tile",
    ),
    pytest.param(
        GymProgram(
            name="scale_rows",
            params=("a", "b"),
            input_shapes=(("a", (128, 128)), ("b", (128, 1))),
            stmts=(
                GymStatement(
                    "np_empty", (("dtype", np.float32),), TensorRef("output", (128, 128), ((0, 128), (0, 128)))
                ),
                GymStatement(
                    "np_slice",
                    (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
                    TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                ),
                GymStatement(
                    "np_slice",
                    (("src", TensorRef("b", (128, 1), ((0, 128), (0, 1)))),),
                    TensorRef("tensor_1", (128, 1), ((0, 128), (0, 1))),
                ),
                GymStatement(
                    "tensor_scalar",
                    (
                        ("data", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                        ("operand0", TensorRef("tensor_1", (128, 1), ((0, 128), (0, 1)))),
                        ("op", np.multiply),
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
            return_var="output",
            output_dtype=np.float32,
        ),
        (
            "import numpy as np\n"
            "import nkigym\n"
            "def scale_rows(a, b):\n"
            "    output = np.empty((128, 128), dtype=np.float32)\n"
            "    tensor_0 = a[0:128, 0:128]\n"
            "    tensor_1 = b[0:128, 0:1]\n"
            "    tensor_2 = nkigym.tensor_scalar(tensor_0[0:128, 0:128], tensor_1[0:128, 0:1], op=np.multiply)\n"
            "    output[0:128, 0:128] = tensor_2[0:128, 0:128]\n"
            "    return output\n"
            "\n"
        ),
        {"a": (128, 128), "b": (128, 1)},
        lambda a, b: a * b,
        id="scale_rows",
    ),
]

F_ROUND_TRIP_CASES = [
    pytest.param(
        "import numpy as np\nimport nkigym\ndef matmul(a, b):\n    _return = nkigym.nc_matmul(a[0:128, 0:128], b[0:128, 0:128])\n    return _return\n\n",
        {"a": (128, 128), "b": (128, 128)},
        id="matmul_128x128",
    ),
    pytest.param(
        "import numpy as np\nimport nkigym\ndef matmul(a, b):\n    _return = nkigym.nc_matmul(a[0:256, 0:128], b[0:256, 0:512])\n    return _return\n\n",
        {"a": (256, 128), "b": (256, 512)},
        id="matmul_nonsquare",
    ),
    pytest.param(
        "import numpy as np\nimport nkigym\ndef double_matmul(a, b, c):\n    _nested_0 = nkigym.nc_matmul(a[0:128, 0:128], b[0:128, 0:128])\n    _return = nkigym.nc_matmul(_nested_0[0:128, 0:128], c[0:128, 0:128])\n    return _return\n\n",
        {"a": (128, 128), "b": (128, 128), "c": (128, 128)},
        id="double_matmul",
    ),
    pytest.param(
        "import numpy as np\nimport nkigym\ndef tensor_tensor(a, b):\n    _return = nkigym.tensor_tensor(a[0:128, 0:128], b[0:128, 0:128], op=np.add)\n    return _return\n\n",
        {"a": (128, 128), "b": (128, 128)},
        id="tensor_tensor",
    ),
    pytest.param(
        "import numpy as np\nimport nkigym\ndef activation(a):\n    _return = nkigym.activation(a[0:128, 0:128], op=np.tanh)\n    return _return\n\n",
        {"a": (128, 128)},
        id="activation_tanh",
    ),
    pytest.param(
        "import numpy as np\nimport nkigym\ndef transpose(a):\n    _return = nkigym.nc_transpose(a[0:128, 0:64])\n    return _return\n\n",
        {"a": (128, 64)},
        id="transpose",
    ),
    pytest.param(
        "import numpy as np\nimport nkigym\ndef transpose(a):\n    _return = nkigym.nc_transpose(a[0:64, 0:256])\n    return _return\n\n",
        {"a": (64, 256)},
        id="transpose_wide",
    ),
    pytest.param(
        "import numpy as np\nimport nkigym\ndef tensor_scalar(a, b):\n    _return = nkigym.tensor_scalar(a[0:128, 0:128], b[0:128, 0:1], op=np.multiply)\n    return _return\n\n",
        {"a": (128, 128), "b": (128, 1)},
        id="tensor_scalar",
    ),
    pytest.param(
        "import numpy as np\nimport nkigym\ndef exp(a):\n    _return = nkigym.activation(a[0:128, 0:128], op=np.exp)\n    return _return\n\n",
        {"a": (128, 128)},
        id="activation_exp",
    ),
    pytest.param(
        "import numpy as np\nimport nkigym\ndef multiply(a, b):\n    _return = nkigym.tensor_tensor(a[0:128, 0:128], b[0:128, 0:128], op=np.multiply)\n    return _return\n\n",
        {"a": (128, 128), "b": (128, 128)},
        id="tensor_tensor_multiply",
    ),
    pytest.param(
        "import numpy as np\nimport nkigym\ndef acc_matmul(a, b, c):\n    _return = nkigym.nc_matmul(a[0:128, 0:128], b[0:128, 0:128], acc=c[0:128, 0:128])\n    return _return\n\n",
        {"a": (128, 128), "b": (128, 128), "c": (128, 128)},
        id="acc_matmul",
    ),
]

P_ROUND_TRIP_CASES = [pytest.param(case.values[3], id=case.id) for case in SOURCE_TO_PROGRAM_CASES]
