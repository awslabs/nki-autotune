"""Golden test data for IR tests: expected GymProgram outputs and parametrized cases."""

import numpy as np
import pytest

import nkigym
from nkigym.ir import GymProgram, GymStatement, TensorRef


def _ref(name: str, shape: tuple[int, ...] = (128, 128)) -> TensorRef:
    """Build a TensorRef with full-range slices."""
    return TensorRef(name, shape, tuple((0, s) for s in shape))


def _fn_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matmul: stationary.T @ moving."""
    return nkigym.nc_matmul(a, b)


def _fn_double_matmul(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Double matmul: (a.T @ b).T @ c."""
    return nkigym.nc_matmul(nkigym.nc_matmul(a, b), c)


def _fn_tensor_tensor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise add."""
    return nkigym.tensor_tensor(a, b, op=np.add)


def _fn_activation(a: np.ndarray) -> np.ndarray:
    """Tanh activation."""
    return nkigym.activation(a, op=np.tanh)


def _fn_assigned(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matmul with assigned variable."""
    result = nkigym.nc_matmul(a, b)
    return result


def _fn_transpose(a: np.ndarray) -> np.ndarray:
    """Transpose."""
    return nkigym.nc_transpose(a)


def _fn_tensor_scalar(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Tensor-scalar multiply."""
    return nkigym.tensor_scalar(a, b, op=np.multiply)


def _fn_exp(a: np.ndarray) -> np.ndarray:
    """Exp activation."""
    return nkigym.activation(a, op=np.exp)


def _fn_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise multiply."""
    return nkigym.tensor_tensor(a, b, op=np.multiply)


FUNC_TO_PROGRAM_CASES = [
    pytest.param(
        _fn_matmul,
        {"a": (128, 128), "b": (128, 128)},
        np.float32,
        GymProgram(
            name="_fn_matmul",
            params=("a", "b"),
            input_shapes=(("a", (128, 128)), ("b", (128, 128))),
            stmts=(GymStatement("nc_matmul", (("stationary", _ref("a")), ("moving", _ref("b"))), _ref("_return")),),
            return_var="_return",
            output_dtype=np.float32,
        ),
        id="matmul_128x128",
    ),
    pytest.param(
        _fn_matmul,
        {"a": (256, 128), "b": (256, 512)},
        np.float32,
        GymProgram(
            name="_fn_matmul",
            params=("a", "b"),
            input_shapes=(("a", (256, 128)), ("b", (256, 512))),
            stmts=(
                GymStatement(
                    "nc_matmul",
                    (("stationary", _ref("a", (256, 128))), ("moving", _ref("b", (256, 512)))),
                    _ref("_return", (128, 512)),
                ),
            ),
            return_var="_return",
            output_dtype=np.float32,
        ),
        id="matmul_nonsquare",
    ),
    pytest.param(
        _fn_double_matmul,
        {"a": (128, 128), "b": (128, 128), "c": (128, 128)},
        np.float32,
        GymProgram(
            name="_fn_double_matmul",
            params=("a", "b", "c"),
            input_shapes=(("a", (128, 128)), ("b", (128, 128)), ("c", (128, 128))),
            stmts=(
                GymStatement("nc_matmul", (("stationary", _ref("a")), ("moving", _ref("b"))), _ref("_nested_0")),
                GymStatement("nc_matmul", (("stationary", _ref("_nested_0")), ("moving", _ref("c"))), _ref("_return")),
            ),
            return_var="_return",
            output_dtype=np.float32,
        ),
        id="double_matmul",
    ),
    pytest.param(
        _fn_tensor_tensor,
        {"a": (128, 128), "b": (128, 128)},
        np.float32,
        GymProgram(
            name="_fn_tensor_tensor",
            params=("a", "b"),
            input_shapes=(("a", (128, 128)), ("b", (128, 128))),
            stmts=(
                GymStatement(
                    "tensor_tensor", (("data1", _ref("a")), ("data2", _ref("b")), ("op", "np.add")), _ref("_return")
                ),
            ),
            return_var="_return",
            output_dtype=np.float32,
        ),
        id="tensor_tensor",
    ),
    pytest.param(
        _fn_activation,
        {"a": (128, 128)},
        np.float32,
        GymProgram(
            name="_fn_activation",
            params=("a",),
            input_shapes=(("a", (128, 128)),),
            stmts=(GymStatement("activation", (("data", _ref("a")), ("op", "np.tanh")), _ref("_return")),),
            return_var="_return",
            output_dtype=np.float32,
        ),
        id="activation",
    ),
    pytest.param(
        _fn_assigned,
        {"a": (128, 128), "b": (128, 128)},
        np.float32,
        GymProgram(
            name="_fn_assigned",
            params=("a", "b"),
            input_shapes=(("a", (128, 128)), ("b", (128, 128))),
            stmts=(GymStatement("nc_matmul", (("stationary", _ref("a")), ("moving", _ref("b"))), _ref("result")),),
            return_var="result",
            output_dtype=np.float32,
        ),
        id="assigned_var",
    ),
    pytest.param(
        _fn_transpose,
        {"a": (128, 64)},
        np.float32,
        GymProgram(
            name="_fn_transpose",
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
        _fn_tensor_scalar,
        {"a": (128, 128), "b": (128, 1)},
        np.float32,
        GymProgram(
            name="_fn_tensor_scalar",
            params=("a", "b"),
            input_shapes=(("a", (128, 128)), ("b", (128, 1))),
            stmts=(
                GymStatement(
                    "tensor_scalar",
                    (
                        ("data", _ref("a")),
                        ("operand0", TensorRef("b", (128, 1), ((0, 128), (0, 1)))),
                        ("op", "np.multiply"),
                    ),
                    _ref("_return"),
                ),
            ),
            return_var="_return",
            output_dtype=np.float32,
        ),
        id="tensor_scalar",
    ),
    pytest.param(
        _fn_exp,
        {"a": (128, 128)},
        np.float32,
        GymProgram(
            name="_fn_exp",
            params=("a",),
            input_shapes=(("a", (128, 128)),),
            stmts=(GymStatement("activation", (("data", _ref("a")), ("op", "np.exp")), _ref("_return")),),
            return_var="_return",
            output_dtype=np.float32,
        ),
        id="activation_exp",
    ),
    pytest.param(
        _fn_multiply,
        {"a": (128, 128), "b": (128, 128)},
        np.float32,
        GymProgram(
            name="_fn_multiply",
            params=("a", "b"),
            input_shapes=(("a", (128, 128)), ("b", (128, 128))),
            stmts=(
                GymStatement(
                    "tensor_tensor",
                    (("data1", _ref("a")), ("data2", _ref("b")), ("op", "np.multiply")),
                    _ref("_return"),
                ),
            ),
            return_var="_return",
            output_dtype=np.float32,
        ),
        id="tensor_tensor_multiply",
    ),
]

PROGRAM_TO_FUNC_CASES = [
    pytest.param(
        GymProgram(
            name="matmul",
            params=("a", "b"),
            input_shapes=(("a", (128, 128)), ("b", (128, 128))),
            stmts=(
                GymStatement("np_empty", (("dtype", "np.float32"),), _ref("output")),
                GymStatement("np_slice", (("src", _ref("a")),), _ref("tensor_0")),
                GymStatement("np_slice", (("src", _ref("b")),), _ref("tensor_1")),
                GymStatement(
                    "nc_matmul", (("stationary", _ref("tensor_0")), ("moving", _ref("tensor_1"))), _ref("tensor_2")
                ),
                GymStatement("np_store", (("src", _ref("tensor_2")), ("dst", _ref("output"))), _ref("output")),
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
                GymStatement("np_empty", (("dtype", "np.float32"),), _ref("output")),
                GymStatement("np_slice", (("src", _ref("a")),), _ref("tensor_0")),
                GymStatement("np_store", (("src", _ref("tensor_0")), ("dst", _ref("output"))), _ref("output")),
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
                GymStatement("np_empty", (("dtype", "np.float32"),), _ref("output")),
                GymStatement(
                    "np_slice", (("src", TensorRef("a", (256, 128), ((0, 128), (0, 128)))),), _ref("tensor_0")
                ),
                GymStatement(
                    "np_slice", (("src", TensorRef("b", (256, 128), ((0, 128), (0, 128)))),), _ref("tensor_1")
                ),
                GymStatement(
                    "nc_matmul", (("stationary", _ref("tensor_0")), ("moving", _ref("tensor_1"))), _ref("tensor_2")
                ),
                GymStatement(
                    "np_slice", (("src", TensorRef("a", (256, 128), ((128, 256), (0, 128)))),), _ref("tensor_3")
                ),
                GymStatement(
                    "np_slice", (("src", TensorRef("b", (256, 128), ((128, 256), (0, 128)))),), _ref("tensor_4")
                ),
                GymStatement(
                    "nc_matmul",
                    (("stationary", _ref("tensor_3")), ("moving", _ref("tensor_4")), ("acc", _ref("tensor_2"))),
                    _ref("tensor_2"),
                ),
                GymStatement("np_store", (("src", _ref("tensor_2")), ("dst", _ref("output"))), _ref("output")),
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
            "    tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])\n"
            "    output[0:128, 0:128] = tensor_2[0:128, 0:128]\n"
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
                GymStatement("np_empty", (("dtype", "np.float32"),), _ref("output")),
                GymStatement("np_slice", (("src", _ref("a")),), _ref("tensor_0")),
                GymStatement("np_slice", (("src", _ref("b")),), _ref("tensor_1")),
                GymStatement(
                    "tensor_tensor",
                    (("data1", _ref("tensor_0")), ("data2", _ref("tensor_1")), ("op", "np.add")),
                    _ref("tensor_2"),
                ),
                GymStatement("np_store", (("src", _ref("tensor_2")), ("dst", _ref("output"))), _ref("output")),
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
                GymStatement(
                    "np_empty", (("dtype", "np.float32"),), TensorRef("output", (64, 128), ((0, 64), (0, 128)))
                ),
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
                GymStatement("np_empty", (("dtype", "np.float32"),), _ref("output")),
                GymStatement("np_slice", (("src", _ref("a")),), _ref("tensor_0")),
                GymStatement(
                    "np_slice",
                    (("src", TensorRef("b", (128, 1), ((0, 128), (0, 1)))),),
                    TensorRef("tensor_1", (128, 1), ((0, 128), (0, 1))),
                ),
                GymStatement(
                    "tensor_scalar",
                    (
                        ("data", _ref("tensor_0")),
                        ("operand0", TensorRef("tensor_1", (128, 1), ((0, 128), (0, 1)))),
                        ("op", "np.multiply"),
                    ),
                    _ref("tensor_2"),
                ),
                GymStatement("np_store", (("src", _ref("tensor_2")), ("dst", _ref("output"))), _ref("output")),
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

ROUND_TRIP_CASES = [
    pytest.param(
        _fn_matmul,
        {"a": (128, 128), "b": (128, 128)},
        (
            "import numpy as np\n"
            "import nkigym\n"
            "def _fn_matmul(a, b):\n"
            "    _return = nkigym.nc_matmul(a[0:128, 0:128], b[0:128, 0:128])\n"
            "    return _return\n"
            "\n"
        ),
        id="matmul_128x128",
    ),
    pytest.param(
        _fn_matmul,
        {"a": (256, 128), "b": (256, 512)},
        (
            "import numpy as np\n"
            "import nkigym\n"
            "def _fn_matmul(a, b):\n"
            "    _return = nkigym.nc_matmul(a[0:256, 0:128], b[0:256, 0:512])\n"
            "    return _return\n"
            "\n"
        ),
        id="matmul_nonsquare",
    ),
    pytest.param(
        _fn_double_matmul,
        {"a": (128, 128), "b": (128, 128), "c": (128, 128)},
        (
            "import numpy as np\n"
            "import nkigym\n"
            "def _fn_double_matmul(a, b, c):\n"
            "    _nested_0 = nkigym.nc_matmul(a[0:128, 0:128], b[0:128, 0:128])\n"
            "    _return = nkigym.nc_matmul(_nested_0[0:128, 0:128], c[0:128, 0:128])\n"
            "    return _return\n"
            "\n"
        ),
        id="double_matmul",
    ),
    pytest.param(
        _fn_tensor_tensor,
        {"a": (128, 128), "b": (128, 128)},
        (
            "import numpy as np\n"
            "import nkigym\n"
            "def _fn_tensor_tensor(a, b):\n"
            "    _return = nkigym.tensor_tensor(a[0:128, 0:128], b[0:128, 0:128], op=np.add)\n"
            "    return _return\n"
            "\n"
        ),
        id="tensor_tensor",
    ),
    pytest.param(
        _fn_activation,
        {"a": (128, 128)},
        (
            "import numpy as np\n"
            "import nkigym\n"
            "def _fn_activation(a):\n"
            "    _return = nkigym.activation(a[0:128, 0:128], op=np.tanh)\n"
            "    return _return\n"
            "\n"
        ),
        id="activation_tanh",
    ),
    pytest.param(
        _fn_transpose,
        {"a": (128, 64)},
        (
            "import numpy as np\n"
            "import nkigym\n"
            "def _fn_transpose(a):\n"
            "    _return = nkigym.nc_transpose(a[0:128, 0:64])\n"
            "    return _return\n"
            "\n"
        ),
        id="transpose",
    ),
    pytest.param(
        _fn_transpose,
        {"a": (64, 256)},
        (
            "import numpy as np\n"
            "import nkigym\n"
            "def _fn_transpose(a):\n"
            "    _return = nkigym.nc_transpose(a[0:64, 0:256])\n"
            "    return _return\n"
            "\n"
        ),
        id="transpose_wide",
    ),
    pytest.param(
        _fn_tensor_scalar,
        {"a": (128, 128), "b": (128, 1)},
        (
            "import numpy as np\n"
            "import nkigym\n"
            "def _fn_tensor_scalar(a, b):\n"
            "    _return = nkigym.tensor_scalar(a[0:128, 0:128], b[0:128, 0:1], op=np.multiply)\n"
            "    return _return\n"
            "\n"
        ),
        id="tensor_scalar",
    ),
    pytest.param(
        _fn_exp,
        {"a": (128, 128)},
        (
            "import numpy as np\n"
            "import nkigym\n"
            "def _fn_exp(a):\n"
            "    _return = nkigym.activation(a[0:128, 0:128], op=np.exp)\n"
            "    return _return\n"
            "\n"
        ),
        id="activation_exp",
    ),
    pytest.param(
        _fn_multiply,
        {"a": (128, 128), "b": (128, 128)},
        (
            "import numpy as np\n"
            "import nkigym\n"
            "def _fn_multiply(a, b):\n"
            "    _return = nkigym.tensor_tensor(a[0:128, 0:128], b[0:128, 0:128], op=np.multiply)\n"
            "    return _return\n"
            "\n"
        ),
        id="tensor_tensor_multiply",
    ),
]
