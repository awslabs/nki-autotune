"""Golden test data for IR tests: expected GymProgram outputs and parametrized cases."""

import numpy as np
import pytest

from nkigym.ir import GymProgram, GymStatement, TensorRef  # type: ignore[import]
from nkigym.ops.activation import ActivationOp  # type: ignore[import]
from nkigym.ops.matmul import MatmulOp  # type: ignore[import]
from nkigym.ops.nc_transpose import NcTransposeOp  # type: ignore[import]
from nkigym.ops.tensor_scalar import TensorScalarOp  # type: ignore[import]
from nkigym.ops.tensor_tensor import TensorTensorOp  # type: ignore[import]
from nkigym.ops.tiling_ops import AllocateOp, LoadOp, StoreOp  # type: ignore[import]


def _kw(shapes: dict) -> dict:
    """Create zero-filled kwargs from shape dict."""
    return {k: np.zeros(v, dtype=np.float32) for k, v in shapes.items()}


SOURCE_TO_PROGRAM_CASES = [
    pytest.param(
        "def matmul(a, b):\n    return nkigym.nc_matmul(a, b)\n",
        {"a": (128, 128), "b": (128, 128)},
        np.float32,
        GymProgram(
            name="matmul",
            kwargs=_kw({"a": (128, 128), "b": (128, 128)}),
            stmts=(
                GymStatement(
                    MatmulOp,
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
            kwargs=_kw({"a": (256, 128), "b": (256, 512)}),
            stmts=(
                GymStatement(
                    MatmulOp,
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
            kwargs=_kw({"a": (128, 128), "b": (128, 128), "c": (128, 128)}),
            stmts=(
                GymStatement(
                    MatmulOp,
                    (
                        ("stationary", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),
                        ("moving", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),
                    ),
                    TensorRef("_nested_0", (128, 128), ((0, 128), (0, 128))),
                ),
                GymStatement(
                    MatmulOp,
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
            kwargs=_kw({"a": (128, 128), "b": (128, 128)}),
            stmts=(
                GymStatement(
                    TensorTensorOp,
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
            kwargs=_kw({"a": (128, 128)}),
            stmts=(
                GymStatement(
                    ActivationOp,
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
            kwargs=_kw({"a": (128, 128), "b": (128, 128)}),
            stmts=(
                GymStatement(
                    MatmulOp,
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
            kwargs=_kw({"a": (128, 64)}),
            stmts=(
                GymStatement(
                    NcTransposeOp,
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
            kwargs=_kw({"a": (128, 128), "b": (128, 1)}),
            stmts=(
                GymStatement(
                    TensorScalarOp,
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
            kwargs=_kw({"a": (128, 128)}),
            stmts=(
                GymStatement(
                    ActivationOp,
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
            kwargs=_kw({"a": (128, 128), "b": (128, 128)}),
            stmts=(
                GymStatement(
                    TensorTensorOp,
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
            kwargs=_kw({"a": (128, 128), "b": (128, 128), "c": (128, 128)}),
            stmts=(
                GymStatement(
                    MatmulOp,
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
            kwargs=_kw({"a": (128, 128), "b": (128, 128)}),
            stmts=(
                GymStatement(
                    AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 128), ((0, 128), (0, 128)))
                ),
                GymStatement(
                    LoadOp,
                    (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
                    TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                ),
                GymStatement(
                    LoadOp,
                    (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
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
            "    tensor_0 = nkigym.load(a[0:128, 0:128])\n"
            "    tensor_1 = nkigym.load(b[0:128, 0:128])\n"
            "    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])\n"
            "    nkigym.store(tensor_2[0:128, 0:128], output[0:128, 0:128])\n"
            "    return output\n"
            "\n"
        ),
        {"a": (128, 128), "b": (128, 128)},
        lambda a, b: a.T @ b,
        id="matmul",
    ),
    pytest.param(
        GymProgram("identity", _kw({"a": (128, 128)}), (), "a", np.float32),
        ("import numpy as np\n" "import nkigym\n" "def identity(a):\n" "    return a\n" "\n"),
        {"a": (128, 128)},
        lambda a: a.copy(),
        id="passthrough",
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
