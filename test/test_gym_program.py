"""Tests for GymProgram parsing and execution."""

import numpy as np
import pytest
from conftest import make_random_array

import nkigym
from nkigym.ir import GymProgram, GymStatement, TensorRef, func_to_program, program_to_func


class TestFuncToProgram:
    """Tests for func_to_program parsing with input_shapes."""

    def test_matmul_kwargs(self) -> None:
        """Verify positional args are mapped to operand names with TensorRef."""

        def fn(lhs, rhs):
            return nkigym.nc_matmul(lhs, rhs)

        program = func_to_program(fn, {"lhs": (128, 64), "rhs": (128, 32)}, np.float32)
        assert program.stmts[0].kwargs == (
            ("stationary", TensorRef("lhs", (128, 64), ((0, 128), (0, 64)))),
            ("moving", TensorRef("rhs", (128, 32), ((0, 128), (0, 32)))),
        )

    def test_activation_with_op_kwarg(self) -> None:
        """Verify op=np.tanh is captured as a kwarg alongside TensorRef."""

        def fn(x):
            return nkigym.activation(x, op=np.tanh)

        program = func_to_program(fn, {"x": (64, 64)}, np.float32)
        stmt = program.stmts[0]
        assert stmt.kwargs == (("data", TensorRef("x", (64, 64), ((0, 64), (0, 64)))), ("op", "np.tanh"))

    def test_tensor_tensor_with_op_kwarg(self) -> None:
        """Verify tensor_tensor captures op kwarg."""

        def fn(a, b):
            return nkigym.tensor_tensor(a, b, op=np.multiply)

        program = func_to_program(fn, {"a": (64, 64), "b": (64, 64)}, np.float32)
        stmt = program.stmts[0]
        assert stmt.kwargs == (
            ("data1", TensorRef("a", (64, 64), ((0, 64), (0, 64)))),
            ("data2", TensorRef("b", (64, 64), ((0, 64), (0, 64)))),
            ("op", "np.multiply"),
        )

    def test_tensor_scalar_with_op_kwarg(self) -> None:
        """Verify tensor_scalar captures op kwarg."""

        def fn(data, scale):
            return nkigym.tensor_scalar(data, scale, op=np.add)

        program = func_to_program(fn, {"data": (64, 64), "scale": ()}, np.float32)
        stmt = program.stmts[0]
        assert stmt.kwargs[0] == ("data", TensorRef("data", (64, 64), ((0, 64), (0, 64))))
        assert stmt.kwargs[1] == ("operand0", TensorRef("scale", (), ()))
        assert stmt.kwargs[2] == ("op", "np.add")

    def test_transpose_no_kwargs(self) -> None:
        """Verify nc_transpose has only positional-mapped kwargs."""

        def fn(x):
            return nkigym.nc_transpose(x)

        program = func_to_program(fn, {"x": (64, 128)}, np.float32)
        assert program.stmts[0].kwargs == (("data", TensorRef("x", (64, 128), ((0, 64), (0, 128)))),)

    def test_multi_statement(self) -> None:
        """Verify multi-statement programs parse correctly with TensorRef."""

        def fn(lhs, rhs):
            t = nkigym.nc_matmul(lhs, rhs)
            return nkigym.activation(t, op=np.tanh)

        program = func_to_program(fn, {"lhs": (128, 64), "rhs": (128, 32)}, np.float32)
        assert len(program.stmts) == 2
        assert program.stmts[0].op == "nc_matmul"
        assert program.stmts[0].output == TensorRef("t", (64, 32), ((0, 64), (0, 32)))
        assert program.stmts[1].op == "activation"
        assert program.stmts[1].kwargs[0] == ("data", TensorRef("t", (64, 32), ((0, 64), (0, 32))))
        assert program.stmts[1].kwargs[1] == ("op", "np.tanh")

    def test_program_is_hashable(self) -> None:
        """Verify GymProgram remains hashable."""

        def fn(a, b):
            return nkigym.tensor_tensor(a, b, op=np.add)

        p1 = func_to_program(fn, {"a": (64, 64), "b": (64, 64)}, np.float32)

        def fn2(a, b):
            return nkigym.tensor_tensor(a, b, op=np.multiply)

        p2 = func_to_program(fn2, {"a": (64, 64), "b": (64, 64)}, np.float32)
        d = {p1: "add", p2: "mul"}
        assert len(d) == 2
        assert d[p1] == "add"

    def test_wrong_arg_count_raises(self) -> None:
        """Verify ValueError when positional arg count mismatches."""

        def fn(a, b, c):
            return nkigym.nc_matmul(a, b, c)

        fn.__source__ = "def fn(a, b, c):\n    return nkigym.nc_matmul(a, b, c)\n"
        with pytest.raises(ValueError, match="expects 2 positional args"):
            func_to_program(fn, {"a": (4, 4), "b": (4, 4), "c": (4, 4)}, np.float32)

    def test_output_dtype_stored(self) -> None:
        """Verify output_dtype is stored on the program."""

        def fn(a, b):
            return nkigym.nc_matmul(a, b)

        program = func_to_program(fn, {"a": (128, 128), "b": (128, 128)}, np.float64)
        assert program.output_dtype is np.float64

    def test_output_shape_inferred(self) -> None:
        """Verify output shapes are correctly inferred through op chain."""

        def fn(lhs, rhs):
            return nkigym.nc_matmul(lhs, rhs)

        program = func_to_program(fn, {"lhs": (128, 64), "rhs": (128, 32)}, np.float32)
        assert program.stmts[0].output.shape == (64, 32)

    def test_input_shapes_stored(self) -> None:
        """Verify input_shapes field matches what was passed."""

        def fn(lhs, rhs):
            return nkigym.nc_matmul(lhs, rhs)

        program = func_to_program(fn, {"lhs": (128, 64), "rhs": (128, 32)}, np.float32)
        assert program.input_shapes == (("lhs", (128, 64)), ("rhs", (128, 32)))


MATMUL_SHAPES = [
    ("128x64_128x32", (128, 64), (128, 32)),
    ("128x128_128x128", (128, 128), (128, 128)),
    ("64x32_64x64", (64, 32), (64, 64)),
]

ELEMENTWISE_SHAPES = [(128, 128), (64, 32), (32, 64)]


class TestProgramExecution:
    """Tests for program_to_func execution."""

    @pytest.mark.parametrize("name,lhs_shape,rhs_shape", MATMUL_SHAPES, ids=[c[0] for c in MATMUL_SHAPES])
    def test_matmul(self, name: str, lhs_shape: tuple[int, int], rhs_shape: tuple[int, int]) -> None:
        """Verify compiled program matches np.matmul(lhs.T, rhs)."""

        def fn(lhs, rhs):
            return nkigym.nc_matmul(lhs, rhs)

        lhs = make_random_array(lhs_shape, seed=42)
        rhs = make_random_array(rhs_shape, seed=43)
        program = func_to_program(fn, {"lhs": lhs_shape, "rhs": rhs_shape}, np.float32)
        actual = program_to_func(program)(lhs, rhs)
        expected = np.matmul(lhs.T, rhs)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES, ids=[f"{s[0]}x{s[1]}" for s in ELEMENTWISE_SHAPES])
    def test_activation_tanh(self, shape: tuple[int, int]) -> None:
        """Verify compiled program applies np.tanh via kwarg."""

        def fn(x):
            return nkigym.activation(x, op=np.tanh)

        x = make_random_array(shape, seed=42)
        program = func_to_program(fn, {"x": shape}, np.float32)
        actual = program_to_func(program)(x)
        expected = np.tanh(x)
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)

    def test_activation_identity(self) -> None:
        """Verify activation without op kwarg is identity."""

        def fn(x):
            return nkigym.activation(x)

        x = make_random_array((64, 64), seed=42)
        program = func_to_program(fn, {"x": (64, 64)}, np.float32)
        actual = program_to_func(program)(x)
        np.testing.assert_array_equal(actual, x)

    @pytest.mark.parametrize(
        "op_attr,np_func",
        [("np.add", np.add), ("np.multiply", np.multiply), ("np.subtract", np.subtract)],
        ids=["add", "multiply", "subtract"],
    )
    def test_tensor_tensor(self, op_attr: str, np_func) -> None:
        """Verify compiled program dispatches tensor_tensor op kwarg."""
        shape = (64, 64)
        stmt = GymStatement(
            op="tensor_tensor",
            kwargs=(
                ("data1", TensorRef("a", shape, ((0, 64), (0, 64)))),
                ("data2", TensorRef("b", shape, ((0, 64), (0, 64)))),
                ("op", op_attr),
            ),
            output=TensorRef("_return", shape, ((0, 64), (0, 64))),
        )
        program = GymProgram(
            name="fn",
            params=("a", "b"),
            input_shapes=(("a", shape), ("b", shape)),
            stmts=(stmt,),
            return_var="_return",
            output_dtype=np.float32,
        )
        a = make_random_array(shape, seed=42)
        b = make_random_array(shape, seed=43)
        actual = program_to_func(program)(a, b)
        expected = np_func(a, b)
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)

    def test_tensor_scalar(self) -> None:
        """Verify compiled program dispatches tensor_scalar op kwarg."""

        def fn(data, scale):
            return nkigym.tensor_scalar(data, scale, op=np.multiply)

        data = make_random_array((64, 64), seed=42)
        scale = np.float32(2.5)
        program = func_to_program(fn, {"data": (64, 64), "scale": ()}, np.float32)
        actual = program_to_func(program)(data, scale)
        expected = np.multiply(data, scale)
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)

    def test_transpose(self) -> None:
        """Verify compiled program transposes correctly."""

        def fn(x):
            return nkigym.nc_transpose(x)

        x = make_random_array((64, 128), seed=42)
        program = func_to_program(fn, {"x": (64, 128)}, np.float32)
        actual = program_to_func(program)(x)
        expected = x.T
        np.testing.assert_array_equal(actual, expected)

    def test_chained_matmul_tanh(self) -> None:
        """Verify multi-statement: matmul -> tanh."""

        def fn(lhs, rhs):
            t = nkigym.nc_matmul(lhs, rhs)
            return nkigym.activation(t, op=np.tanh)

        lhs = make_random_array((128, 64), seed=42)
        rhs = make_random_array((128, 32), seed=43)
        program = func_to_program(fn, {"lhs": (128, 64), "rhs": (128, 32)}, np.float32)
        actual = program_to_func(program)(lhs, rhs)
        expected = np.tanh(np.matmul(lhs.T, rhs))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_double_matmul(self) -> None:
        """Verify chained matmul: (A.T @ B).T @ C."""

        def fn(a, b, c):
            t = nkigym.nc_matmul(a, b)
            return nkigym.nc_matmul(t, c)

        a = make_random_array((128, 64), seed=42)
        b = make_random_array((128, 32), seed=43)
        c = make_random_array((64, 128), seed=44)
        program = func_to_program(fn, {"a": (128, 64), "b": (128, 32), "c": (64, 128)}, np.float32)
        actual = program_to_func(program)(a, b, c)
        t = np.matmul(a.T, b)
        expected = np.matmul(t.T, c)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_missing_input_raises(self) -> None:
        """Verify TypeError when required inputs are missing."""

        def fn(a, b):
            return nkigym.nc_matmul(a, b)

        program = func_to_program(fn, {"a": (4, 4), "b": (4, 4)}, np.float32)
        with pytest.raises(TypeError, match="missing.*required.*argument"):
            program_to_func(program)(np.zeros((4, 4)))

    def test_matches_direct_nkigym(self) -> None:
        """Verify compiled program matches direct nkigym function call."""

        def fn(lhs, rhs):
            t = nkigym.nc_matmul(lhs, rhs)
            return nkigym.activation(t, op=np.tanh)

        lhs = make_random_array((128, 256), seed=42)
        rhs = make_random_array((128, 128), seed=43)
        program = func_to_program(fn, {"lhs": (128, 256), "rhs": (128, 128)}, np.float32)
        actual = program_to_func(program)(lhs, rhs)
        expected = fn(lhs, rhs)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
