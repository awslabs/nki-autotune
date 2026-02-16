"""Tests for IR: func_to_program, program_to_func, and program_to_source."""

import numpy as np
import pytest
from conftest import assert_arrays_close
from golden_ir import FUNC_TO_PROGRAM_CASES, PROGRAM_TO_FUNC_CASES, ROUND_TRIP_CASES, _fn_matmul

import nkigym
from nkigym.ir import func_to_program, program_to_func, program_to_source


def _random_arrays(params: tuple[str, ...], input_shapes: dict[str, tuple[int, ...]]) -> list[np.ndarray]:
    """Generate random float32 arrays for each parameter."""
    rng = np.random.default_rng()
    return [rng.standard_normal(input_shapes[name]).astype(np.float32) for name in params]


class TestFuncToProgram:
    """Tests for func_to_program parsing nkigym callables into GymProgram."""

    @pytest.mark.parametrize("fn,input_shapes,dtype,expected", FUNC_TO_PROGRAM_CASES)
    def test_parse(self, fn, input_shapes, dtype, expected) -> None:
        """Verify func_to_program produces expected GymProgram."""
        actual = func_to_program(fn, input_shapes, dtype)
        assert actual == expected

    def test_dtype_propagated(self) -> None:
        """Verify output_dtype is set correctly on GymProgram."""
        program = func_to_program(_fn_matmul, {"a": (128, 128), "b": (128, 128)}, np.float64)
        assert program.output_dtype is np.float64

    def test_error_unknown_op(self) -> None:
        """Verify KeyError for unknown nkigym operation."""

        def bad_fn(a):
            """Bad function."""
            return nkigym.unknown_op(a)

        with pytest.raises(KeyError, match="Unknown op"):
            func_to_program(bad_fn, {"a": (128, 128)}, np.float32)


class TestProgramToFunc:
    """Tests for program_to_func and program_to_source with hand-crafted GymPrograms."""

    @pytest.mark.parametrize("program,expected_source,input_shapes,expected_fn", PROGRAM_TO_FUNC_CASES)
    def test_source_and_execution(self, program, expected_source, input_shapes, expected_fn) -> None:
        """Verify source rendering matches expected string and execution is numerically correct."""
        actual_source = program_to_source(program)
        assert actual_source == expected_source
        func = program_to_func(program)
        assert hasattr(func, "__source__")
        arrays = _random_arrays(program.params, input_shapes)
        expected = expected_fn(*arrays)
        actual = func(*arrays)
        assert_arrays_close(actual, expected)


class TestRoundTrip:
    """Tests for func_to_program -> program_to_func round trip."""

    @pytest.mark.parametrize("fn,input_shapes,expected_source", ROUND_TRIP_CASES)
    def test_round_trip(self, fn, input_shapes, expected_source) -> None:
        """Verify func_to_program -> program_to_source/program_to_func round trip."""
        program = func_to_program(fn, input_shapes, np.float32)
        assert program_to_source(program) == expected_source
        func = program_to_func(program)
        arrays = _random_arrays(program.params, input_shapes)
        assert_arrays_close(func(*arrays), fn(*arrays))

    def test_program_hashable(self) -> None:
        """Verify GymPrograms can be used as dict keys."""
        p1 = func_to_program(_fn_matmul, {"a": (128, 128), "b": (128, 128)}, np.float32)
        p2 = func_to_program(_fn_matmul, {"a": (128, 256), "b": (128, 256)}, np.float32)
        d = {p1: "small", p2: "large"}
        assert len(d) == 2
        assert d[p1] == "small"
