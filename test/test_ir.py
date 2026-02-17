"""Tests for IR: source_to_program, program_to_source, and round trips."""

import numpy as np
import pytest
from golden_ir import F_ROUND_TRIP_CASES, P_ROUND_TRIP_CASES, PROGRAM_TO_SOURCE_CASES, SOURCE_TO_PROGRAM_CASES

from nkigym.ir import program_to_source, source_to_program
from nkigym.utils.source import source_to_callable


def _random_arrays(params: tuple[str, ...], input_shapes: dict[str, tuple[int, ...]]) -> list[np.ndarray]:
    """Generate random float32 arrays for each parameter."""
    rng = np.random.default_rng()
    return [rng.standard_normal(input_shapes[name]).astype(np.float32) for name in params]


class TestSourceToProgram:
    """Tests for source_to_program parsing nkigym source into GymProgram."""

    @pytest.mark.parametrize("source,input_shapes,dtype,expected", SOURCE_TO_PROGRAM_CASES)
    def test_parse(self, source, input_shapes, dtype, expected) -> None:
        """Verify source_to_program produces expected GymProgram."""
        actual = source_to_program(source, input_shapes, dtype)
        assert actual == expected

    def test_dtype_propagated(self) -> None:
        """Verify output_dtype is set correctly on GymProgram."""
        source = "def matmul(a, b):\n    return nkigym.nc_matmul(a, b)\n"
        program = source_to_program(source, {"a": (128, 128), "b": (128, 128)}, np.float64)
        assert program.output_dtype is np.float64

    def test_error_unknown_op(self) -> None:
        """Verify KeyError for unknown nkigym operation."""
        source = "def bad_fn(a):\n    return nkigym.unknown_op(a)\n"
        with pytest.raises(KeyError, match="Unknown op"):
            source_to_program(source, {"a": (128, 128)}, np.float32)


class TestProgramToSource:
    """Tests for program_to_source and source_to_callable with hand-crafted GymPrograms."""

    @pytest.mark.parametrize("program,expected_source,input_shapes,expected_fn", PROGRAM_TO_SOURCE_CASES)
    def test_source_and_execution(self, program, expected_source, input_shapes, expected_fn) -> None:
        """Verify source rendering matches expected string and execution is numerically correct."""
        actual_source = program_to_source(program)
        assert actual_source == expected_source
        func = source_to_callable(actual_source, program.name)
        assert hasattr(func, "__source__")
        arrays = _random_arrays(program.params, input_shapes)
        expected = expected_fn(*arrays)
        actual = func(*arrays)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


class TestRoundTrip:
    """Tests for source↔program round-trip stability."""

    @pytest.mark.parametrize("source,input_shapes", F_ROUND_TRIP_CASES)
    def test_source_round_trip(self, source, input_shapes) -> None:
        """Verify source survives one round trip: program_to_source(source_to_program(s)) == s."""
        program = source_to_program(source, input_shapes, np.float32)
        assert program_to_source(program) == source

    @pytest.mark.parametrize("program", P_ROUND_TRIP_CASES)
    def test_program_round_trip(self, program) -> None:
        """Verify program is stable: program_to_source → source_to_program gives back the same program."""
        source = program_to_source(program)
        result = source_to_program(source, dict(program.input_shapes), program.output_dtype)
        assert result == program

    def test_program_hashable(self) -> None:
        """Verify GymPrograms can be used as dict keys."""
        source = "def matmul(a, b):\n    return nkigym.nc_matmul(a, b)\n"
        p1 = source_to_program(source, {"a": (128, 128), "b": (128, 128)}, np.float32)
        p2 = source_to_program(source, {"a": (128, 256), "b": (128, 256)}, np.float32)
        d = {p1: "small", p2: "large"}
        assert len(d) == 2
        assert d[p1] == "small"
